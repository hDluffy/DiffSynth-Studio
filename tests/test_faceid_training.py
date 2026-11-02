import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from diffsynth.extensions.faceid.alignment import align_faces, estimate_alignment, TEMPLATE
from diffsynth.extensions.faceid.arcface import identity_embedding
from diffsynth.extensions.faceid.decoder import FaceIDDecoder, MEAN, STD
from diffsynth.extensions.faceid.loss import flow_match_faceid_loss
from diffsynth.models.wan_video_vae import VideoVAE_
from diffsynth.diffusion.logger import ModelLogger
from diffsynth.diffusion.loss import FlowMatchSFTLoss
from diffsynth.extensions.faceid.preprocess import FaceIDAnnotator


class TestFaceAlignment(unittest.TestCase):
    def test_template_identity_and_image_gradient(self):
        matrix = estimate_alignment(TEMPLATE)
        image = torch.rand(1, 3, 112, 112, requires_grad=True)
        aligned = align_faces(image, matrix.unsqueeze(0))
        torch.testing.assert_close(aligned, image, atol=2e-5, rtol=2e-5)
        aligned.mean().backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)

    def test_frozen_encoder_retains_input_gradient(self):
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 5)).eval().requires_grad_(False)
        image = torch.rand(1, 3, 2, 2, requires_grad=True)
        embedding = identity_embedding(encoder, image)
        (1 - embedding[0, 0]).backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in encoder.parameters()))
        torch.testing.assert_close(embedding.norm(dim=-1), torch.ones(1))


class TestCausalDecoder(unittest.TestCase):
    def test_window_matches_full_decode_and_truncates_only_prefix_gradient(self):
        torch.manual_seed(123)
        reference = VideoVAE_(dim=4).eval().requires_grad_(False)
        decoder = FaceIDDecoder(dim=4).eval().requires_grad_(False)
        decoder.load_state_dict({k: v for k, v in reference.state_dict().items()
                                 if k.startswith(('decoder.', 'conv2.'))})
        latents = torch.randn(1, 16, 3, 2, 2, requires_grad=True)
        with torch.no_grad():
            full = reference.decode(latents, [torch.tensor(MEAN), 1 / torch.tensor(STD)])
        # Test a late window and a second invocation with a different boundary:
        # caches must be local, not inherited from the previous decode.
        for chunk in (2, 1):
            window = decoder.decode_chunk(latents, chunk, save_on_cpu=True)
            torch.testing.assert_close(window, full[:, :, 4*chunk-3:4*chunk+1], rtol=2e-4, atol=2e-4)
        window.square().mean().backward()
        self.assertEqual(latents.grad[:, :, :1].abs().sum().item(), 0)
        self.assertGreater(latents.grad[:, :, 1:2].abs().sum().item(), 0)
        self.assertEqual(latents.grad[:, :, 2:].abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in decoder.parameters()))


class FakeScheduler:
    timesteps = torch.tensor([400.])
    sigmas = torch.tensor([.4])
    def add_noise(self, clean, noise, timestep): return clean * .6 + noise * .4
    def training_target(self, clean, noise, timestep): return noise-clean
    def training_weight(self, timestep): return torch.tensor(1.)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(.7), requires_grad=False)
    def decode_chunk(self, x, j, save_on_cpu=True):
        return (x[:, :3, j:j+1] * self.scale).repeat(1, 1, 4, 1, 1)


class TinyEncoder(nn.Module):
    def forward(self, faces): return faces.mean((-1, -2))


class TestFaceIDLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.parameter = nn.Parameter(torch.tensor(.2))
        self.pipe = SimpleNamespace(scheduler=FakeScheduler(), torch_dtype=torch.float32,
                                    device=torch.device('cpu'), in_iteration_models=[],
                                    model_fn=lambda **kw: self.parameter * (kw['latents'] +
                                        torch.arange(16).view(1, 16, 1, 1, 1).float()))
        self.aux = SimpleNamespace(decoder=TinyDecoder(), encoder=TinyEncoder(), dtype=torch.float32)
        self.config = SimpleNamespace(faceid_every=2, faceid_weight=.05, faceid_warmup_steps=0,
                                      faceid_sigma_min=.1, faceid_sigma_max=.5,
                                      faceid_frames=2, faceid_activation_offload=True)
        matrix = torch.tensor([[16/112, 0., 0.], [0., 16/112, 0.]])
        self.annotation = dict(valid=torch.zeros(81, dtype=torch.bool), reference_valid=True,
                               inverse_matrices=matrix.repeat(81, 1, 1),
                               reference_embedding=torch.tensor([1., 0., 0.]),
                               quality=torch.ones(81), height=16, width=16)
        self.annotation['valid'][2:5:2] = True
        self.inputs = dict(input_latents=torch.randn(1, 16, 21, 16, 16),
                           first_frame_latents=torch.randn(1, 16, 1, 16, 16))

    def run_loss(self, step=0):
        return flow_match_faceid_loss(self.pipe, self.aux, self.annotation, self.config, step, **self.inputs)

    def test_faceid_alone_backpropagates_to_trainable_parameter(self):
        # Compare against zero FaceID weight with identical noise to isolate its gradient.
        torch.manual_seed(17)
        result = self.run_loss()
        total_grad = torch.autograd.grad(result['loss'], self.parameter)[0]
        self.config.faceid_weight = 0.
        torch.manual_seed(17)
        baseline = self.run_loss()
        fm_grad = torch.autograd.grad(baseline['loss'], self.parameter)[0]
        self.assertTrue(torch.isfinite(total_grad))
        self.assertGreater((total_grad-fm_grad).abs().item(), 1e-7)
        self.assertEqual(result['metrics']['faceid/active'], 1.)
        self.assertEqual(result['metrics']['faceid/frames_used'], 2)
        self.assertIsNone(self.aux.decoder.scale.grad)

    def test_invalid_face_and_inactive_step_keep_fm_gradient(self):
        for step, reference_valid in ((1, True), (0, False)):
            self.annotation['reference_valid'] = reference_valid
            result = self.run_loss(step)
            self.assertEqual(result['metrics']['faceid/active'], 0.)
            self.assertTrue(torch.isfinite(torch.autograd.grad(result['loss'], self.parameter)[0]))

    def test_warmup_and_sigma_gate(self):
        self.config.faceid_warmup_steps = 100
        self.assertEqual(self.run_loss(0)['metrics']['faceid/active'], 0.)
        self.assertAlmostEqual(self.run_loss(50)['metrics']['faceid/weight'], .025)
        self.config.faceid_sigma_max = .2
        self.assertEqual(self.run_loss(100)['metrics']['faceid/active'], 0.)

    def test_disabled_identity_matches_existing_sft_loss(self):
        self.config.faceid_weight = 0.
        torch.manual_seed(57)
        actual = self.run_loss()['loss']
        torch.manual_seed(57)
        expected = FlowMatchSFTLoss(self.pipe, **self.inputs)
        torch.testing.assert_close(actual, expected)


class TestFaceCache(unittest.TestCase):
    def test_candidate_indices_and_missing_reference(self):
        from PIL import Image
        annotator = object.__new__(FaceIDAnnotator)
        annotator.config = SimpleNamespace(faceid_min_similarity=.2)
        annotator.fingerprint = 'encoder'
        annotator.detector_fingerprint = 'detector'
        embedding = torch.nn.functional.normalize(torch.ones(512), dim=0)
        annotator.face = lambda image: (torch.eye(2, 3), embedding, .9)
        frames = [Image.new('RGB', (64, 64)) for _ in range(81)]
        annotation = annotator(frames, frames[0])
        self.assertEqual(annotation['valid'].sum().item(), 40)
        self.assertFalse(annotation['valid'][0])
        self.assertEqual(annotation['valid'].nonzero().flatten().tolist(), list(range(2, 81, 2)))
        annotator.face = lambda image: None
        annotation = annotator(frames, frames[0])
        self.assertFalse(annotation['reference_valid'])
        self.assertFalse(annotation['valid'].any())


class TestMetricLogging(unittest.TestCase):
    def test_tensorboard_events_include_reduced_losses(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        accelerator = SimpleNamespace(is_main_process=True, device=torch.device('cpu'),
                                      reduce=lambda x, reduction: x)
        with tempfile.TemporaryDirectory() as path:
            logger = ModelLogger(path, enable_tensorboard_log=True)
            logger.on_step_end(accelerator, None, loss=torch.tensor(1.), metrics={
                'loss/fm': torch.tensor(.9), 'loss/faceid': torch.tensor(.2),
                'loss/total': torch.tensor(1.), 'faceid/cosine': .8, 'faceid/active': 1.,
            })
            logger.loggers[0].close()
            events = EventAccumulator(str(Path(path)/'tensorboard_log')).Reload()
            self.assertIn('loss/fm', events.Tags()['scalars'])
            self.assertAlmostEqual(events.Scalars('faceid/cosine_when_active')[0].value, .8, places=5)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
