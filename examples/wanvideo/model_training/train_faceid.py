"""Opt-in 81-frame single-identity S2V training. See faceid/README.md."""
import argparse
import contextlib
import functools
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from train import WanTrainingModule, wan_parser, main
from diffsynth.extensions.faceid.arcface import file_sha256, load_arcface
from diffsynth.extensions.faceid.decoder import FaceIDDecoder
from diffsynth.extensions.faceid.preprocess import FaceIDAnnotator, CACHE_VERSION
from diffsynth.extensions.faceid.loss import flow_match_faceid_loss


class FaceIDTrainingModule(WanTrainingModule):
    def __init__(self, *args, faceid_config, **kwargs):
        from transformers.integrations import is_deepspeed_zero3_enabled
        if kwargs.get('task') == 'sft:data_process' and is_deepspeed_zero3_enabled():
            raise ValueError('Extract FaceID caches with the data_process config, not ZeRO-3')
        super().__init__(*args, **kwargs)
        self.faceid_config = faceid_config
        # Plain containers intentionally keep teacher parameters outside the
        # trainable module, DeepSpeed engine, optimizer and exported checkpoint.
        self.faceid_auxiliary = None
        self.faceid_annotator = None
        self.faceid_optimizer_step = 0

    def get_pipeline_inputs(self, data):
        inputs = super().get_pipeline_inputs(data)
        reference = inputs[0].get('input_image')
        if isinstance(reference, str):
            from PIL import Image
            path = Path(reference)
            if not path.is_absolute():
                path = Path(self.faceid_config.dataset_base_path) / path
            with Image.open(path) as image:
                inputs[0]['input_image'] = image.convert('RGB')
        return inputs

    def setup_training_auxiliaries(self, accelerator):
        plugin = getattr(accelerator.state, 'deepspeed_plugin', None)
        context = plugin.zero3_init_context_manager(enable=False) if plugin else contextlib.nullcontext()
        with context:
            decoder = FaceIDDecoder.from_checkpoint(self.faceid_config.faceid_vae_path, accelerator.device)
            encoder = load_arcface(self.faceid_config.faceid_weights, accelerator.device)
        for module in (decoder, encoder):
            if any(hasattr(p, 'ds_id') for p in module.parameters()):
                raise RuntimeError('FaceID teacher was unexpectedly partitioned by ZeRO-3')
            assert not any(p.requires_grad for p in module.parameters())
        self.faceid_auxiliary = SimpleNamespace(decoder=decoder, encoder=encoder, dtype=torch.bfloat16)
        self.pipe.device = accelerator.device
        if accelerator.is_main_process:
            output = Path(self.faceid_config.output_path)
            output.mkdir(parents=True, exist_ok=True)
            (output / 'faceid_training_config.json').write_text(
                json.dumps(vars(self.faceid_config), indent=2, ensure_ascii=False) + '\n'
            )
            print('FaceID teachers ready: frozen local Wan decoder + FP32 ArcFace iResNet100; '
                  'excluded from optimizer and saved LoRA.')

    def set_optimizer_step(self, step):
        self.faceid_optimizer_step = step

    def forward(self, data, inputs=None):
        if self.task == 'sft:data_process':
            if self.faceid_annotator is None:
                self.faceid_annotator = FaceIDAnnotator(self.faceid_config, self.pipe.device)
            shared = self.get_pipeline_inputs(data)[0]
            annotation = self.faceid_annotator(data['video'], shared['input_image'])
            payload = super().forward(data, inputs)
            # Attach AFTER GeneralUnit_RemoveCache, which prunes unknown fields.
            payload[0]['faceid'] = annotation
            return payload
        if self.faceid_auxiliary is None:
            raise RuntimeError('FaceID teachers must be initialized after accelerator.prepare')
        if inputs is None or 'faceid' not in inputs[0]:
            raise ValueError('FaceID training requires an annotated sft:train cache; rebuild it with train_faceid.py')
        shared, positive, negative = inputs
        annotation = shared['faceid']
        if annotation['version'] != CACHE_VERSION or annotation['model_sha256'] != self.faceid_config.faceid_model_sha256:
            raise ValueError('FaceID cache/model fingerprint mismatch; rebuild the cache')
        if annotation['num_frames'] != 81:
            raise ValueError('FaceID v1 only supports 81 frames')
        # Keep alignment matrices and embeddings FP32, masks bool and indices int.
        inputs = ({k: v for k, v in shared.items() if k != 'faceid'}, positive, negative)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        inputs[0]['use_gradient_checkpointing'] = self.use_gradient_checkpointing
        inputs[0]['use_gradient_checkpointing_offload'] = self.use_gradient_checkpointing_offload
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        result = flow_match_faceid_loss(self.pipe, self.faceid_auxiliary, annotation,
                                       self.faceid_config, self.faceid_optimizer_step,
                                       **{**inputs[0], **inputs[1]})
        if torch.cuda.is_available():
            result['metrics']['memory/cuda_allocated_gib'] = torch.cuda.memory_allocated(self.pipe.device) / 2**30
            result['metrics']['memory/cuda_peak_allocated_gib'] = torch.cuda.max_memory_allocated(self.pipe.device) / 2**30
        return result


def build_parser():
    parser = wan_parser()
    parser.add_argument('--faceid_weights', required=True, help='Local arcface_torch iResNet100 backbone.pth')
    parser.add_argument('--faceid_vae_path', required=True, help='Local Wan2.1 VAE checkpoint used to create the cache')
    parser.add_argument('--faceid_detector_path', help='Local SCRFD detector ONNX (e.g. det_10g.onnx), preprocessing only')
    parser.add_argument('--faceid_weight', type=float, default=.05)
    parser.add_argument('--faceid_warmup_steps', type=int, default=100)
    parser.add_argument('--faceid_every', type=int, default=2, help='Optimizer-step interval')
    parser.add_argument('--faceid_frames', type=int, default=2, choices=(1, 2))
    parser.add_argument('--faceid_sigma_min', type=float, default=.1)
    parser.add_argument('--faceid_sigma_max', type=float, default=.5)
    parser.add_argument('--faceid_activation_offload', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--faceid_det_threshold', type=float, default=.6)
    parser.add_argument('--faceid_min_face_size', type=int, default=48)
    parser.add_argument('--faceid_min_similarity', type=float, default=.2)
    return parser


def validate_config(parser, args):
    if args.task not in ('sft:data_process', 'sft:train'):
        parser.error('Use sft:data_process followed by sft:train; raw-data FaceID training is not supported')
    if args.num_frames != 81 or args.framewise_decoding:
        parser.error('FaceID v1 requires --num_frames 81 and the temporal Wan2.1 VAE')
    if args.enable_model_cpu_offload or args.offload_models or args.fp8_models:
        parser.error('FaceID v1 uses DeepSpeed parameter management; do not combine custom/FP8 offloading')
    if args.trainable_models or args.lora_base_model != 'dit':
        parser.error('Use --lora_base_model dit without --trainable_models (LoRA-only)')
    if args.faceid_weight < 0 or args.faceid_every < 1 or args.faceid_warmup_steps < 0:
        parser.error('Invalid FaceID weight, interval or warmup')
    if not 0 <= args.faceid_sigma_min < args.faceid_sigma_max <= 1:
        parser.error('Require 0 <= faceid_sigma_min < faceid_sigma_max <= 1')
    if not 0 < args.faceid_det_threshold < 1 or args.faceid_min_face_size <= 0 or not -1 <= args.faceid_min_similarity <= 1:
        parser.error('Invalid face detection/identity quality thresholds')
    for path in (args.faceid_weights, args.faceid_vae_path):
        if not Path(path).is_file():
            parser.error(f'Weight file not found: {path}')
    args.faceid_model_sha256 = file_sha256(args.faceid_weights)
    vae_sha = file_sha256(args.faceid_vae_path)
    args.faceid_vae_sha256 = vae_sha
    rope_settings = {key: getattr(args, key) for key in (
        's2v_ref_rope_mode', 's2v_ref_source_id', 's2v_ref_rope_theta',
        's2v_ref_time_base', 's2v_ref_time_margin')}
    if args.task == 'sft:data_process':
        if args.min_num_frames != 81:
            parser.error('Cache extraction requires --min_num_frames 81')
        if 'input_image' not in (args.extra_inputs or '').split(','):
            parser.error('Include input_image in --extra_inputs')
        if not args.faceid_detector_path or not Path(args.faceid_detector_path).is_file():
            parser.error('Cache extraction requires a local --faceid_detector_path')
        # Require the exact local VAE in the pipeline's model list too.
        if not args.model_paths or str(Path(args.faceid_vae_path)) not in json.loads(args.model_paths):
            parser.error('Include faceid_vae_path in the --model_paths JSON list during cache extraction')
        signature = dict(version=CACHE_VERSION, model_sha256=args.faceid_model_sha256, vae_sha256=vae_sha,
                         detector_sha256=file_sha256(args.faceid_detector_path),
                         det_threshold=args.faceid_det_threshold, min_face_size=args.faceid_min_face_size,
                         min_similarity=args.faceid_min_similarity, alignment='arcface112_rgb_grid_v1',
                         rope=rope_settings)
        parser.set_defaults(faceid_cache_signature=signature)
    else:
        if args.dataset_metadata_path is not None:
            parser.error('sft:train must read a cache directory without --dataset_metadata_path')
        manifest_path = Path(args.dataset_base_path) / '_cache_manifest.json'
        if not manifest_path.is_file():
            parser.error(f'Missing cache manifest: {manifest_path}')
        manifest = json.loads(manifest_path.read_text())
        signature = manifest.get('config', {}).get('faceid', {})
        if manifest.get('status') != 'complete' or signature.get('version') != CACHE_VERSION:
            parser.error('Cache is incomplete or lacks FaceID annotations')
        if signature.get('model_sha256') != args.faceid_model_sha256 or signature.get('vae_sha256') != vae_sha:
            parser.error('ArcFace/VAE weights differ from those used for the feature cache')
        if signature.get('rope') != rope_settings:
            parser.error('Training RoPE settings differ from FaceID feature cache')


if __name__ == '__main__':
    parser = build_parser()
    config = parser.parse_args()
    validate_config(parser, config)
    main(parser, functools.partial(FaceIDTrainingModule, faceid_config=config))
