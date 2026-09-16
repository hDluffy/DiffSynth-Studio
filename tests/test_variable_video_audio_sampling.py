import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import (
    AlignAudioToVideo,
    FrameSamplerByRateMixin,
    VideoFrames,
)
from diffsynth.diffusion.runner import launch_data_process_task


class FakeReader:
    def __init__(self, num_frames, frame_rate):
        self.num_frames = num_frames
        self.frame_rate = frame_rate

    def count_frames(self):
        return self.num_frames

    def get_meta_data(self):
        return {"fps": self.frame_rate}


class VariableFrameSamplingTest(unittest.TestCase):
    def make_sampler(self, **kwargs):
        config = dict(
            num_frames=113,
            frame_rate=16,
            fix_frame_rate=True,
            frame_count_stride=16,
            frame_count_remainder=1,
            frame_count_rounding="nearest",
            min_num_frames=81,
            max_frame_padding=8,
        )
        config.update(kwargs)
        return FrameSamplerByRateMixin(**config)

    def test_nearest_rounding_uses_16n_plus_1(self):
        sampler = self.make_sampler()
        plan = sampler.get_sampling_plan(FakeReader(150, 30))
        self.assertEqual(plan.available_num_frames, 80)
        self.assertEqual(plan.selected_num_frames, 81)
        self.assertEqual(plan.padded_num_frames, 1)

        self.assertEqual(sampler.select_num_frames(89), 81)  # tie: truncate
        self.assertEqual(sampler.select_num_frames(90), 97)
        self.assertEqual(sampler.select_num_frames(110), 113)
        self.assertEqual(sampler.select_num_frames(140), 113)

    def test_excessive_padding_is_rejected(self):
        sampler = self.make_sampler()
        with self.assertRaisesRegex(ValueError, "max_frame_padding"):
            sampler.select_num_frames(70)

    def test_invalid_constraint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame_count_remainder"):
            self.make_sampler(frame_count_remainder=16)
        with self.assertRaisesRegex(ValueError, "frame_count_stride"):
            self.make_sampler(frame_count_stride=0)


class AudioVideoAlignmentTest(unittest.TestCase):
    def make_data(self, frames, audio_samples):
        return {
            "video": VideoFrames([object()] * frames, {"padded_num_frames": 0}),
            "input_audio": np.ones(audio_samples, dtype=np.float32),
            "_data_cache_key": "sample",
        }

    def test_audio_is_trimmed_to_video_timestamps(self):
        aligner = AlignAudioToVideo(frame_rate=16, sample_rate=16000, log_first_n=0)
        data = aligner(self.make_data(97, 100000))
        self.assertEqual(data["input_audio"].shape[-1], 96000)
        self.assertEqual(data["audio_num_samples"], 96000)
        self.assertEqual(data["sample_duration_seconds"], 6.0)

    def test_short_audio_is_zero_padded(self):
        aligner = AlignAudioToVideo(
            frame_rate=16, sample_rate=16000, max_padding_seconds=0.5, log_first_n=0
        )
        data = aligner(self.make_data(81, 76000))
        self.assertEqual(data["input_audio"].shape[-1], 80000)
        np.testing.assert_array_equal(data["input_audio"][-4000:], 0)

    def test_excessively_short_audio_is_rejected(self):
        aligner = AlignAudioToVideo(
            frame_rate=16, sample_rate=16000, max_padding_seconds=0.5, log_first_n=0
        )
        with self.assertRaisesRegex(ValueError, "max_audio_padding_seconds"):
            aligner(self.make_data(81, 70000))

    def test_strict_policy_rejects_duration_mismatch(self):
        aligner = AlignAudioToVideo(
            frame_rate=16,
            sample_rate=16000,
            policy="strict",
            tolerance_seconds=0.05,
            log_first_n=0,
        )
        with self.assertRaisesRegex(ValueError, "duration mismatch"):
            aligner(self.make_data(81, 78000))


class CacheManifestTest(unittest.TestCase):
    def test_complete_manifest_is_validated(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir)
            torch.save(({"sample_num_frames": 81}, {}, {}), cache_path / "sample.pth")
            (cache_path / "_cache_manifest.json").write_text(json.dumps({
                "status": "complete",
                "cached_files": 1,
            }))
            dataset = UnifiedDataset(
                base_path=tmp_dir,
                metadata_path=None,
                cache_manifest_required=True,
            )
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0][0]["sample_num_frames"], 81)

    def test_manifest_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir)
            torch.save(({}, {}, {}), cache_path / "sample.pth")
            (cache_path / "_cache_manifest.json").write_text(json.dumps({
                "status": "complete",
                "cached_files": 2,
            }))
            with self.assertRaisesRegex(RuntimeError, "file count mismatch"):
                UnifiedDataset(
                    base_path=tmp_dir,
                    metadata_path=None,
                    cache_manifest_required=True,
                )


class FeatureCacheGenerationTest(unittest.TestCase):
    class Dataset(torch.utils.data.Dataset):
        repeat = 1
        cache_key_field = "_data_cache_key"
        metadata_path = None

        def __len__(self):
            return 3

        def __getitem__(self, index):
            return {
                "_data_cache_key": f"sample_{index}",
                "sample_num_frames": (81, 97, 113)[index],
                "value": torch.tensor([index], dtype=torch.float32),
            }

    class Model(torch.nn.Module):
        def forward(self, data):
            return (
                {
                    "input_latents": data["value"],
                    "sample_num_frames": data["sample_num_frames"],
                },
                {},
                {},
            )

    class Accelerator:
        num_processes = 1
        process_index = 0
        is_main_process = True
        device = torch.device("cpu")
        even_batches = True

        def prepare(self, value):
            return value

        def wait_for_everyone(self):
            pass

        def print(self, *args, **kwargs):
            pass

    def test_cache_generation_writes_complete_manifest(self):
        args = SimpleNamespace(
            dataset_num_workers=0,
            enable_model_cpu_offload=False,
            enable_optimizer_cpu_offload=False,
            cpu_offload_split_threshold=None,
            resume_feature_cache=False,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            accelerator = self.Accelerator()
            launch_data_process_task(
                accelerator,
                self.Dataset(),
                self.Model(),
                SimpleNamespace(output_path=tmp_dir),
                args=args,
            )
            manifest = json.loads(
                (Path(tmp_dir) / "_cache_manifest.json").read_text()
            )
            self.assertFalse(accelerator.even_batches)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["source_items"], 3)
            self.assertEqual(manifest["cached_files"], 3)
            self.assertEqual(
                manifest["frame_count_distribution"],
                {"81": 1, "97": 1, "113": 1},
            )


if __name__ == "__main__":
    unittest.main()
