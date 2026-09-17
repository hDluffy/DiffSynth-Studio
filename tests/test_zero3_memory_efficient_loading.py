import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from diffsynth.core.loader.model import load_state_dict_for_zero3


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(2, 3))
        self.register_buffer("running", torch.zeros(2))


class TestZero3MemoryEfficientLoading(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self.temp_dir.name) / "model.safetensors"
        save_file(
            {
                "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
                "running": torch.tensor([4.0, 5.0]),
            },
            self.checkpoint,
        )
        self.model = ToyModel()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _load_as_rank(self, rank, key_prefix=None):
        with (
            patch("torch.distributed.is_available", return_value=True),
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=rank),
        ):
            return load_state_dict_for_zero3(
                self.model, str(self.checkpoint), key_prefix=key_prefix
            )

    def test_rank_zero_materializes_parameters_and_buffers(self):
        state_dict = self._load_as_rank(0)
        self.assertTrue(torch.equal(state_dict["weight"], torch.arange(6).reshape(2, 3)))
        self.assertTrue(torch.equal(state_dict["running"], torch.tensor([4.0, 5.0])))

    def test_nonzero_rank_materializes_buffers_only(self):
        state_dict = self._load_as_rank(3)
        self.assertIsNone(state_dict["weight"])
        self.assertTrue(torch.equal(state_dict["running"], torch.tensor([4.0, 5.0])))

    def test_nonzero_rank_preserves_prefixed_parameter_keys(self):
        state_dict = self._load_as_rank(2, key_prefix="pipe.dit.")
        self.assertEqual(set(state_dict), {"pipe.dit.weight", "pipe.dit.running"})
        self.assertIsNone(state_dict["pipe.dit.weight"])
        self.assertIsNone(state_dict["pipe.dit.running"])

    def test_uninitialized_distributed_falls_back_to_full_load(self):
        with patch("torch.distributed.is_initialized", return_value=False):
            state_dict = load_state_dict_for_zero3(self.model, str(self.checkpoint))
        self.assertTrue(torch.equal(state_dict["weight"], torch.arange(6).reshape(2, 3)))


if __name__ == "__main__":
    unittest.main()
