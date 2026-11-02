"""Two-rank GPU integration smoke test (small synthetic data, no external weights).

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 tests/faceid_zero3_smoke.py
Tests real PEFT LoRA, ZeRO-3 backward, unequal FaceID masks and gathered export.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from accelerate import Accelerator, DeepSpeedPlugin
from peft import LoraConfig, inject_adapter_in_model
from diffsynth.extensions.faceid.loss import flow_match_faceid_loss
from test_faceid_training import FakeScheduler, TinyDecoder, TinyEncoder


class ToyS2V(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(16, 16, bias=False)
        self.requires_grad_(False)
        inject_adapter_in_model(LoraConfig(r=2, lora_alpha=2, target_modules=['proj']), self)
        self.aux = SimpleNamespace(decoder=TinyDecoder(), encoder=TinyEncoder(), dtype=torch.float32)
        self.config = SimpleNamespace(faceid_every=1, faceid_weight=.1, faceid_warmup_steps=0,
                                      faceid_sigma_min=.1, faceid_sigma_max=.5,
                                      faceid_frames=2, faceid_activation_offload=True)

    def predict(self, latents, **kwargs):
        return self.proj(latents.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

    def forward(self, value, annotation):
        pipe = SimpleNamespace(scheduler=FakeScheduler(), torch_dtype=torch.float32,
                               device=value.device, in_iteration_models=[], model_fn=self.predict)
        return flow_match_faceid_loss(pipe, self.aux, annotation, self.config, 1,
                                     input_latents=value, first_frame_latents=value[:, :, :1])


def main():
    plugin = DeepSpeedPlugin(hf_ds_config={
        'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
        'zero_optimization': {'stage': 3, 'stage3_gather_16bit_weights_on_model_save': True},
        'fp16': {'enabled': False}, 'bf16': {'enabled': False},
    }, zero3_init_flag=False)
    accelerator = Accelerator(deepspeed_plugin=plugin)
    model = ToyS2V()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model, optimizer = accelerator.prepare(model, optimizer)
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.aux.decoder.to(accelerator.device)
    value = torch.randn(1, 16, 21, 16, 16, device=accelerator.device)
    annotation = dict(valid=torch.zeros(81, dtype=torch.bool),
                      reference_valid=accelerator.process_index == 0,
                      inverse_matrices=torch.tensor([[16/112, 0., 0.], [0., 16/112, 0.]]).repeat(81, 1, 1),
                      reference_embedding=torch.tensor([1., 0., 0.]), quality=torch.ones(81), height=16, width=16)
    annotation['valid'][2:5:2] = True
    for _ in range(2):
        result = model(value, annotation)
        assert result['metrics']['faceid/active'] == float(accelerator.process_index == 0)
        assert torch.isfinite(result['loss'])
        accelerator.backward(result['loss'])
        optimizer.step()
        optimizer.zero_grad()
        reduced = accelerator.reduce(result['metrics']['loss/total'], reduction='mean')
        assert torch.isfinite(reduced)
    # Gathered parameters must have real shapes, and LoRA B must have updated.
    state = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        lora = {k: v for k, v in state.items() if 'lora_' in k}
        assert len(lora) == 2
        assert any(v.abs().sum() > 0 for k, v in lora.items() if 'lora_B' in k)
        with tempfile.TemporaryDirectory() as directory:
            from safetensors.torch import save_file, load_file
            path = str(Path(directory)/'smoke.safetensors')
            save_file(lora, path)
            restored = load_file(path)
            for key in lora:
                torch.testing.assert_close(lora[key].cpu(), restored[key])
        print('PASS: two-rank ZeRO-3 + PEFT LoRA + unequal FaceID branches + gathered save/reload')
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    main()
