import unittest

import torch

from diffsynth.core.gradient.gradient_checkpoint import gradient_checkpoint_forward


class ScaleAndBias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, value, bias):
        return value * self.scale + bias


class TestGradientCheckpointOffload(unittest.TestCase):
    def test_reentrant_offload_preserves_input_and_parameter_gradients(self):
        model = ScaleAndBias()
        value = torch.tensor(3.0, requires_grad=True)
        bias = torch.tensor(4.0, requires_grad=True)

        output = gradient_checkpoint_forward(
            model,
            True,
            True,
            value,
            bias,
            checkpoint_use_reentrant=True,
        )
        output.backward()

        self.assertEqual(value.grad.item(), 2.0)
        self.assertEqual(bias.grad.item(), 1.0)
        self.assertEqual(model.scale.grad.item(), 3.0)

    def test_reentrant_offload_supports_shared_explicit_gradient_input(self):
        model = ScaleAndBias()
        value = torch.tensor(3.0, requires_grad=True)
        condition_source = torch.tensor(4.0, requires_grad=True)
        condition = condition_source * 2.0

        for _ in range(2):
            value = gradient_checkpoint_forward(
                model,
                True,
                True,
                value,
                condition,
                checkpoint_use_reentrant=True,
            )
        value.backward()

        self.assertEqual(condition_source.grad.item(), 6.0)
        self.assertEqual(model.scale.grad.item(), 20.0)

    def test_reentrant_offload_supports_shared_captured_no_grad_input(self):
        model = ScaleAndBias()
        value = torch.tensor(3.0, requires_grad=True)
        condition = torch.tensor(4.0)

        for _ in range(2):
            value = gradient_checkpoint_forward(
                lambda checkpoint_value: model(checkpoint_value, condition),
                True,
                True,
                value,
                checkpoint_use_reentrant=True,
            )
        value.backward()

        self.assertEqual(model.scale.grad.item(), 16.0)


if __name__ == "__main__":
    unittest.main()
