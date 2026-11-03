"""Training timestep-index sampling, filtered by the scheduler's effective sigma.

The interval distribution is uniform over eligible ORIGINAL indices, not over
continuous sigma values. BF16 timestep rounding and nearest-index ties follow
scheduler.add_noise exactly. No importance correction is applied to FM loss.
"""
import torch


class FaceIDSigmaSampler:
    def __init__(self, scheduler, dtype, config, min_boundary=0., max_boundary=1.):
        self.scheduler = scheduler
        self.dtype = dtype
        self.mode = getattr(config, 'sigma_sampling_mode', 'original')
        self.prob = getattr(config, 'sigma_sampling_prob', .5)
        lower = getattr(config, 'sigma_sampling_min', None)
        upper = getattr(config, 'sigma_sampling_max', None)
        self.lower = config.faceid_sigma_min if lower is None else lower
        self.upper = config.faceid_sigma_max if upper is None else upper
        if self.mode not in ('original', 'interval', 'mixed'):
            raise ValueError(f'Unknown sigma sampling mode: {self.mode}')
        if not 0 <= self.prob <= 1 or not 0 <= self.lower < self.upper <= 1:
            raise ValueError('Require sampling probability in [0,1] and 0 <= sigma min < max <= 1')
        if not 0 <= min_boundary < max_boundary <= 1:
            raise ValueError('Require 0 <= min_timestep_boundary < max_timestep_boundary <= 1')
        self.minimum = int(min_boundary * len(scheduler.timesteps))
        self.maximum = int(max_boundary * len(scheduler.timesteps))
        if self.minimum >= self.maximum:
            raise ValueError('Timestep boundaries contain no scheduler indices')
        self.interval_probability = 1. if self.mode == 'interval' else self.prob if self.mode == 'mixed' else 0.
        self.candidates = None
        if self.interval_probability > 0:
            timesteps = scheduler.timesteps.detach().cpu()
            original = torch.arange(self.minimum, self.maximum)
            rounded = timesteps[original].to(dtype).to(timesteps.dtype)
            nearest = (timesteps[None, :] - rounded[:, None]).abs().argmin(dim=1)
            effective = scheduler.sigmas.detach().cpu()[nearest].float()
            self.candidates = original[(effective >= self.lower) & (effective <= self.upper)]
            if not len(self.candidates):
                raise ValueError(f'No effective scheduler sigmas in [{self.lower}, {self.upper}] '
                                 'after timestep rounding and timestep-boundary filtering')

    def sample(self, device):
        focused = self.interval_probability == 1 or (
            self.interval_probability > 0 and torch.rand(()).item() < self.interval_probability)
        if focused:
            index = self.candidates[torch.randint(len(self.candidates), (1,))]
        else:
            # Keep this identical to legacy sampling, without an extra RNG draw.
            index = torch.randint(self.minimum, self.maximum, (1,))
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=device)
        nearest = torch.argmin((self.scheduler.timesteps - timestep.to(self.scheduler.timesteps.device)).abs())
        sigma = self.scheduler.sigmas[nearest].to(device=device, dtype=torch.float32)
        return timestep, sigma, focused
