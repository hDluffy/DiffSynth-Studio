import unittest
from types import SimpleNamespace
import torch
from diffsynth.diffusion.flow_match import FlowMatchScheduler
from diffsynth.extensions.faceid.sampling import FaceIDSigmaSampler


class TestSigmaSampling(unittest.TestCase):
    def setUp(self):
        self.scheduler = FlowMatchScheduler(template='Wan')
        self.scheduler.set_timesteps(1000, training=True)

    def sampler(self, mode='original', prob=.5, lower=.1, upper=.5, boundaries=(0., 1.)):
        cfg = SimpleNamespace(faceid_sigma_min=lower, faceid_sigma_max=upper,
                              sigma_sampling_mode=mode, sigma_sampling_prob=prob)
        return FaceIDSigmaSampler(self.scheduler, torch.bfloat16, cfg, *boundaries)

    def test_original_preserves_legacy_rng_and_boundary(self):
        sampler = self.sampler(boundaries=(.2, .8))
        torch.manual_seed(174)
        expected = [self.scheduler.timesteps[torch.randint(200, 800, (1,))].to(torch.bfloat16) for _ in range(100)]
        next_random = torch.rand(3)
        torch.manual_seed(174)
        for timestep in expected:
            actual, _, focused = sampler.sample('cpu')
            torch.testing.assert_close(actual, timestep)
            self.assertFalse(focused)
        torch.testing.assert_close(torch.rand(3), next_random)

    def test_interval_uses_actual_noise_sigma_including_bf16_rounding(self):
        sampler = self.sampler('interval')
        # Exhaustively verify every allowed original index against scheduler.add_noise.
        for index in sampler.candidates:
            timestep = self.scheduler.timesteps[index:index+1].to(torch.bfloat16)
            actual_sigma = self.scheduler.add_noise(torch.tensor(0.), torch.tensor(1.), timestep).item()
            self.assertGreaterEqual(actual_sigma, .1)
            self.assertLessEqual(actual_sigma, .5)
        for _ in range(100):
            timestep, sigma, focused = sampler.sample('cpu')
            self.assertTrue(focused)
            actual = self.scheduler.add_noise(torch.tensor(0.), torch.tensor(1.), timestep)
            torch.testing.assert_close(sigma, actual)

    def test_mixture_branch_and_hit_frequency(self):
        sampler = self.sampler('mixed', prob=.75)
        torch.manual_seed(19)
        focused = hits = 0
        count = 6000
        for _ in range(count):
            _, sigma, branch = sampler.sample('cpu')
            focused += branch
            hits += .1 <= sigma.item() <= .5
        original_hit = len(sampler.candidates) / 1000
        self.assertAlmostEqual(focused/count, .75, delta=.025)
        self.assertAlmostEqual(hits/count, .75+.25*original_hit, delta=.025)

    def test_zero_probability_matches_original_and_one_matches_interval(self):
        for mode, probability in [('original', 0.), ('interval', 1.)]:
            a, b = self.sampler(mode), self.sampler('mixed', prob=probability)
            torch.manual_seed(25)
            expected = [a.sample('cpu')[:2] for _ in range(30)]
            torch.manual_seed(25)
            for t, sigma in expected:
                actual_t, actual_sigma, _ = b.sample('cpu')
                torch.testing.assert_close(t, actual_t)
                torch.testing.assert_close(sigma, actual_sigma)

    def test_empty_intersection_fails_instead_of_silent_fallback(self):
        with self.assertRaisesRegex(ValueError, 'No effective'):
            self.sampler('interval', boundaries=(0., .01))
        with self.assertRaisesRegex(ValueError, 'no scheduler indices'):
            self.sampler(boundaries=(0., .0001))
        # No need for an interval intersection when preserving original sampling.
        self.sampler('mixed', prob=0., boundaries=(0., .01))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
