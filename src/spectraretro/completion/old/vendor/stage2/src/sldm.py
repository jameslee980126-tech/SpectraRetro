"""SLDM coordinate process with the official GDiffRetro VP element process.

The EGNN, element noise-MSE, masks and fragment COM are unchanged. This is a
conditional SLDM adaptation, not upstream DGAP/atom-decoder reproduction.
See README_SLDM.md for equations, endpoint approximation and provenance.
"""
import math

import torch

from src.edm import EDM


class SLDM(EDM):
    VERSION = 4  # v4 restores the VP element transition from t=1.

    def __init__(self, *args, sigma_min=0.05, temperature_power=0.5, **kwargs):
        if kwargs.get('loss_type', 'vlb') != 'l2':
            raise ValueError('This SLDM adaptation supports noise MSE only, not a VP VLB')
        if kwargs.get('noise_schedule', 'learned') == 'learned':
            raise ValueError('The element process requires the official predefined VP schedule')
        for value, name in ((sigma_min, 'sigma_min'), (temperature_power, 'temperature_power')):
            if not math.isfinite(value) or value < 0 or (name == 'sigma_min' and value == 0):
                raise ValueError(f'Invalid SLDM {name}')
        super().__init__(*args, **kwargs)
        self.sigma_min = float(sigma_min)
        self.temperature_power = float(temperature_power)
        self.training_steps = self.T
        self.last_sample_nfe = 0
        self.register_buffer('_sldm_version', torch.tensor(self.VERSION))
        # Sampling budget may change, but the training process must not silently change.
        self.register_buffer('_sldm_contract', torch.tensor([
            self.sigma_min, self.temperature_power, self.training_steps,
            self.n_dims, self.in_node_nf, *self.norm_values,
        ], dtype=torch.float64))
        self.validate_sampling()

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        for key in ('_sldm_version', '_sldm_contract'):
            value = state_dict.get(prefix + key)
            if value is None or not torch.equal(value.cpu(), getattr(self, key).cpu()):
                raise RuntimeError('Incompatible SLDM checkpoint/process; use a verified explicit migration or retrain')
        schedule = state_dict.get(prefix + 'gamma.gamma')
        if schedule is None or not torch.equal(schedule.cpu(), self.gamma.gamma.cpu()):
            raise RuntimeError('Checkpoint element noise schedule differs from configuration')
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def validate_sampling(self):
        if isinstance(self.T, bool) or not isinstance(self.T, int) or not 2 <= self.T <= self.training_steps:
            raise ValueError('SLDM sampling steps must be an integer in [2, training diffusion_steps]')

    @staticmethod
    def validate_masks(node_mask, fragment_mask, linker_mask):
        for mask in (node_mask, fragment_mask, linker_mask):
            if mask.ndim != 3 or mask.shape[-1] != 1 or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError('Expected binary masks [batch, nodes, 1]')
        if fragment_mask.shape != node_mask.shape or linker_mask.shape != node_mask.shape:
            raise ValueError('Mask shapes differ')
        if not torch.equal(fragment_mask.float() + linker_mask.float(), node_mask.float()):
            raise ValueError('Fixed and generated atoms must partition the node mask')
        if torch.any(fragment_mask.sum((1, 2)) == 0):
            raise ValueError('Every example requires a fixed synthon')

    def noise_state(self, xh, eps, t, fragment_mask, linker_mask):
        gamma = self.gamma(t)
        x = (1 - t.unsqueeze(-1)) * xh[..., :self.n_dims] + self.sigma_min * eps[..., :self.n_dims]
        h = self.alpha(gamma, xh) * xh[..., self.n_dims:] + self.sigma(gamma, xh) * eps[..., self.n_dims:]
        return xh * fragment_mask + torch.cat((x, h), -1) * linker_mask

    def forward(self, x, h, node_mask, fragment_mask, linker_mask, edge_mask, context=None):
        self.validate_masks(node_mask, fragment_mask, linker_mask)
        x, h = self.normalize(x, h)
        xh = torch.cat((x, h), -1)
        # Same uniform discrete training time distribution as official EDM.
        t = torch.randint(0, self.training_steps + 1, (x.shape[0], 1), device=x.device).float() / self.training_steps
        eps = self.sample_combined_position_feature_noise(x.shape[0], x.shape[1], linker_mask)
        z = self.noise_state(xh, eps, t, fragment_mask, linker_mask)
        prediction = self.dynamics(t=t, xh=z, node_mask=node_mask, linker_mask=linker_mask,
                                   edge_mask=edge_mask, context=context) * linker_mask
        # Preserve official joint noise-MSE weighting and treatment of empty linkers.
        width = self.n_dims + self.in_node_nf
        denominator = width * self.numbers_of_nodes(linker_mask).clamp_min(1)
        loss = (self.sum_except_batch((prediction - eps).square()) / denominator).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite SLDM training loss')
        zero = loss.new_zeros(())
        # Interface placeholders only; DDPM does not report these as SLDM likelihoods.
        return zero, zero, zero, zero, loss, torch.norm(prediction, dim=(1, 2)).mean(), zero

    def coordinate_mean(self, z_x, eps_x, s, t):
        if torch.any(t >= 1):
            raise ValueError('SLDM reverse transition is singular at t=1')
        return ((1 - s) / (1 - t)).unsqueeze(-1) * (z_x - self.sigma_min * eps_x)

    def element_transition(self, z_h, eps_h, s, t, noise):
        # Exactly the official VP posterior, restricted to element channels.
        gamma_s, gamma_t = self.gamma(s), self.gamma(t)
        variance, std, alpha = self.sigma_and_alpha_t_given_s(gamma_t, gamma_s, z_h)
        sigma_s, sigma_t = self.sigma(gamma_s, z_h), self.sigma(gamma_t, z_h)
        mean = z_h / alpha - variance / alpha / sigma_t * eps_h
        return mean + std * sigma_s / sigma_t * noise

    @torch.no_grad()
    def sample_chain(self, x, h, node_mask, fragment_mask, linker_mask, edge_mask, context, keep_frames=None):
        self.validate_sampling()
        self.validate_masks(node_mask, fragment_mask, linker_mask)
        keep_frames = self.T if keep_frames is None else keep_frames
        if isinstance(keep_frames, bool) or not isinstance(keep_frames, int) or not 1 <= keep_frames <= self.T:
            raise ValueError('keep_frames must be in [1, sampling steps]')
        x, h = self.normalize(x, h)
        # Generated target values must not affect sampling, even if they are NaNs.
        fixed = torch.where(fragment_mask.bool(), torch.cat((x, h), -1), 0.)
        if not torch.isfinite(fixed).all():
            raise ValueError('Non-finite fixed-synthon input')
        self.last_sample_nfe = 0
        if not torch.any(linker_mask):
            return self.unnormalize_z(fixed).unsqueeze(0).repeat(keep_frames, 1, 1, 1)
        noise = self.sample_combined_position_feature_noise(x.shape[0], x.shape[1], linker_mask)
        noise[..., :self.n_dims] *= self.sigma_min
        z = fixed + noise
        # Only the SLDM coordinate transition is singular at t=1. The retained
        # VP element process still needs its first reverse transition, exactly
        # as in official EDM. Keep the coordinates at their small-noise prior.
        t1 = z.new_ones((x.shape[0], 1))
        first_t = z.new_full((x.shape[0], 1), (self.T - 1) / self.T)
        prediction = self.dynamics(t=t1, xh=z, node_mask=node_mask, linker_mask=linker_mask,
                                   edge_mask=edge_mask, context=context) * linker_mask
        self.last_sample_nfe += 1
        eps = self.sample_combined_position_feature_noise(x.shape[0], x.shape[1], linker_mask)
        first_h = self.element_transition(z[..., self.n_dims:], prediction[..., self.n_dims:],
                                          first_t, t1, eps[..., self.n_dims:])
        z = torch.cat((z[..., :self.n_dims], torch.where(linker_mask.bool(), first_h, fixed[..., self.n_dims:])), -1)
        if not torch.isfinite(z).all():
            raise FloatingPointError('Non-finite VP element bootstrap')
        chain = z.new_empty((keep_frames,) + z.shape)
        chain[-1] = self.unnormalize_z(z)
        # The coordinate part retains upstream's approximation at the skipped endpoint.
        for i in range(self.T - 1, 0, -1):
            t = z.new_full((x.shape[0], 1), i / self.T)
            s = z.new_full((x.shape[0], 1), (i - 1) / self.T)
            prediction = self.dynamics(t=t, xh=z, node_mask=node_mask, linker_mask=linker_mask,
                                       edge_mask=edge_mask, context=context) * linker_mask
            self.last_sample_nfe += 1
            eps = self.sample_combined_position_feature_noise(x.shape[0], x.shape[1], linker_mask)
            next_x = self.coordinate_mean(z[..., :self.n_dims], prediction[..., :self.n_dims], s, t)
            if i > 1:
                next_x += math.sqrt(2) * self.sigma_min * t.unsqueeze(-1).pow(self.temperature_power) * eps[..., :self.n_dims]
            next_h = self.element_transition(z[..., self.n_dims:], prediction[..., self.n_dims:], s, t, eps[..., self.n_dims:])
            z = torch.where(linker_mask.bool(), torch.cat((next_x, next_h), -1), fixed)
            if not torch.isfinite(z).all():
                raise FloatingPointError(f'Non-finite SLDM sample at step {i}; no clipping is applied')
            chain[((i - 1) * keep_frames) // self.T] = self.unnormalize_z(z)
        # Official element p(h|z0); coordinates already reached their final SLDM mean.
        t0 = z.new_zeros((x.shape[0], 1))
        prediction = self.dynamics(t=t0, xh=z, node_mask=node_mask, linker_mask=linker_mask,
                                   edge_mask=edge_mask, context=context) * linker_mask
        self.last_sample_nfe += 1
        gamma0 = self.gamma(t0)
        mean_h = self.compute_x_pred(prediction[..., self.n_dims:], z[..., self.n_dims:], gamma0)
        eps_h = torch.randn_like(mean_h)
        final_h = mean_h + self.SNR(-0.5 * gamma0).unsqueeze(1) * eps_h
        final = torch.cat((z[..., :self.n_dims], final_h), -1)
        final = torch.where(linker_mask.bool(), final, fixed)
        if not torch.isfinite(final).all():
            raise FloatingPointError('Non-finite SLDM final sample')
        chain[0] = self.unnormalize_z(final)
        return chain
