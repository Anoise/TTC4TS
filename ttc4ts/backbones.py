"""Differentiable frozen wrappers; every candidate uses the same point decoder."""
import hashlib
import torch
from torch import nn


def state_digest(module):
    h = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenBackbone(nn.Module):
    max_context = 2048

    def train(self, mode=True):
        super().train(False)
        return self

    def predict(self, x, horizon):
        if x.ndim != 3 or horizon < 1:
            raise ValueError('Expected [B,T,C] and positive horizon')
        b, t, c = x.shape
        flat = x.transpose(1, 2).reshape(b * c, t)
        pieces = [self._predict_flat(part, horizon) for part in flat.split(self.batch_size)]
        return torch.cat(pieces).reshape(b, c, horizon).transpose(1, 2)


class ChronosBolt(FrozenBackbone):
    def __init__(self, checkpoint, device='cuda', batch_size=32):
        super().__init__()
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(str(checkpoint), device_map=device, torch_dtype=torch.float32)
        self.model = pipeline.model.eval().requires_grad_(False)
        levels = self.model.config.chronos_config['quantiles']
        self.median = min(range(len(levels)), key=lambda i: abs(levels[i] - .5))
        self.max_context = self.model.config.chronos_config['context_length']
        self.batch_size = batch_size
        self.eval()

    def _predict_flat(self, x, horizon):
        context, blocks, remaining = x, [], horizon
        while remaining > 0:
            y = self.model(context=context[:, -self.max_context:]).quantile_preds[:, self.median]
            take = min(y.shape[-1], remaining)
            blocks.append(y[:, :take])
            context = torch.cat([context, y[:, :take]], -1)
            remaining -= take
        return torch.cat(blocks, -1)


class TimesFM(FrozenBackbone):
    max_context = 2048

    def __init__(self, checkpoint, device='cuda', batch_size=16):
        super().__init__()
        import timesfm
        pipeline = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend='gpu' if 'cuda' in device else 'cpu',
                per_core_batch_size=batch_size, context_len=2048, horizon_len=128,
                num_layers=50, use_positional_embedding=False, point_forecast_mode='median'),
            checkpoint=timesfm.TimesFmCheckpoint(version='torch', path=str(checkpoint)))
        self.model = pipeline._model.eval().requires_grad_(False)
        self.batch_size = batch_size
        self.eval()

    def _predict_flat(self, x, horizon):
        context, blocks, remaining = x, [], horizon
        while remaining > 0:
            p = context[:, -self.max_context:]
            pad = (-p.shape[-1]) % 32
            padding = torch.zeros_like(p)
            if pad:
                p = torch.nn.functional.pad(p, (pad, 0))
                padding = torch.nn.functional.pad(padding, (pad, 0), value=1.)
            raw = self.model(p, padding, torch.zeros((p.shape[0], 1), device=p.device, dtype=torch.long))
            # Official decoder: channel 0 is point output; quantiles .1 ... .9 follow.
            y = raw[:, -1, :, 5]
            take = min(y.shape[-1], remaining)
            blocks.append(y[:, :take])
            context = torch.cat([context, y[:, :take]], -1)
            remaining -= take
        return torch.cat(blocks, -1)


class SeasonalNaive(FrozenBackbone):
    def __init__(self, period=24):
        super().__init__()
        self.period, self.batch_size = period, 256
        self.register_buffer('anchor', torch.zeros(()))

    def _predict_flat(self, x, horizon):
        p = min(self.period, x.shape[-1])
        return x[:, -p:][:, torch.arange(horizon, device=x.device) % p]


def load_backbone(name, root, device='cuda'):
    if name == 'naive':
        return SeasonalNaive().to(device)
    if name.startswith('chronos-bolt'):
        return ChronosBolt(root / name, device)
    if name.startswith('timesfm'):
        return TimesFM(root / name / 'torch_model.ckpt', device)
    raise ValueError(f'Unsupported backbone: {name}')
