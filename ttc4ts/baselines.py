"""Standalone adapter and demo-only retrieval comparison, without future inputs."""
import copy
import torch
from torch import nn
from .adapters import Adapter

BASELINE_VERSION = 'standalone-d03-demo10-v1'
CHECKPOINT_STEPS = (32, 80, 160)


class PlainAdapter(nn.Module):
    """One separately trained additive delta-Adapter, not a multi-delta head."""
    def __init__(self, context, horizon, channels, mode, hidden_dim=512, delta=.3):
        super().__init__()
        if mode not in {'input', 'output', 'joint'}:
            raise ValueError(mode)
        self.mode, self.horizon, self.delta = mode, horizon, delta
        self.config = dict(context=context, horizon=horizon, channels=channels,
                           mode=mode, hidden_dim=hidden_dim, delta=delta)
        if mode in {'input', 'joint'}:
            self.x_adapter = Adapter(context, channels, hidden_dim, deltas=[delta])
        if mode in {'output', 'joint'}:
            self.y_adapter = Adapter(horizon, channels, hidden_dim, deltas=[delta])

    def forward(self, backbone, x, base=None):
        if self.mode in {'input', 'joint'}:
            y = backbone.predict(self.x_adapter.apply_delta(x, 0), self.horizon)
        else:
            y = backbone.predict(x, self.horizon) if base is None else base
        return self.y_adapter.apply_delta(y, 0) if self.mode in {'output', 'joint'} else y


def fit_plain(backbone, x, y, mode, seed, batch_size, checkpoints=CHECKPOINT_STEPS,
              hidden_dim=512, delta=.3, lr=1e-4):
    """Return frozen checkpoint states; callers select using validation only."""
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Expected a frozen backbone')
    torch.manual_seed(seed)
    adapter = PlainAdapter(x.shape[1], y.shape[1], x.shape[2], mode,
                           hidden_dim, delta).to(x.device)
    opt = torch.optim.Adam(adapter.parameters(), lr=lr)
    base = None
    if mode == 'output':
        with torch.no_grad():
            base = torch.cat([backbone.predict(z, y.shape[1]) for z in x.split(batch_size)])
    snapshots, logs = {}, []
    # Frozen differentiable model wrappers can have undefined input derivatives
    # at constant contexts. Use the identical boundary safeguard as fit_adapters;
    # count every replacement so this cannot silently hide numerical failures.
    invalid = [0]
    def boundary_hook(module, inputs, output):
        def finite_gradient(grad):
            invalid[0] += int((~torch.isfinite(grad)).sum())
            return torch.nan_to_num(grad, nan=0., posinf=0., neginf=0.)
        if output.requires_grad:
            output.register_hook(finite_gradient)
    handle = adapter.x_adapter.fc3.register_forward_hook(boundary_hook) if mode != 'output' else None
    for step in range(1, max(checkpoints)+1):
        ix = torch.randint(len(x), (min(batch_size, len(x)),), device=x.device)
        opt.zero_grad(set_to_none=True)
        pred = adapter(backbone, x[ix], None if base is None else base[ix])
        loss = (pred-y[ix]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite plain-adapter loss')
        loss.backward()
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise FloatingPointError('Nonfinite plain-adapter gradient')
        opt.step()
        logs.append(float(loss.detach()))
        if step in checkpoints:
            snapshots[step] = {k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()}
    if handle is not None:
        handle.remove()
    return adapter.eval(), snapshots, dict(loss=logs, undefined_input_gradients=invalid[0])


@torch.no_grad()
def raw_demo_forecasts(backbone, bank, x, origins, top_k=10):
    """Unaligned counterpart of the exact existing independently evaluated demos."""
    pairs, ids, distances = bank.retrieve(x, origins, topk=top_k)
    context = pairs.shape[2]+x.shape[1]
    if context > backbone.max_context:
        raise ValueError('Retrieval context would be truncated')
    candidates = torch.stack([backbone.predict(torch.cat([pairs[:,j], x], 1), bank.horizon)
                              for j in range(top_k)], 1)
    if not torch.isfinite(candidates).all():
        raise FloatingPointError('Nonfinite unaligned retrieval forecast')
    return candidates, dict(backbone_calls=top_k, context_tokens=top_k*context)
