"""Multi-sample residual adapters following the supplied Adapter class.

One adapter predicts a single bounded residual direction. The complete sample
family is obtained by multiplying that residual by
``Delta = linspace(0, max_delta, num_samples)``. The sample heads therefore
share the adapter parameters.
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ADAPTER_VERSION = 'shared-delta-v4-d03'
DEFAULT_MAX_DELTA = 0.3
DEFAULT_NUM_SAMPLES = 20
DEFAULT_DELTAS = tuple(float(v) for v in np.linspace(0., DEFAULT_MAX_DELTA,
                                                       DEFAULT_NUM_SAMPLES))


class Adapter(nn.Module):
    """The reference Adapter, usable for input or forecast tensors."""
    def __init__(self, seq_len, enc_dim=None, hidden_dim=512,
                 max_delta=DEFAULT_MAX_DELTA, num_samples=DEFAULT_NUM_SAMPLES,
                 deltas=None, operation='add', channels=None):
        super().__init__()
        if enc_dim is None:
            enc_dim = channels
        if enc_dim is None:
            raise ValueError('enc_dim/channels is required')
        self.seq_len, self.enc_dim = int(seq_len), int(enc_dim)
        self.num_sample = int(num_samples if deltas is None else len(deltas))
        self.max_delta = float(max_delta if deltas is None else max(deltas))
        self.operation = operation
        self.fc1 = nn.Linear(self.seq_len*self.enc_dim, hidden_dim)
        self.bn1 = nn.InstanceNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.InstanceNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.seq_len*self.enc_dim)
        values = np.linspace(0., self.max_delta, self.num_sample) if deltas is None else np.asarray(deltas)
        self.register_buffer('Delta', torch.tensor(values.reshape(1, self.num_sample, 1, 1), dtype=torch.float32))

    @property
    def deltas(self):
        return self.Delta.reshape(-1)

    def residual(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        return torch.tanh(self.fc3(x)).reshape(-1, self.seq_len, self.enc_dim)

    def candidates(self, x):
        base = x.reshape(x.shape[0], self.seq_len, self.enc_dim)
        residual = self.residual(base)
        if self.operation == 'add':
            return base[:, None] + self.Delta.to(base) * residual[:, None]
        if self.operation == 'multiply':
            return base[:, None] * (1. + self.Delta.to(base) * residual[:, None])
        raise ValueError(self.operation)

    def forward(self, x):
        return self.candidates(x).reshape(-1, self.seq_len, self.enc_dim)

    def aligned(self, x):
        """Apply the j-th Delta to the j-th item of an existing candidate axis."""
        if x.ndim != 4 or x.shape[1] != self.num_sample:
            raise ValueError('Expected [B,num_samples,L,C]')
        base = x.reshape(-1, self.seq_len, self.enc_dim)
        residual = self.residual(base).reshape(x.shape)
        d = self.Delta.to(x)
        if self.operation == 'add':
            return x + d * residual
        if self.operation == 'multiply':
            return x * (1. + d * residual)
        raise ValueError(self.operation)

    def apply_delta(self, x, index):
        """Apply one member of the Delta grid to a batch of tensors."""
        base = x.reshape(x.shape[0], self.seq_len, self.enc_dim)
        residual = self.residual(base)
        d = self.Delta.reshape(-1)[index].to(base)
        if self.operation == 'add':
            return base + d*residual
        if self.operation == 'multiply':
            return base*(1.+d*residual)
        raise ValueError(self.operation)


InputAdapter = Adapter
OutputAdapter = Adapter


class AdapterSuite(nn.Module):
    """Input and output adapters sharing the reference sample grid."""
    def __init__(self, context, horizon, channels, hidden_dim=512,
                 max_delta=DEFAULT_MAX_DELTA, num_samples=DEFAULT_NUM_SAMPLES,
                 deltas=None, operation='add'):
        super().__init__()
        if deltas is not None:
            num_samples = len(deltas)
            max_delta = float(max(deltas))
        self.config = dict(context=context, horizon=horizon, channels=channels,
                           hidden_dim=hidden_dim, max_delta=max_delta,
                           num_samples=num_samples, deltas=[float(v) for v in (np.linspace(0., max_delta, num_samples) if deltas is None else deltas)],
                           operation=operation)
        self.x_only = Adapter(context, channels, hidden_dim, max_delta, num_samples, deltas, operation)
        self.y_only = Adapter(horizon, channels, hidden_dim, max_delta, num_samples, deltas, operation)
        self.joint_x = self.x_only
        self.joint_y = self.y_only

    @property
    def deltas(self):
        return self.joint_x.deltas

    def checkpoint(self):
        return dict(version=ADAPTER_VERSION, config=self.config, state_dict=self.state_dict())

    @classmethod
    def from_checkpoint(cls, checkpoint, device='cpu'):
        if checkpoint['version'] != ADAPTER_VERSION:
            raise ValueError('Superseded adapter checkpoint')
        obj = cls(**checkpoint['config']).to(device)
        obj.load_state_dict(checkpoint['state_dict'])
        return obj.eval()


def fit_adapters(backbone, x, y, steps=32, seed=0, batch_size=2, lr=1e-4,
                 hidden_dim=512, max_delta=DEFAULT_MAX_DELTA,
                 num_samples=DEFAULT_NUM_SAMPLES, deltas=None, operation='add'):
    """Fit one shared residual direction for input, output, and joint paths.

    Losses are accumulated head by head to avoid retaining all backbone graphs.
    The resulting optimizer update is the mean loss over the complete Delta
    sample grid.
    """
    torch.manual_seed(seed)
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Backbone parameters must be frozen')
    device = x.device
    suite = AdapterSuite(x.shape[1], y.shape[1], x.shape[2], hidden_dim,
                         max_delta, num_samples, deltas, operation).to(device)
    optim = torch.optim.Adam(suite.parameters(), lr=lr)
    with torch.no_grad():
        base = torch.cat([backbone.predict(p, y.shape[1]) for p in x.split(batch_size)])
    k = suite.x_only.num_sample
    logs = []
    for step in range(steps):
        ix = torch.randint(len(x), (min(batch_size, len(x)),), device=device)
        bx, by, bb = x[ix], y[ix], base[ix]
        optim.zero_grad(set_to_none=True)
        x_res = suite.joint_x.residual(bx)
        y_res = suite.joint_y.residual(bb)
        invalid = [0]
        def finite_gradient(grad):
            invalid[0] += int((~torch.isfinite(grad)).sum())
            return torch.nan_to_num(grad, nan=0., posinf=0., neginf=0.)
        x_res.register_hook(finite_gradient)
        y_res.register_hook(finite_gradient)
        input_values, output_values, joint_values = [], [], []
        for j in range(k):
            d = suite.deltas[j]
            if operation == 'add':
                xm = bx + d*x_res
                oy = bb + d*y_res
            else:
                xm = bx*(1.+d*x_res)
                oy = bb*(1.+d*y_res)
            if j == 0:
                ip = op = jp = bb
            else:
                ip = backbone.predict(xm, y.shape[1])
                op = oy
                jp = ip
                jr = suite.joint_y.residual(jp)
                if operation == 'add':
                    jp = jp + d*jr
                else:
                    jp = jp*(1.+d*jr)
            li = (ip-by).square().mean()
            lo = (op-by).square().mean()
            lj = (jp-by).square().mean()
            input_values.append(li.detach()); output_values.append(lo.detach()); joint_values.append(lj.detach())
            if j > 0:
                ((li+lo+lj)/(3*k)).backward(retain_graph=(j < k-1))
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in suite.parameters()):
            raise FloatingPointError('Nonfinite adapter parameter gradient')
        optim.step()
        logs.append(dict(loss=float((torch.stack(input_values).mean()+torch.stack(output_values).mean()+torch.stack(joint_values).mean()).item()),
                         input=float(torch.stack(input_values).mean()), output=float(torch.stack(output_values).mean()),
                         joint=float(torch.stack(joint_values).mean()), undefined_input_gradients=invalid[0]))
    return suite.eval(), logs
