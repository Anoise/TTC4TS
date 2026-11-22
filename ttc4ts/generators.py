"""All requested candidate operators, with forecast-space inverses where valid."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .retrieval import align_pairs
from .adapters import DEFAULT_DELTAS

GENERATOR_VERSION = 'demo-only-top10-v1'
DEFAULT_RETRIEVAL_TOP_K = 10


@dataclass(frozen=True)
class Spec:
    name: str
    family: str
    value: float = 0.


def candidate_specs(deltas=DEFAULT_DELTAS, retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K):
    deltas=tuple(float(d) for d in deltas)
    return ([Spec('identity','identity')]
        + [Spec(f'input_{d:g}','adapter',d) for d in deltas[1:]]
        + [Spec(f'output_{d:g}','adapter',d) for d in deltas[1:]]
        + [Spec(f'joint_{d:g}','adapter',d) for d in deltas[1:]]
        + [Spec(n,'augmentation') for n in ['magnitude_flip','time_flip','magnitude_warp',
            'time_warp','ewma','jitter','horizontal','vertical_plus','vertical_minus',
            'slope_plus','slope_minus','spike']]
        + [Spec(f'retrieval_demo{j+1}','retrieval') for j in range(retrieval_top_k)])


def transform(name, x, horizon, seed=0):
    b,t,c=x.shape
    mu=x.mean(1,keepdim=True)
    sd=x.std(1,keepdim=True,unbiased=False).clamp_min(.05)
    inv=lambda y:y
    if name=='magnitude_flip':
        return 2*mu-x, lambda y:2*mu-y
    if name=='time_flip':
        return x.flip(1), inv  # stress perturbation, not a time-reversal symmetry
    if name=='magnitude_warp':
        f=lambda z:torch.exp(.15*torch.sin(2*torch.pi*z/t))[None,:,None]
        return mu+(x-mu)*f(torch.arange(t,device=x.device)), lambda y:mu+(y-mu)/f(torch.arange(t,t+horizon,device=x.device))
    if name=='time_warp':
        z=torch.linspace(0,1,t,device=x.device)
        loc=(z+.05*torch.sin(2*torch.pi*z))*(t-1)
        lo=loc.floor().long().clamp(0,t-1); hi=(lo+1).clamp_max(t-1)
        return x[:,lo]*(1-(loc-lo))[None,:,None]+x[:,hi]*(loc-lo)[None,:,None],inv
    if name=='ewma':
        vals=[x[:,0]]
        for j in range(1,t): vals.append(.3*x[:,j]+.7*vals[-1])
        return torch.stack(vals,1),inv
    if name=='jitter':
        # Stateless pseudo-random normals: independent of batching/evaluation order.
        z=torch.arange(t,device=x.device)[None,:,None]+x[:,-1:]*31.17+seed*13.1
        u=(torch.sin(z*12.9898)*43758.5453).frac().abs().clamp(.0001,.9999)
        v=(torch.sin(z*78.233+1.7)*12345.6789).frac().abs()
        noise=(-2*u.log()).sqrt()*torch.cos(2*torch.pi*v)
        return x+.03*sd*noise,inv
    if name=='horizontal':
        shift=max(1,t//20)
        return torch.cat([x[:,:1].expand(-1,shift,-1),x[:,:-shift]],1),inv
    if name.startswith('vertical'):
        shift=(.5 if name.endswith('plus') else -.5)*sd
        return x+shift,lambda y:y-shift
    if name.startswith('slope'):
        slope=(.5 if name.endswith('plus') else -.5)*sd/t
        return x+slope*torch.arange(-t+1,1,device=x.device)[None,:,None],lambda y:y-slope*torch.arange(1,horizon+1,device=x.device)[None,:,None]
    if name=='spike':
        z=x.clone(); z[:,-max(2,t//10)]+=(2*sd).squeeze(1)
        return z,inv
    raise ValueError(name)


class Generator:
    def __init__(self, backbone, adapters, bank, seed=0, retrieval_top_k=DEFAULT_RETRIEVAL_TOP_K):
        self.backbone,self.adapters,self.bank=backbone,adapters,bank
        self.deltas=tuple(float(d) for d in adapters.config['deltas'])
        if not self.deltas or self.deltas[0] != 0.:
            raise ValueError('Adapter candidate grids must start at delta=0')
        self.retrieval_top_k=int(retrieval_top_k)
        if self.retrieval_top_k < 1:
            raise ValueError('retrieval_top_k must be positive')
        self.specs=candidate_specs(self.deltas,self.retrieval_top_k)
        self.seed=seed

    def _delta_index(self, value):
        return list(self.deltas).index(value)

    @torch.no_grad()
    def predict(self, x, origins, horizon, names=None, base_forecast=None):
        wanted={s.name for s in self.specs} if names is None else set(names)
        unknown=wanted-{s.name for s in self.specs}
        if unknown: raise ValueError(f'Unknown candidates: {sorted(unknown)}')
        base=self.backbone.predict(x,horizon) if base_forecast is None else base_forecast
        outputs={'identity':base}
        calls=int(base_forecast is None); tokens=x.shape[1]*calls
        for s in self.specs[1:]:
            if s.name not in wanted: continue
            if s.name.startswith('input_'):
                idx=self._delta_index(s.value)
                z=self.adapters.x_only.candidates(x)[:,idx]
                outputs[s.name]=self.backbone.predict(z,horizon)
                calls+=1; tokens+=x.shape[1]
            elif s.name.startswith('output_'):
                idx=self._delta_index(s.value)
                outputs[s.name]=self.adapters.y_only.candidates(base)[:,idx]
            elif s.name.startswith('joint_'):
                idx=self._delta_index(s.value)
                x_heads=self.adapters.joint_x.candidates(x)
                z=x_heads[:,idx]
                y=self.backbone.predict(z,horizon)
                outputs[s.name]=self.adapters.joint_y.apply_delta(y,idx)
                calls+=1; tokens+=x.shape[1]
            elif s.family=='augmentation':
                z,inverse=transform(s.name,x,horizon,self.seed)
                outputs[s.name]=inverse(self.backbone.predict(z,horizon))
                calls+=1; tokens+=z.shape[1]
        if any(s.family=='retrieval' and s.name in wanted for s in self.specs):
            pairs,ids,distance=self.bank.retrieve(x,origins,self.retrieval_top_k)
            q=x[:,-self.bank.past:]
            aligned=align_pairs(q,pairs,True,True)
            for j in range(self.retrieval_top_k):
                name=f'retrieval_demo{j+1}'
                if name not in wanted: continue
                # Each neighbour yields its own forecast; aggregation is a separate stage.
                z=torch.cat([aligned[:,j],x],1)
                if z.shape[1]>self.backbone.max_context:
                    raise ValueError('Backbone context would truncate a retrieval demonstration')
                outputs[name]=self.backbone.predict(z,horizon)
                calls+=1; tokens+=z.shape[1]
        names=[s.name for s in self.specs if s.name in wanted]
        values=torch.stack([outputs[n] for n in names],1)
        if not torch.isfinite(values).all():
            bad=[names[i] for i in range(len(names)) if not torch.isfinite(values[:,i]).all()]
            raise ValueError(f'Nonfinite candidates: {bad}')
        return values,dict(backbone_calls=calls,context_tokens=tokens,names=names)
