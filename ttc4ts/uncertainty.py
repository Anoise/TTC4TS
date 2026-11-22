"""Candidate-informed prediction intervals; all labels enter fit/calibrate only.

Split-conformal coverage requires exchangeable scores. Time-series experiments
report empirical coverage, not an unconditional distribution-free guarantee.
"""
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

UQ_VERSION = 'candidate-uq-v1'
ALPHAS = (.20, .10, .05)
QUANTILES = (.025, .05, .10, .50, .90, .95, .975)


def family_ids(names):
    groups = [[i for i,n in enumerate(names) if n.startswith(('input_', 'output_', 'joint_'))],
              [i for i,n in enumerate(names) if i and not n.startswith(('input_', 'output_', 'joint_', 'retrieval_'))],
              [i for i,n in enumerate(names) if n.startswith('retrieval_')]]
    if any(not g for g in groups) or sum(map(len, groups)) != len(names)-1:
        raise ValueError('Expected one identity and the three generator families')
    return groups


def uncertainty_features(candidates, point, history, historical_losses, names):
    """Return [N,H,C,12] observable features and a family-balanced spread.

    First five features contain no current candidate dispersion. Historical
    losses are from an already observed forecasting horizon, never this target.
    """
    groups = family_ids(names)
    n, _, h, c = candidates.shape
    if point.shape != (n,h,c) or history.shape[0] != n:
        raise ValueError('Mismatched candidate/history/point shapes')
    std = history.std(1, unbiased=False)[:,None,:].expand(n,h,c)
    change = history.diff(dim=1).square().mean(1).sqrt()[:,None,:].expand(n,h,c)
    past = historical_losses[:,0].clamp_min(0).sqrt()[:,None,None].expand(n,h,c)
    lead = torch.linspace(1/h,1,h,device=point.device)[None,:,None].expand(n,h,c)
    displacement = (point-history[:,-1:, :]).abs()
    basic = [std.log1p(), change.log1p(), past.log1p(), lead, displacement.log1p()]
    means = torch.stack([candidates[:,g].mean(1) for g in groups],1)
    within = [candidates[:,g].std(1,unbiased=False) for g in groups]
    between = means.std(1,unbiased=False)
    # Each family has mass 1/3, independent of the number of adapter grid points.
    spread = torch.stack([(candidates[:,g]-point[:,None]).square().mean(1)
                          for g in groups]).mean(0).sqrt()
    lower,upper = torch.quantile(candidates, candidates.new_tensor([.1,.9]), dim=1)
    signed_log = lambda x: x.sign()*x.abs().log1p()
    extra = [v.log1p() for v in within]+[between.log1p(),spread.log1p(),
              signed_log(lower-point), signed_log(upper-point)]
    features = torch.stack(basic+extra,-1)
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite uncertainty features')
    return features, spread


class UncertaintyHead(nn.Module):
    """Small scale or ordered residual-quantile head on observed features."""
    def __init__(self, dim=12, mode='scale', width=32):
        super().__init__()
        if mode not in ('scale','quantile'):
            raise ValueError(mode)
        self.dim,self.mode,self.width = dim,mode,width
        self.net = nn.Sequential(nn.Linear(dim,width),nn.ReLU(),
                                 nn.Linear(width,1 if mode=='scale' else len(QUANTILES)))
        self.register_buffer('mean',torch.zeros(dim))
        self.register_buffer('std',torch.ones(dim))

    def forward(self, features):
        raw = self.net(((features[...,:self.dim]-self.mean)/self.std).clamp(-10,10))
        return F.softplus(raw[...,0])+.01 if self.mode=='scale' else raw.sort(-1).values

    def fit(self, features, residual, steps=150, seed=0, batch_size=4096):
        torch.manual_seed(seed)
        x=features[...,:self.dim].reshape(-1,self.dim)
        y=residual.reshape(-1)
        self.mean.copy_(x.mean(0)); self.std.copy_(x.std(0,unbiased=False).clamp_min(.05))
        opt=torch.optim.AdamW(self.parameters(),lr=.003,weight_decay=.001)
        losses=[]
        for _ in range(steps):
            ix=torch.randint(len(x),(min(batch_size,len(x)),),device=x.device)
            pred=self(x[ix]); target=y[ix]
            if self.mode=='scale':
                loss=(.5*(target/pred).square()+pred.log()).mean()
            else:
                diff=target[:,None]-pred
                q=diff.new_tensor(QUANTILES)
                loss=torch.maximum(q*diff,(q-1)*diff).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite UQ training objective')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(self.parameters(),10)
            opt.step(); losses.append(float(loss.detach()))
        self.eval()
        return losses

    def checkpoint(self):
        return dict(version=UQ_VERSION,dim=self.dim,mode=self.mode,width=self.width,
                    state_dict=self.state_dict())


def conformal_quantile(scores, alpha):
    """Corrected order statistic over origins, independently per coordinate.

    Infinite quantiles are explicit when the requested level exceeds the sample
    resolution. Never silently replace them with the maximum observed residual.
    """
    values=np.asarray(scores,dtype=np.float64)
    if values.ndim < 1 or len(values)==0 or not np.isfinite(values).all():
        raise ValueError('Calibration scores must be nonempty and finite')
    n=len(values)
    alpha=np.asarray(alpha,dtype=np.float64)
    if np.any((alpha<=0)|(alpha>=1)):
        raise ValueError('alpha must be between zero and one')
    rank=np.ceil((n+1)*(1-alpha)-1e-12).astype(int)
    ordered=np.sort(values,axis=0)
    if rank.ndim==0:
        return np.full(values.shape[1:],np.inf) if rank>n else ordered[max(0,int(rank)-1)]
    rank=np.broadcast_to(rank,values.shape[1:])
    q=np.take_along_axis(ordered,np.clip(rank-1,0,n-1)[None],axis=0)[0]
    return np.where(rank>n,np.inf,q)


class ScaledConformal:
    """Calibrate a fixed point/scale predictor; predict() accepts no targets."""
    def __init__(self, alpha=.1, simultaneous=False):
        self.alpha,self.simultaneous=alpha,simultaneous
        self.q=None

    def fit(self, target, point, scale):
        target,point,scale=map(np.asarray,(target,point,scale))
        if target.shape!=point.shape or scale.shape!=point.shape or np.any(scale<=0):
            raise ValueError('Expected equal [N,H,C] arrays and positive scales')
        scores=np.abs(target-point)/scale
        if self.simultaneous: scores=scores.max(axis=1,keepdims=True)
        self.q=conformal_quantile(scores,self.alpha)
        return self

    def predict(self, point, scale):
        if self.q is None: raise RuntimeError('Calibrate before predicting')
        point,scale=np.asarray(point),np.asarray(scale)
        if point.shape!=scale.shape or np.any(scale<=0): raise ValueError('Invalid scale')
        return point-self.q*scale,point+self.q*scale


class RollingConformal:
    """Causal rolling calibration with explicit forecast-then-observe calls.

    Bounded ACI updates alpha by gamma*(target_alpha-miss) and projects into
    [1/(n+1), .5]. This finite-width variant has no unrestricted ACI guarantee.
    """
    def __init__(self,scores,alpha=.1,adaptive=False,gamma=.01):
        self.scores=np.array(scores,dtype=np.float64,copy=True)
        if len(self.scores)<2: raise ValueError('At least two calibration origins')
        self.alpha,self.adaptive,self.gamma=alpha,adaptive,gamma
        self.current_alpha=np.full(self.scores.shape[1:],alpha)
        self.pending=None; self.updates=0; self.clipped_updates=0

    def predict(self,point,scale):
        if self.pending is not None: raise RuntimeError('Observe previous horizon before next forecast')
        q=conformal_quantile(self.scores,self.current_alpha)
        self.pending=(np.array(point,copy=True),np.array(scale,copy=True),q)
        return point-q*scale,point+q*scale

    def observe(self,target):
        if self.pending is None: raise RuntimeError('Cannot observe before predicting')
        point,scale,q=self.pending
        score=np.abs(np.asarray(target)-point)/scale
        if self.adaptive:
            desired=self.current_alpha+self.gamma*(self.alpha-(score>q))
            self.current_alpha=np.clip(desired,1/(len(self.scores)+1),.5)
            self.clipped_updates+=int(np.count_nonzero(desired!=self.current_alpha))
        self.scores=np.concatenate([self.scores[1:],score[None]],axis=0)
        self.pending=None;self.updates+=1


def interval_metrics(target,lower,upper,alpha):
    target,lower,upper=map(np.asarray,(target,lower,upper))
    if target.shape!=lower.shape or target.shape!=upper.shape or np.any(lower>upper):
        raise ValueError('Invalid prediction interval shapes or endpoints')
    if not all(np.isfinite(v).all() for v in (target,lower,upper)):
        raise ValueError('Report infinite intervals explicitly instead of finite-only metrics')
    covered=(target>=lower)&(target<=upper)
    width=upper-lower
    score=width+2/alpha*(np.maximum(lower-target,0)+np.maximum(target-upper,0))
    return dict(coverage=float(covered.mean()),width=float(width.mean()),
                interval_score=float(score.mean()),
                trajectory_coverage=float(covered.all(axis=1).mean()),
                joint_coverage=float(covered.all(axis=(1,2)).mean()),
                per_lead_coverage=covered.mean(axis=(0,2)).tolist(),
                per_origin_coverage=covered.mean(axis=(1,2)).tolist(),
                per_origin_width=width.mean(axis=(1,2)).tolist(),
                per_origin_interval_score=score.mean(axis=(1,2)).tolist())


def weighted_interval_score(target,median,intervals,alphas=ALPHAS):
    score=.5*np.abs(np.asarray(target)-median)
    for alpha,(lo,hi) in zip(alphas,intervals):
        score+=(alpha/2)*(hi-lo)+(np.maximum(lo-target,0)+np.maximum(target-hi,0))
    return score/(len(alphas)+.5)
