"""Additional public model families with differentiable point-forecast interfaces."""
import math
from pathlib import Path
import sys
import torch
from ttc4ts.backbones import FrozenBackbone


class Timer(FrozenBackbone):
    def __init__(self,path,device='cuda'):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.model=AutoModelForCausalLM.from_pretrained(str(path),trust_remote_code=True,
            local_files_only=True,torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        self.model.config.use_cache=False
        self.batch_size=32
        self.patch=self.model.config.input_token_len
        self.max_context=2048
        self.eval()

    def _predict_flat(self,x,horizon):
        context=x; parts=[]; remaining=horizon
        while remaining>0:
            p=context[:,-self.max_context:]
            p=p[:,-(p.shape[-1]//self.patch*self.patch):]
            mean=p.mean(-1,keepdim=True);std=p.std(-1,keepdim=True).clamp_min(1e-5)
            z=(p-mean)/std
            y=self.model(input_ids=z,use_cache=False,return_dict=True,revin=False).logits
            y=y*std+mean; take=min(remaining,y.shape[-1])
            parts.append(y[:,:take]);context=torch.cat([context,y[:,:take]],-1);remaining-=take
        return torch.cat(parts,-1)


class Sundial(Timer):
    """Mean of eight common-random-number flow trajectories, batch invariant."""
    def __init__(self,path,device='cuda'):
        super().__init__(path,device)
        self.batch_size=16
        self.samples=8
        random=torch.Generator(device='cpu').manual_seed(1907)
        noise=torch.randn(self.samples,self.model.flow_loss.in_channels,generator=random)
        self.register_buffer('fixed_noise',noise.to(device))

    def _predict_flat(self,x,horizon):
        p=x[:,-self.max_context:]
        p=p[:,-(p.shape[-1]//self.patch*self.patch):]
        mean=p.mean(-1,keepdim=True);std=p.std(-1,keepdim=True).clamp_min(1e-5)
        z=(p-mean)/std
        hidden=self.model.model(input_ids=z,use_cache=False,return_dict=True).last_hidden_state[:,-1]
        b=len(z);condition=hidden.repeat(self.samples,1)
        noise=self.fixed_noise[:,None].expand(-1,b,-1).reshape(self.samples*b,-1)
        flow=self.model.flow_loss
        value=noise
        for i in range(flow.num_sampling_steps):
            t=value.new_full((len(value),),i/flow.num_sampling_steps)
            pred=flow.net(value,t*1000,condition)
            value=value+(pred-noise)/flow.num_sampling_steps
        y=value.reshape(self.samples,b,-1).mean(0)*std+mean
        if horizon>y.shape[-1]:raise ValueError('Requested horizon exceeds Sundial output block')
        return y[:,:horizon]


class Chronos2(FrozenBackbone):
    def __init__(self,path,device='cuda'):
        super().__init__()
        # The additional package directory isolates v2 from the existing Bolt runtime.
        from chronos.chronos2 import Chronos2Model
        self.model=Chronos2Model.from_pretrained(str(path),local_files_only=True,
            torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        conf=self.model.config.chronos_config
        self.median=min(range(len(conf['quantiles'])),key=lambda i:abs(conf['quantiles'][i]-.5))
        self.output_patch=conf['output_patch_size'];self.max_context=conf['context_length']
        self.batch_size=32;self.eval()

    def _predict_flat(self,x,horizon):
        output=self.model(context=x[:,-self.max_context:],group_ids=torch.arange(len(x),device=x.device),
            num_output_patches=math.ceil(horizon/self.output_patch))
        return output.quantile_preds[:,self.median,:horizon]


class TTM(FrozenBackbone):
    """IBM TTM research checkpoint (the 512-context/96-step branch)."""
    max_context=512
    def __init__(self, path, device='cuda'):
        super().__init__()
        from tsfm_public import TinyTimeMixerForPrediction
        self.model=TinyTimeMixerForPrediction.from_pretrained(
            str(path), local_files_only=True).to(device).eval().requires_grad_(False)
        self.batch_size=64
        self.eval()
    def _predict_flat(self,x,horizon):
        if horizon>96: raise ValueError('TTM research-r2 main branch supports horizon <=96')
        if x.shape[-1] != self.max_context:
            raise ValueError('TTM requires an exact 512-step context; no demonstration truncation is allowed')
        out=self.model(past_values=x[:,None,:],return_loss=False)
        y=out.prediction_outputs
        if y.ndim==3: y=y[:,0]
        return y[:,:horizon]


def load_extra(name,root,device='cuda'):
    directory=root.parent/'checkpoints_extra'/name
    if name=='timer-base-84m':return Timer(directory,device)
    if name=='sundial-base-128m':return Sundial(directory,device)
    if name=='chronos-2':return Chronos2(directory,device)
    if name=='ttm-research-r2':return TTM(directory,device)
    if name.startswith('ttm-'):return TTM(directory,device)
    raise ValueError(name)
