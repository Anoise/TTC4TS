"""Deployment API: takes observed history only, never a future target."""
from dataclasses import dataclass
import torch
from .aggregators import errors,features,retrospective_weights


@dataclass
class Forecast:
    prediction: torch.Tensor
    weights: torch.Tensor
    candidates: torch.Tensor
    cost: dict


class TTC4TS:
    def __init__(self,generator,context=512,horizon=96,mode='uniform',aggregator=None,
                 temperature=.3,shrinkage=.5,static_weights=None):
        self.generator=generator
        self.context,self.horizon=context,horizon
        self.mode,self.aggregator=mode,aggregator
        self.temperature,self.shrinkage=temperature,shrinkage
        self.static_weights=static_weights

    @torch.no_grad()
    def predict(self,history,origins):
        if history.ndim!=3 or history.shape[1]<self.context:
            raise ValueError('Expected observed [B,T,C] history of sufficient length')
        x=history[:,-self.context:]
        c,cost=self.generator.predict(x,origins,self.horizon)
        b,k=c.shape[:2]
        if self.mode=='uniform':w=c.new_full((b,k),1/k)
        elif self.mode=='static':w=self.static_weights.to(c).expand(b,k)
        elif self.mode=='weighted':w=self.aggregator.bias.to(c).softmax(-1).expand(b,k)
        elif self.mode=='classifier':w=self.aggregator.weights(candidates=c)
        else:
            if history.shape[1]<self.context+self.horizon:
                raise ValueError('History scoring needs context+horizon observed points')
            previous=history[:,-self.context-self.horizon:-self.horizon]
            pc,extra=self.generator.predict(previous,origins-self.horizon,self.horizon)
            observed=history[:,-self.horizon:]
            loss=errors(pc,observed)
            cost={**cost,'backbone_calls':cost['backbone_calls']+extra['backbone_calls'],
                  'context_tokens':cost['context_tokens']+extra['context_tokens']}
            if self.mode=='retrospective':w=retrospective_weights(loss,self.temperature,self.shrinkage)
            else:w=self.aggregator.weights(features(x,c,loss))
        return Forecast((c*w[:,:,None,None]).sum(1),w,c,cost)
