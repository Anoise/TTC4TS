"""Aggregation targets are accepted by fit(), never by forecast()."""
import torch
from torch import nn
import torch.nn.functional as F

AGGREGATOR_VERSION = 'trajectory-classifier-v2'


class Classifer(nn.Module):
    """User-supplied classifier; input is [B*K, seq_len*enc_dim].

    Flattening outside the module preserves the reference 2D InstanceNorm
    semantics: each trajectory is normalized independently across hidden units.
    """
    def __init__(self, seq_len, enc_dim, hidden_dim, num_samples=10):
        super().__init__()
        self.seq_len=seq_len
        self.enc_dim=enc_dim
        self.num_sample=num_samples
        self.fc1=nn.Linear(seq_len*enc_dim,hidden_dim)
        self.bn1=nn.InstanceNorm1d(hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.bn2=nn.InstanceNorm1d(hidden_dim)
        self.fc3=nn.Linear(hidden_dim,1)

    def forward(self,y):
        y=F.relu(self.fc1(y))
        y=self.bn1(y)
        y=F.relu(self.fc2(y))
        y=self.bn2(y)
        y=self.fc3(y)
        return y.squeeze(-1)


Classifier = Classifer


def errors(candidates, target):
    return (candidates-target[:,None]).square().mean((2,3))


def features(x, candidates, history_losses):
    scale=x.std((1,2),unbiased=False).clamp_min(.1)
    base=candidates[:,:1]
    normalized=candidates/scale[:,None,None,None]
    delta=(candidates-base)/scale[:,None,None,None]
    last=x[:,-1:].mean((1,2))/scale
    risk=history_losses/history_losses[:,:1].clamp_min(1e-5)
    return torch.stack([normalized.mean((2,3)),normalized.std((2,3),unbiased=False),
        delta.mean((2,3)),delta.square().mean((2,3)).sqrt(),
        normalized[:,:,0].mean(-1)-last[:,None],
        normalized[:,:,-1].mean(-1)-normalized[:,:,0].mean(-1),
        risk.clamp_max(100).log1p(), (history_losses+1e-5).log()],-1).clamp(-100,100)


class LearnedAggregator(nn.Module):
    def __init__(self, candidates, mode='weighted', width=24,
                 seq_len=None, enc_dim=None, hidden_dim=512):
        super().__init__()
        self.mode=mode
        if mode=='classifier':
            if seq_len is None or enc_dim is None:
                raise ValueError('Trajectory classifier requires seq_len and enc_dim')
            self.classifier=Classifer(seq_len,enc_dim,hidden_dim,candidates)
            return
        self.bias=nn.Parameter(torch.zeros(candidates))
        self.net=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,1))
        self.register_buffer('mean',torch.zeros(8))
        self.register_buffer('scale',torch.ones(8))

    def scores(self, feats=None, candidates=None):
        if self.mode=='classifier':
            if candidates is None or candidates.ndim!=4:
                raise ValueError('Classifier expects candidates [B,K,H,C]')
            b,k,h,c=candidates.shape
            if (h,c)!=(self.classifier.seq_len,self.classifier.enc_dim):
                raise ValueError('Classifier trajectory dimensions do not match')
            return self.classifier(candidates.reshape(b*k,h*c)).reshape(b,k)
        return self.net((feats-self.mean)/self.scale).squeeze(-1)+self.bias

    def weights(self, feats=None, candidates=None):
        if self.mode=='weighted': return self.bias.softmax(-1).expand(feats.shape[:2])
        score=self.scores(feats,candidates)
        if self.mode in ['classifier','ranking']:
            return F.one_hot(score.argmax(-1),score.shape[-1]).to(score.dtype)
        return (-score).softmax(-1)

    def forecast(self, candidates, feats=None):
        w=self.weights(feats,candidates)
        return (candidates*w[:,:,None,None]).sum(1)

    def fit(self, candidates, feats, targets, epochs=160, seed=0):
        torch.manual_seed(seed)
        if self.mode!='classifier':
            self.mean.copy_(feats.mean((0,1)))
            self.scale.copy_(feats.std((0,1),unbiased=False).clamp_min(.1))
        opt=torch.optim.AdamW(self.parameters(),lr=.01,weight_decay=.03)
        risk=errors(candidates,targets)
        labels=risk.argmin(-1)
        losslogs=[]
        for epoch in range(epochs):
            if self.mode=='weighted': loss=(self.forecast(candidates,feats)-targets).square().mean()
            else:
                score=self.scores(feats,candidates)
                if self.mode=='classifier': loss=F.cross_entropy(score,labels)
                elif self.mode=='risk': loss=F.smooth_l1_loss(score,(risk/risk[:,:1].clamp_min(1e-5)).clamp_max(100).log1p())
                elif self.mode=='ranking':
                    # Rank pairs by training-target error; ties carry zero weight.
                    sign=(risk[:,:,None]-risk[:,None,:]).sign()
                    mask=sign.ne(0)
                    pair=F.softplus((score[:,:,None]-score[:,None,:])*sign)
                    loss=pair[mask].mean() if mask.any() else score.sum()*0
                else: raise ValueError(self.mode)
            opt.zero_grad(); loss.backward(); opt.step()
            losslogs.append(float(loss.detach()))
        self.eval()
        return losslogs


def retrospective_weights(losses, temperature=.25, shrinkage=.25):
    scale=losses[:,:1].clamp_min(1e-5)
    w=torch.softmax(-losses/(temperature*scale),-1)*(1-shrinkage)
    w[:,0]+=shrinkage
    return w


def greedy_weights(candidates,target,steps=40):
    """Caruana-style selection with replacement, fit on development only."""
    k=candidates.shape[1]; w=torch.zeros(k)
    current=torch.zeros_like(target)
    for n in range(steps):
        risk=(((n*current[:,None]+candidates)/(n+1)-target[:,None]).square()).mean((0,2,3))
        best=int(risk.argmin()); w[best]+=1
        current=(n*current+candidates[:,best])/(n+1)
    return w/steps
