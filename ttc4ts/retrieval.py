"""Training-bank retrieval with explicit timestamps and past-only alignment."""
import torch


class RetrievalBank:
    def __init__(self, data, train_end, horizon, past=64, stride=16):
        self.past, self.horizon, self.train_end = past, horizon, train_end
        self.starts = torch.arange(0, train_end-past-horizon+1, stride, device=data.device)
        if not len(self.starts):
            raise ValueError('Training bank is too short')
        ix = self.starts[:, None] + torch.arange(past+horizon, device=data.device)
        self.pairs = data[ix]  # [N,S+H,C]
        self.ends = self.starts + past + horizon
        p = self.pairs[:, :past]
        self.norm = (p-p.mean(1,keepdim=True))/p.std(1,keepdim=True,unbiased=False).clamp_min(.01)

    def retrieve(self, x, origins, topk=2):
        q = x[:, -self.past:]
        norm = (q-q.mean(1,keepdim=True))/q.std(1,keepdim=True,unbiased=False).clamp_min(.01)
        # [B,C,N]; avoid a B*N*S*C temporary.
        dist = (norm.square().mean(1)[:, :, None] + self.norm.square().mean(1).T[None]
                - 2*torch.einsum('bsc,nsc->bcn', norm, self.norm)/self.past)
        mask = self.ends[None] <= origins[:, None]
        if (mask.sum(-1)<topk).any():
            raise ValueError('Not enough causally available reference pairs')
        dist = dist.masked_fill(~mask[:, None], float('inf'))
        ids = dist.topk(topk, largest=False).indices
        pairs = self.pairs.permute(2,0,1)[torch.arange(x.shape[2],device=x.device)[None,:,None],ids]
        pairs = pairs.permute(0,2,3,1)  # [B,K,S+H,C]
        ends = self.ends[ids]
        assert (ends <= origins[:,None,None]).all()
        return pairs, ids, dist.gather(-1,ids)


def align_pairs(query, pairs, amplitude=True, phase=True, max_lag=8):
    s = query.shape[1]
    result = pairs.clone()
    if phase:
        q = query-query.mean(1,keepdim=True)
        p = pairs[:,:,:s]-pairs[:,:,:s].mean(2,keepdim=True)
        scores = []
        lags = list(range(-min(max_lag,s//4),min(max_lag,s//4)+1))
        for lag in lags:
            qs, ps = (q[:,lag:],p[:,:,:s-lag]) if lag>0 else ((q[:,:s+lag],p[:,:,-lag:s]) if lag<0 else (q,p))
            score = (qs[:,None]*ps).sum(2)/(qs.square().sum(1)[:,None]*ps.square().sum(2)).sqrt().clamp_min(1e-6)
            scores.append(score)
        lag = torch.tensor(lags,device=pairs.device)[torch.stack(scores,2).argmax(2)]
        index = (torch.arange(pairs.shape[2],device=pairs.device)[None,None,:,None]-lag[:,:,None]).clamp(0,pairs.shape[2]-1)
        result = result.gather(2,index.expand_as(result))
    if amplitude:
        # Fit affine parameters on aligned reference past and query past only.
        pm = result[:,:,:s].mean(2,keepdim=True)
        ps = result[:,:,:s].std(2,keepdim=True,unbiased=False)
        qm = query.mean(1,keepdim=True)[:,None]
        qs = query.std(1,keepdim=True,unbiased=False)[:,None]
        scale = qs*ps/(ps.square()+(qs/5).square()+1e-6)
        result = (result-pm)*scale+qm
    return result
