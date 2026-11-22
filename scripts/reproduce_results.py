"""Recompute released numerical summaries and figures on CPU, without downloads."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ttc4ts.results import (load_snapshot, summarize, main_summary, lw_budget_summary,
                           uncertainty_summary, MODEL_LABELS, METHOD_LABELS, geometric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/reproduced')
    args = parser.parse_args()
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=True)
    tables = {'main': main_summary(ROOT), 'learned_weights_budget': lw_budget_summary(ROOT),
              'uncertainty': uncertainty_summary(ROOT),
              'baselines': summarize(load_snapshot('baselines', ROOT)),
              'adapters': summarize(load_snapshot('adapters', ROOT), ['num_samples', 'method']),
              'retrieval': summarize(load_snapshot('retrieval', ROOT))}
    for name, data in tables.items(): data.to_csv(out/(name+'.csv'), index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    s = tables['main']
    fig, ax = plt.subplots(figsize=(8, 5), layout='constrained')
    ax.barh([METHOD_LABELS[x] for x in s.method], s.relative_mse,
            color=['#138c85' if x=='weighted' else '#879cad' for x in s.method])
    ax.axvline(1, color='black', linestyle='--', linewidth=1)
    ax.invert_yaxis(); ax.set_xlabel('Geometric relative MSE (Frozen = 1)')
    ax.set_title('Six checkpoints × six datasets × four horizons × three seeds')
    fig.savefig(out/'aggregation.png', dpi=160); fig.savefig(out/'aggregation.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), layout='constrained', sharey=False)
    raw = load_snapshot('lw_budget', ROOT)
    raw = raw[(raw.split=='test') & (raw.rule=='weighted')]
    for ax, dataset in zip(axes, ['electricity','national_illness','exchange_rate']):
        for model, group in raw[raw.dataset==dataset].groupby('model', sort=False):
            curve = group.groupby('K').relative_mse.agg(geometric)
            ax.plot(curve.index, curve.values, 'o-', label=MODEL_LABELS[model], markersize=3)
        ax.axhline(1, color='black', ls='--', lw=.8)
        ax.set_xscale('log',base=2); ax.set_xticks([1,2,4,8,16,80],['1','2','4','8','16','80'])
        ax.set_title(dataset.replace('_',' ').title()); ax.set_xlabel('Candidates K')
        ax.set_ylabel('Relative MSE'); ax.grid(alpha=.2)
    axes[-1].legend(fontsize=7)
    fig.savefig(out/'learned_weights_budget.png', dpi=160)
    fig.savefig(out/'learned_weights_budget.pdf'); plt.close(fig)
    print('Recomputed six measured summary tables and two figure pairs in', args.output)
    print('No GPU inference was run; manuscript source and published figures were not overwritten.')


if __name__ == '__main__': main()
