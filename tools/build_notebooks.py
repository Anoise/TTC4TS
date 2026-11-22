"""Rebuild portable notebook sources (execution is a separate explicit step)."""
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def md(source): return ('markdown', textwrap.dedent(source).strip())
def code(source): return ('code', textwrap.dedent(source).strip())


BOOT = code('''
from pathlib import Path
import sys, json
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'artifacts/manifest.json').exists())
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from ttc4ts.results import (checked_path, load_snapshot, geometric, summarize,
    main_summary, lw_budget_summary, uncertainty_summary, MODELS, MODEL_LABELS, METHODS, METHOD_LABELS)
%matplotlib inline
plt.rcParams.update({'figure.dpi': 115, 'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18})
pd.set_option('display.max_rows', 60)
pd.set_option('display.float_format', lambda x: f'{x:.5f}')
print('Offline mode: numerical snapshots are verified against the released SHA-256 manifest.')
''')


def write(name, cells):
    out = []
    for index, (kind, source) in enumerate(cells):
        cell = dict(cell_type=kind, metadata={}, source=source+'\n',
                    id=hashlib.sha256(f'{name}:{index}:{source}'.encode()).hexdigest()[:12])
        if kind == 'code': cell.update(execution_count=None, outputs=[])
        out.append(cell)
    nb = dict(nbformat=4, nbformat_minor=5, cells=out,
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      language_info=dict(name='python'), ttc4ts=dict(offline_default=True, release='0.3.0')))
    (ROOT/'notebooks'/name).write_text(json.dumps(nb, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')


def main():
    (ROOT/'notebooks').mkdir(exist_ok=True)
    write('00_cpu_quickstart.ipynb', [md('''
    # 00 · A complete TTC4TS prediction on CPU
    **Purpose:** understand the API and leakage boundaries without downloading models.
    This notebook fits lightweight adapters and a Risk regressor around a frozen seasonal-naive
    backbone. The generated sine/cosine series and its scores are **synthetic demonstrations**, not paper evidence.
    Install `pip install -e ".[analysis,notebooks]"` from the repository and run all cells.
    '''), BOOT, code('''
    import torch
    import warnings
    # Reference InstanceNorm intentionally normalizes 2D hidden activations.
    # Suppress only its known feature-count warning to keep outputs portable.
    warnings.filterwarnings('ignore', message="input's size at dim=0 does not match num_features", category=UserWarning)
    from ttc4ts.adapters import fit_adapters
    from ttc4ts.aggregators import LearnedAggregator, errors, features
    from ttc4ts.backbones import SeasonalNaive, state_digest
    from ttc4ts.generators import Generator
    from ttc4ts.predictor import TTC4TS
    from ttc4ts.retrieval import RetrievalBank
    torch.set_num_threads(2)
    torch.manual_seed(7)
    context, horizon, fitting_end = 64, 16, 480
    t = torch.arange(1100, dtype=torch.float32)
    raw = torch.stack([torch.sin(t/8), torch.cos(t/13)+.0005*t], -1) + .05*torch.randn(1100, 2)
    data = (raw-raw[:fitting_end].mean(0))/raw[:fitting_end].std(0, unbiased=False).clamp_min(1e-6)
    def windows(origins):
        return (torch.stack([data[o-context:o] for o in origins]),
                torch.stack([data[o:o+horizon] for o in origins]))
    backbone = SeasonalNaive(period=24)
    before = state_digest(backbone)
    x, y = windows(torch.arange(96, 400, 32))
    adapters, trace = fit_adapters(backbone, x, y, steps=2, hidden_dim=16,
                                   max_delta=.3, num_samples=5, seed=7)
    bank = RetrievalBank(data, fitting_end, horizon, past=32, stride=8)
    generator = Generator(backbone, adapters, bank, seed=7)
    print('Adapter Delta:', adapters.deltas.detach().cpu().flatten().tolist())
    '''), md('''
    The aggregator fits on later observed windows. Its historical losses use the already
    observed horizon immediately preceding each origin. No current future is used at deployment.
    Five deltas produce twelve nonzero Adapter candidates; the toy pool still contains all
    twelve augmentations and ten Retrieval-demo candidates.
    '''), code('''
    origins = torch.arange(560, 800, 32)
    x, y = windows(origins)
    candidates, _ = generator.predict(x, origins, horizon)
    previous, _ = generator.predict(windows(origins-horizon)[0], origins-horizon, horizon)
    observed = torch.stack([data[o-horizon:o] for o in origins])
    feats = features(x, candidates, errors(previous, observed))
    risk = LearnedAggregator(candidates.shape[1], mode='risk')
    risk.fit(candidates, feats, y, epochs=10, seed=7)
    predictor = TTC4TS(generator, context, horizon, mode='risk', aggregator=risk)
    origin = 960
    result = predictor.predict(data[origin-context-horizon:origin][None], torch.tensor([origin]))
    assert state_digest(backbone) == before
    assert result.prediction.shape == (1, horizon, 2)
    assert torch.allclose(result.weights.sum(-1), torch.ones(1))
    target = data[origin:origin+horizon]  # evaluation only, after prediction
    frozen = result.candidates[0, 0]
    display(pd.DataFrame({'method': ['Frozen', 'TTC4TS Risk'],
        'synthetic_MSE': [float((frozen-target).square().mean()),
                          float((result.prediction[0]-target).square().mean())]}))
    print({'candidate_count': result.candidates.shape[1], 'backbone_unchanged': True, 'cost': result.cost})
    '''), code('''
    fig, ax = plt.subplots(figsize=(9, 3.5), layout='constrained')
    ax.plot(range(-context, 0), data[origin-context:origin, 0], color='gray', label='Observed')
    ax.plot(range(horizon), target[:, 0], color='black', label='Target')
    ax.plot(range(horizon), frozen[:, 0], '--', color='#ce7731', label='Frozen')
    ax.plot(range(horizon), result.prediction[0, :, 0], color='#138c85', label='TTC4TS Risk')
    ax.axvline(-.5, color='gray', lw=.8); ax.set(xlabel='Step from forecast origin', ylabel='Standardized value',
        title='Synthetic demonstration · channel 0 · two Adapter updates')
    ax.legend(ncol=4, fontsize=8); plt.show()
    '''), md('''
    This verifies shapes, convex weights, frozen state, and the deployment interface. It does
    not establish superiority or calibrated uncertainty. Use notebook 01 for measured comparisons.
    ''')])

    write('01_main_results.ipynb', [md('''
    # 01 · Main aggregation and independent baselines
    **Measured scope:** six checkpoints × ETTh1/ETTh2/ETTm1/ETTm2/Appliances/BeijingWeather ×
    horizons 96/192/336/720 × three seeds = 432 cells. Relative errors divide each cell by its
    frozen backbone. Geometric means give equal weight to cells, not datasets' raw loss scales.
    Oracle uses unavailable test labels and is never a deployed method. These are the revised
    max_delta=0.3, 80-candidate, Retrieval-demo top-10 measurements.
    '''), BOOT, code('''
    raw = load_snapshot('main')
    headline = raw[raw.method.isin(METHODS)].copy()
    assert len(headline) == 432*11
    assert not headline.duplicated(['model','dataset','horizon','seed','method']).any()
    table = main_summary()
    display(table.assign(method=table.method.map(METHOD_LABELS)))
    '''), code('''
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    colors = ['#138c85' if m=='weighted' else '#b2bac4' if m=='oracle' else '#547b9e' for m in table.method]
    bars = ax.barh(table.method.map(METHOD_LABELS), table.relative_mse, color=colors)
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8)
    ax.axvline(1, color='black', ls='--', lw=1); ax.invert_yaxis()
    ax.set_xlim(0, max(table.relative_mse)*1.12)
    ax.set(xlabel='Geometric relative MSE · lower is better', title='All 432 cells · Frozen = 1')
    plt.show()
    by_dataset = summarize(headline[headline.method=='weighted'], ['dataset','model'])
    display(by_dataset.pivot(index='dataset', columns='model', values='relative_mse').rename(columns=MODEL_LABELS))
    '''), md('''
    The primary fixed Learned-weights rule and validation-selected TTC4TS are different estimators.
    The Classifier is retained even when it degrades performance. Win counts are descriptive
    counts across correlated tasks, not independent significance tests.

    ## Independent controls
    The 17-method follow-up includes separately trained plain adapters and retrieval-transfer
    controls. It uses the same 432-cell scope. Transfer controls are not in TTC4TS's candidate pool.
    '''), code('''
    baselines = load_snapshot('baselines')
    assert len(baselines) == 432*17
    baseline_table = summarize(baselines).sort_values('relative_mse')
    display(baseline_table)
    print('Best observed independent-control MSE:', baseline_table.iloc[0]['method'])
    print('Do not infer that every generator combination beats every component.')
    '''), md('''
    For underlying measured rows inspect `artifacts/main/all_metrics.csv.gz` and
    `artifacts/baselines/all_metrics.csv`. The historical raw-prediction audit is included under
    `artifacts/main/reports/`; reproducing its target-level checks requires newly generated or
    separately archived raw arrays, not these summary CSVs alone.
    ''')])

    write('02_generators_and_adapters.ipynb', [md('''
    # 02 · Generators, Retrieval-demo and Adapter grids
    Main family comparisons use **Uniform aggregation** and include the frozen identity once.
    Main scope: six checkpoints, six datasets, four horizons, three seeds (432 cells).
    The Adapter grid sweep below is a distinct 72-cell experiment with a **Risk regressor**.
    '''), BOOT, code('''
    main = load_snapshot('main')
    families = main[main.method.str.startswith('families/') | (main.method=='backbone')]
    family_table = summarize(families)
    display(family_table)
    fig, ax = plt.subplots(figsize=(9, 4.3), layout='constrained')
    bars = ax.barh(family_table.method.str.replace('families/', '', regex=False), family_table.relative_mse, color='#547b9e')
    ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=8); ax.invert_yaxis()
    ax.axvline(1, color='black', ls='--'); ax.set_xlim(0, family_table.relative_mse.max()*1.12)
    ax.set(xlabel='Geometric relative MSE', title='Uniform combinations · 432 measured cells')
    plt.show()
    display(summarize(load_snapshot('retrieval')))
    '''), md('''
    Retrieval-demo uses ten independently aligned one-demo forecasts plus identity. A top-10
    retrieval list does not imply concatenating ten demonstrations. Uniform/risk/validation-best
    aggregation is measured explicitly; the code does not assume retrieval must improve a frozen backbone.

    ## Adapter samples: five to forty
    Bolt-S, Bolt-B and TimesFM × ETTh1, ETTm1 and Appliances × eight N values;
    H=96, seed=0. Each N refits the Adapter and Risk regressor. All-view candidates number
    `1+3*(N-1)`; N is the size of each delta grid, not the total pool size.
    '''), code('''
    sweep = load_snapshot('adapters')
    assert len(sweep) == 72*6
    sweep_table = summarize(sweep, ['num_samples','method'])
    display(sweep_table[sweep_table.method=='risk_all'])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained', sharey=True)
    for ax, dataset in zip(axes, ['ETTh1','ETTm1','Appliances']):
        for model, group in sweep[(sweep.dataset==dataset) & (sweep.method=='risk_all')].groupby('model'):
            curve = group.sort_values('num_samples')
            ax.plot(curve.num_samples, curve.relative_mse, 'o-', label=MODEL_LABELS[model], ms=4)
        ax.axhline(1, color='black', ls='--', lw=.8)
        ax.set(title=dataset, xlabel='Samples per Adapter grid N', ylabel='Relative MSE')
        ax.set_xticks([5,10,15,20,25,30,35,40])
    axes[-1].legend(fontsize=8); plt.show()
    curves = sweep[sweep.method=='risk_all'].sort_values('num_samples').groupby(['model','dataset'])
    monotone = sum(np.all(np.diff(group.relative_mse)<=1e-10) for _, group in curves)
    print(f'Monotonically nonincreasing Risk curves: {monotone}/{len(curves)}')
    '''), md('''
    More samples do not guarantee lower test MSE. The shared residual creates correlated
    candidates; refitting adds optimization variability; a learned selector can overfit. The
    paper's compute-benefit analysis states conditions and separates population optima from
    finite-sample fitted performance. This one-seed sweep is a sensitivity study, not a scaling law.
    ''')])

    write('03_candidate_budgets.ipynb', [md('''
    # 03 · Candidate-budget curves, with exact dataset versions
    **Primary study:** fixed Learned weights, 160 fitting updates at each K, no fallback.
    Six models × three datasets × three seeds = 54 cells. K = 1,2,4,8,16,80.
    Electricity: 321 channels, L512/H96; National Illness: 7 channels, L104/H24;
    Exchange Rate: 8 channels, L512/H96. National Illness has only eight test origins.
    '''), BOOT, code('''
    all_budget = load_snapshot('lw_budget')
    assert len(all_budget) == 1944
    budget = all_budget[(all_budget.split=='test') & (all_budget.rule=='weighted')]
    display(lw_budget_summary())
    def budget_plot(frame, title):
        datasets = list(frame.dataset.unique())
        fig, axes = plt.subplots(1, len(datasets), figsize=(13, 4), layout='constrained')
        for ax, dataset in zip(axes, datasets):
            for model, group in frame[frame.dataset==dataset].groupby('model'):
                curve = group.groupby('K').relative_mse.agg(geometric)
                ax.plot(curve.index, curve.values, 'o-', ms=3, label=MODEL_LABELS[model])
            ax.axhline(1, color='black', ls='--', lw=.8)
            ax.set_xscale('log', base=2); ax.set_xticks([1,2,4,8,16,80], ['1','2','4','8','16','80'])
            ax.set(title=dataset.replace('_',' ').title(), xlabel='Candidate count K', ylabel='Relative MSE')
        axes[-1].legend(fontsize=7); fig.suptitle(title, fontsize=12); plt.show()
    budget_plot(budget, 'Multivariate files · fixed Learned weights · geometric mean over three seeds')
    detail = summarize(budget, ['dataset','model','K'])
    display(detail.pivot(index=['dataset','model'], columns='K', values='relative_mse'))
    curves = detail.sort_values('K').groupby(['model','dataset'])
    print('Nonincreasing curves:', sum(np.all(np.diff(g.relative_mse)<=1e-10) for _, g in curves), '/', len(curves))
    '''), md('''
    ## Historical single-channel experiment — different files
    Traffic, Electricity and Exchange Rate each had one `OT` channel, L512/H96.
    Here we show the original **validation-selected TTC4TS** curve. Its fitting schedule and
    selection rule differ from the multivariate study. Same dataset name does not mean same data.
    '''), code('''
    old = load_snapshot('single_channel')
    assert len(old)==3564 and set(old.channels)=={1}
    old_selected = old[old.rule=='validation_selected']
    display(summarize(old_selected, ['dataset','K']))
    budget_plot(old_selected, 'Historical single-channel files · validation-selected TTC4TS')
    current_hashes = json.loads(checked_path('provenance/lw_dataset_manifest.json').read_text())
    historic_hashes = json.loads(checked_path('provenance/single_channel_dataset_manifest.json').read_text())
    display(pd.DataFrame([dict(study=study, dataset=dataset, channels=entry['metadata']['channels'],
        sha256=entry['metadata']['sha256']) for study, manifest in [('multivariate',current_hashes),
        ('historical_single_channel',historic_hashes)] for dataset,entry in manifest.items()]))
    '''), md('''
    The complete per-seed MSE/MAE records are available in the two budget CSVs. Displayed curves
    are geometric means, not confidence intervals. K=1 is identity. A larger nested candidate
    set enlarges the oracle choice set; learned finite-step fits need not attain that optimum.
    See `docs/reproduction.md` for hash-guarded GPU commands.
    ''')])

    write('04_uncertainty.ipynb', [md('''
    # 04 · Candidate-based uncertainty and calibration
    **Scope:** 108 H96 cells, six checkpoints, six main datasets, three seeds; 13 methods,
    nominal 80/90/95% intervals. Point aggregators were refit with separate purged UQ blocks.
    These forecasts are not reused from the primary fixed Learned-weights study.

    We report empirical marginal coverage, width and weighted interval score (WIS). Candidate
    spread is not a calibrated predictive distribution. Rolling/ACI rules additionally observe
    revealed past targets online; compare their information access separately.
    '''), BOOT, code('''
    uq = load_snapshot('uncertainty')
    assert len(uq)==108*13*3
    table = uncertainty_summary()
    online = {'frozen_rolling','ttc_rolling','bounded_aci','candidate_rolling'}
    table['calibration'] = np.where(table.method.isin(online), 'online updates', 'fixed')
    display(table.sort_values(['calibration','relative_wis']))
    # WIS is repeated at each alpha in the raw CSV; count it once per cell/method.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout='constrained')
    ordered = table.sort_values('coverage90')
    color = ['#ce7731' if m in online else '#138c85' for m in ordered.method]
    axes[0].barh(ordered.method, ordered.coverage90, color=color)
    axes[0].axvline(.9, color='black', ls='--'); axes[0].set(xlabel='Mean empirical coverage', title='Nominal 90%')
    axes[1].barh(ordered.method, ordered.relative_wis, color=color)
    axes[1].axvline(1, color='black', ls='--'); axes[1].set(xlabel='Geometric relative WIS (Frozen split = 1)', title='Lower WIS is better')
    fig.suptitle('Teal: fixed calibration · orange: online updates', fontsize=11); plt.show()
    '''), md('''
    ## Measured interval excerpt
    The first saved example is Bolt-S / ETTh1 / seed 0 at the prespecified first test origin (channel OT).
    Bands below are its recorded 90% intervals in fitting-standardized units. An individual window cannot establish coverage; use all 108 cells above. Intervals are
    marginal, not simultaneous trajectory guarantees.
    '''), code('''
    example = pd.read_csv(checked_path('examples/interval_1.csv'))
    history = pd.read_csv(checked_path('examples/interval_1_history.csv'))
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), layout='constrained', sharex=True)
    methods = [('raw_candidates','Raw candidate quantiles'), ('frozen_split','Frozen split conformal'),
               ('candidate_quantile_cp','Candidate quantiles + conformal correction')]
    stats = []
    for ax, (method, label) in zip(axes, methods):
        low, high = example[method+'_a10_lower'], example[method+'_a10_upper']
        ax.plot(history.step, history.history, color='gray', lw=1, label='Observed')
        ax.fill_between(example.step, low, high, color='#138c85', alpha=.22, label='90% band')
        ax.plot(example.step, example.target, color='black', lw=1, label='Target')
        ax.plot(example.step, example.frozen if method=='frozen_split' else example.point, '--', color='#ce7731', lw=1, label='Point forecast')
        ax.axvline(0, color='gray', lw=.8); ax.set(title=label, ylabel='Standardized value')
        stats.append(dict(method=method, example_coverage=float(((example.target>=low)&(example.target<=high)).mean()),
                          example_mean_width=float((high-low).mean())))
    axes[0].legend(ncol=4, fontsize=8); axes[-1].set_xlabel('Step relative to forecast origin'); plt.show()
    display(pd.DataFrame(stats))
    '''), md('''
    Split conformal's finite-sample coverage statement requires exchangeability of calibration
    and future scores; nonstationary time series do not automatically satisfy it. The reported
    intervals are empirically evaluated, not claimed to have unconditional time-series coverage.
    All six saved examples can be plotted by changing the excerpt number above. Full fitting
    implementations live in `ttc4ts/uncertainty.py` and `scripts/run_uncertainty.py`.
    ''')])

    write('05_forecast_examples.ipynb', [md('''
    # 05 · Six measured forecast illustrations
    These are post-hoc **success illustrations**, selected using test targets. They are not a
    random sample, not an estimate of typical improvement, and not used to train/select a predictor.
    The `ttc4ts` column is the historical validation-selected method, whose actual deployed rule
    is printed below. It must not be relabeled as the primary fixed Learned-weights method.
    '''), BOOT, code('''
    manifest = json.loads(checked_path('examples/forecast_manifest.json').read_text())
    print(manifest['selection_rule'])
    metadata = pd.DataFrame(manifest['examples'])
    display(metadata[['number','model','dataset','channel','forecast_start','deployment_method','frozen_mse','ttc4ts_mse','gain']])
    '''), code('''
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), layout='constrained')
    checks = []
    for ax, item in zip(axes.flat, manifest['examples']):
        n = item['number']
        future = pd.read_csv(checked_path(f'examples/forecast_{n}.csv'))
        history = pd.read_csv(checked_path(f'examples/forecast_{n}_history.csv'))
        mse = float(np.mean((future.ttc4ts-future.target)**2))
        frozen_mse = float(np.mean((future.frozen-future.target)**2))
        assert np.isclose(mse, item['ttc4ts_mse'], rtol=1e-6)
        assert np.isclose(frozen_mse, item['frozen_mse'], rtol=1e-6)
        checks.append(dict(example=n, frozen_mse=frozen_mse, ttc4ts_mse=mse, reduction=1-mse/frozen_mse))
        ax.plot(history.step, history.history, color='gray', lw=1, label='Observed')
        ax.plot(future.step, future.target, color='black', lw=1, label='Target')
        ax.plot(future.step, future.frozen, '--', color='#ce7731', lw=1, label='Frozen')
        ax.plot(future.step, future.ttc4ts, color='#138c85', lw=1.4, label='Validation-selected TTC4TS')
        ax.axvline(0, color='gray', lw=.8)
        ax.set(title=f"{MODEL_LABELS[item['model']]} · {item['dataset']} · {item['channel']}",
               xlabel='Step relative to forecast origin', ylabel='Standardized value')
    axes[0,0].legend(fontsize=7, ncol=2); plt.show()
    display(pd.DataFrame(checks))
    '''), md('''
    Errors above are recomputed from the numeric forecast excerpts and checked against the
    saved manifest. The excerpt conversion retained floating-point values and discarded object
    date arrays; dates/origins/channel identities are supplied by the manifest. All source-file
    digests are retained. Full paper-wide summary metrics are in notebook 01.
    ''')])

    write('06_gpu_reproduction.ipynb', [md('''
    # 06 · Opt-in fresh GPU experiment and its measured reference
    **Default execution is offline.** This notebook displays the exact full-cell command and
    its already measured reference rows. Change `RUN_GPU = True` only after following
    `docs/reproduction.md`, downloading the exact data/model, and installing the GPU extras.
    One full Bolt-S / ETTh1 / H96 / seed-0 cell trains lightweight modules and runs all methods.
    It can take significant GPU time; the notebook does not automatically start all 432 cells.
    '''), BOOT, code('''
    RUN_GPU = False
    MODEL, DATASET, HORIZON, SEED = 'chronos-bolt-small', 'ETTh1', 96, 0
    reference = load_snapshot('main')
    reference = reference[(reference.model==MODEL) & (reference.dataset==DATASET) &
        (reference.horizon==HORIZON) & (reference.seed==SEED) & reference.method.isin(METHODS)]
    display(reference[['method','mse','mae','relative_mse','relative_mae']])
    command = [sys.executable, 'scripts/run_experiment.py', '--model', MODEL, '--dataset', DATASET,
        '--horizon', str(HORIZON), '--seed', str(SEED), '--output', 'results_notebook_reproduction',
        '--adapter-steps', '32', '--aggregator-steps', '160', '--budget-aggregator-steps', '40']
    # Print a portable command, never an interpreter's private machine path.
    print('python ' + ' '.join(command[1:]))
    '''), code('''
    if RUN_GPU:
        import subprocess, torch
        assert torch.cuda.is_available(), 'Install the GPU environment first.'
        subprocess.run([sys.executable, 'scripts/check_data.py', '--study', 'main'], cwd=ROOT, check=True)
        cell = f'{MODEL}_{DATASET}_h{HORIZON}_s{SEED}'
        destination = ROOT/'results_notebook_reproduction'/cell
        if destination.exists():
            raise FileExistsError('Choose a new output directory or inspect the existing cell before repeating.')
        subprocess.run(command, cwd=ROOT, check=True)
        metrics = json.loads((destination/'metrics.json').read_text())
        print('Fresh cell completed. Inspect its metrics and source hashes before comparison.')
        display(pd.DataFrame(metrics['results']).T)
    else:
        print('GPU branch disabled. The table above is the checked historical measurement, not a new run.')
    '''), md('''
    ## Full study commands
    See `docs/reproduction.md` for the 432-cell campaign, 72 Adapter-grid cells, 54 fixed
    Learned-weights budget cells, independent baselines and 108 UQ cells. Audits must pass before
    labeling any campaign complete. Some original Hub revisions are not retained, so a newly
    downloaded model can differ even with the same repository name. Compare the recorded
    frozen-state digest, package versions, dataset hashes, candidate names and split origins.

    Offline notebook execution is not evidence that this GPU branch has been run on a new machine.
    ''')])
    print('Built seven notebook sources; run tools/execute_notebooks.py to populate outputs.')


if __name__ == '__main__': main()
