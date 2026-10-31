"""Summarize saved 010 optima, rerun them, and plot emittance versus local field.

Run with the py313 Python, e.g.:
    python 011_rerun_field_scan.py --workers 4

Each case uses its archived source and selected weights in a fresh process.
Orbit/optics matching and radiation analysis are repeated; the weights are not
re-optimized. The working 010_parametric_version.py is not modified.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

ROOT = Path(__file__).resolve().parent
PARAMETERS = ('B0', 'r0', 'theta', 'sol_half_length', 'B0_local',
              'r0_local', 'l_local')


def collect_cases():
    cases = []
    for selected_path in sorted(ROOT.glob('optimization_010*/selected_best.json')):
        selected = json.loads(selected_path.read_text())
        snapshot = ROOT / Path(selected['source_result']).parent / '010_source.py'
        source = snapshot.read_text()
        assignments = {
            node.targets[0].id: node.value for node in ast.parse(source).body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        params = {name: ast.literal_eval(assignments[name]) for name in PARAMETERS}
        for name in PARAMETERS:
            if name in selected and selected[name] != params[name]:
                raise ValueError(f'{selected_path}: {name} differs from snapshot')
        segments = ast.literal_eval(assignments['first_corr_segments'])
        for segment, weight in zip(segments, selected['weights'], strict=True):
            segment.pop('strength', None)
            segment['weight'] = weight
        if 'segments' in selected:
            assert [list(s['s_range']) for s in segments] == [
                list(s['s_range']) for s in selected['segments']]
        cases.append(dict(
            case_id=selected_path.parent.name, parameters=params,
            segments=segments, saved_emittance_pm_4ip=selected['emittance_pm_4ip'],
            saved_status=selected.get('status', ''),
            selected_file=str(selected_path.relative_to(ROOT)),
            source_snapshot=str(snapshot.relative_to(ROOT)),
            source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        ))
    if not cases:
        raise ValueError('No selected optimization results found')
    return sorted(cases, key=lambda c: (-len(c['segments']), c['parameters']['B0_local']))


def run_case(case, output_dir):
    import numpy as np

    os.chdir(ROOT)
    run_dir = Path(output_dir) / 'runs' / case['case_id']
    run_dir.mkdir(parents=True, exist_ok=True)
    source = (ROOT / case['source_snapshot']).read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == case['source_sha256']
    tree = ast.parse(source)
    overrides = dict(case['parameters'], plot=False, radiation_analysis=True,
                     first_corr_segments=case['segments'])
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in overrides:
                node.value = ast.parse(repr(overrides[name]), mode='eval').body
    code = ast.unparse(tree)
    needle = '    opt_orbit.generate_knob()'
    assert code.count(needle) == 1
    code = code.replace(needle,
        "    rerun_match_checks[ip_name] = {\n"
        "        'orbit': bool(np.all(opt_orbit.target_status(ret=True).tol_met)),\n"
        "        'optics': bool(np.all(opt_optics.target_status(ret=True).tol_met)),\n"
        "    }\n" + needle)
    effective_source = run_dir / 'run_source.py'
    effective_source.write_text(code + '\n')
    start = time.time()
    ns = {'__name__': '__main__', 'rerun_match_checks': {}}
    result = dict(case_id=case['case_id'])
    try:
        with (run_dir / 'run.log').open('w', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                exec(compile(code, str(effective_source), 'exec'), ns)
        value = float(ns['out']['gemitt_y']) * 1e12
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'Invalid emittance: {value}')
        result.update(
            rerun_emittance_pm_4ip=value,
            difference_pm=value - case['saved_emittance_pm_4ip'],
            relative_difference=value / case['saved_emittance_pm_4ip'] - 1,
            matching_checks=ns['rerun_match_checks'],
            radiation_analysis_completed='tw_rad' in ns,
            matched_knobs={side: float(ns['env'][f'acbv1_sol_{side}_ipa'])
                           for side in ('right', 'left')},
        )
        assert all(all(check.values()) for check in ns['rerun_match_checks'].values())
        assert result['radiation_analysis_completed']
        np.testing.assert_allclose(value, case['saved_emittance_pm_4ip'], rtol=2e-5)
        result['status'] = 'PASS'
    except Exception:
        result.update(status='FAIL', error=traceback.format_exc())
    result['seconds'] = time.time() - start
    (run_dir / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def write_report(cases, results, output_dir):
    lines = [
        'SOLENOID COMPENSATION: SAVED OPTIMA AND FRESH RERUNS',
        '==================================================',
        'Emittance [pm] = 4 * twiss4d.rad_int_eq_gemitt_y * 1e12 (four-IP convention).',
        'Each rerun repeats orbit/optics matching and full radiation analysis at',
        'the saved weights; it does not perform another weight optimization.',
        'The plotted quantity is the 4D radiation-integral estimate, not the 6D emittance.',
        'PASS requires matching tolerances, completed radiation analysis, and',
        'agreement with the saved emittance within relative tolerance 2e-5.',
        'The earlier three-segment result is listed separately from the six-segment scan.',
        'The zero-field runs retain l_local = 0.5884884906480151 m from their snapshots;',
        'all nonzero-field runs use l_local = 0.6 m. Length has no field effect at B0_local = 0.',
        '',
        'Segments  B0_local[T]   Saved ey[pm]     Rerun ey[pm]      Difference[pm]  Status',
    ]
    for case in cases:
        r = results.get(case['case_id'], {})
        value = r.get('rerun_emittance_pm_4ip')
        diff = r.get('difference_pm')
        lines.append(f"{len(case['segments']):8d} {case['parameters']['B0_local']:12.1f} "
                     f"{case['saved_emittance_pm_4ip']:16.10f} "
                     f"{value if value is not None else float('nan'):16.10f} "
                     f"{diff if diff is not None else float('nan'):16.6g} "
                     f"{r.get('status', 'PENDING')}")
    lines += ['', 'FULL PARAMETERS AND WEIGHTS (in increasing positive s from the IP)',
              'Left-hand ranges are mirrored; each side has its own matched acbv1 knob.',
              'Weights multiply integrated k0sl. Last-segment weight is normalized to 1.']
    for case in cases:
        lines += ['', case['case_id'], '  Parameters: ' + repr(case['parameters'])]
        for i, seg in enumerate(case['segments'], 1):
            lo, hi = seg['s_range']
            lines.append(f"  Segment {i}: ({lo:.12g}, {hi:.12g}) m; weight = {seg['weight']:.17g}")
        lines += ['  Selected result: ' + case['selected_file'],
                  '  Source snapshot: ' + case['source_snapshot'],
                  '  Snapshot SHA256: ' + case['source_sha256'],
                  '  Original status: ' + case['saved_status']]
        r = results.get(case['case_id'])
        if r:
            lines.append('  Rerun: ' + json.dumps(r, sort_keys=True))
    (output_dir / 'results.txt').write_text('\n'.join(lines) + '\n')
    (output_dir / 'results.json').write_text(json.dumps(
        dict(cases=cases, reruns=results), indent=2) + '\n')
    with (output_dir / 'results.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['case', 'segments', 'B0_local_T', 'saved_emittance_pm_4ip',
                         'rerun_emittance_pm_4ip', 'relative_difference', 'status',
                         *[f'weight_{i}' for i in range(1, 7)]])
        for case in cases:
            r = results.get(case['case_id'], {})
            weights = [s['weight'] for s in case['segments']]
            writer.writerow([case['case_id'], len(weights), case['parameters']['B0_local'],
                             case['saved_emittance_pm_4ip'], r.get('rerun_emittance_pm_4ip', ''),
                             r.get('relative_difference', ''), r.get('status', 'PENDING'),
                             *weights, *[''] * (6 - len(weights))])


def plot_results(cases, results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    scan = sorted((c for c in cases if len(c['segments']) == 6),
                  key=lambda c: abs(c['parameters']['B0_local']))
    x = [abs(c['parameters']['B0_local']) for c in scan]
    y = [results[c['case_id']]['rerun_emittance_pm_4ip'] for c in scan]
    fig, ax = plt.subplots(figsize=(8.2, 5.2), layout='constrained')
    ax.plot(x, y, 'o-', color='#205a91', lw=1.8, ms=5.5, label='Six segments — fresh reruns')
    ax.set(xlabel=r'$|B_{0,\mathrm{local}}|$ [T]',
           ylabel=r'Vertical emittance $\varepsilon_y$ [pm] (4 IPs)',
           title='Solenoid compensation: optimized weights at each field',
           xlim=(min(x) - 0.15, max(x) + 0.15), ylim=(0, max(y) * 1.1))
    ax.set_xticks([i / 2 for i in range(0, 11)])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc='upper left')
    fig.savefig(output_dir / 'emittance_vs_B0_local.png', dpi=200)
    fig.savefig(output_dir / 'emittance_vs_B0_local.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output', default='010_field_scan_results')
    args = parser.parse_args()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = '1'
    os.environ['MPLBACKEND'] = 'Agg'
    output_dir = ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(output_dir / 'matplotlib_cache')
    cases, results = collect_cases(), {}
    write_report(cases, results, output_dir)
    print(f'Saved {len(cases)} historical results to {output_dir / "results.txt"}', flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             max_tasks_per_child=1) as pool:
        pending = {pool.submit(run_case, case, str(output_dir)): case for case in cases}
        for future in as_completed(pending):
            case = pending[future]
            result = future.result()
            results[case['case_id']] = result
            write_report(cases, results, output_dir)
            print(f"{len(results)}/{len(cases)} {case['case_id']}: {result['status']}; "
                  f"ey={result.get('rerun_emittance_pm_4ip')} pm", flush=True)
    failed = [name for name, result in results.items() if result['status'] != 'PASS']
    if failed:
        raise RuntimeError(f'Rerun checks failed: {failed}; see per-case result.json files')
    plot_results(cases, results, output_dir)
    print(f'All {len(cases)} reruns passed. Plot saved in {output_dir}.', flush=True)


if __name__ == '__main__':
    main()
