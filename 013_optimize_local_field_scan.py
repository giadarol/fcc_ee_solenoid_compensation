"""Optimize weights at one local field using an isolated copy of 010.

Run from the project directory with py313. The production configuration is
unchanged. Each run saves inputs, optimization history and fresh validation.
"""
import argparse
import ast
import contextlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent


def run(args):
    directory = ROOT / args.directory
    directory.mkdir(parents=True, exist_ok=True)
    source_path = ROOT / '010_parametric_version.py'
    tree = ast.parse(source_path.read_text())
    seed = json.loads(Path(args.seed).read_text())['weights'] if args.seed else None
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        name = getattr(node.targets[0], 'id', '')
        if name == 'B0_local':
            node.value = ast.Constant(args.field)
        elif name == 'first_corr_segments' and seed is not None:
            segments = ast.literal_eval(node.value)
            for segment, weight in zip(segments, seed, strict=True):
                segment.pop('strength', None)
                segment['weight'] = weight
            node.value = ast.parse(repr(segments), mode='eval').body
    inputs = directory / 'input'
    inputs.mkdir(exist_ok=True)
    (inputs / source_path.name).write_text(ast.unparse(tree) + '\n')
    spec = importlib.util.spec_from_file_location('weight_driver',
                                                ROOT / '010_optimize_weights.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    driver.ROOT = inputs
    driver.DIRECTORY = directory
    evaluator = driver.Evaluator()
    baseline = evaluator(evaluator.weights[:-1])
    (directory / 'initial.json').write_text(json.dumps(dict(
        weights=evaluator.weights.tolist(), emittance_pm_4ip=baseline,
        B0_local=args.field), indent=2) + '\n')
    best_at_last_progress = evaluator.best
    last_progress = evaluator.count

    class SearchComplete(Exception):
        pass

    def objective(ratios):
        nonlocal best_at_last_progress, last_progress
        if evaluator.count >= args.max_evaluations:
            raise SearchComplete('Evaluation budget reached')
        if evaluator.count - last_progress >= 30:
            raise SearchComplete('No improvement above 5e-6 pm in 30 evaluations')
        value = evaluator(ratios)
        if best_at_last_progress - evaluator.best > 5e-6:
            best_at_last_progress = evaluator.best
            last_progress = evaluator.count
        return value

    try:
        result = minimize(objective, evaluator.weights[:-1], method='BFGS',
                          options=dict(eps=args.eps, gtol=0.002, maxiter=30))
        status = dict(success=bool(result.success), message=str(result.message))
    except (SearchComplete, KeyboardInterrupt) as exc:
        status = dict(success=False, message=str(exc))
    status['evaluations'] = evaluator.count
    (directory / 'optimizer.json').write_text(json.dumps(status, indent=2) + '\n')

    best = json.loads((directory / 'best.json').read_text())
    # Rebuild from the saved source for independent orbit, optics and radiation checks.
    tree = ast.parse((directory / '010_source.py').read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        name = getattr(node.targets[0], 'id', '')
        if name in ('plot', 'radiation_analysis'):
            node.value = ast.Constant(name == 'radiation_analysis')
        elif name == 'first_corr_segments':
            segments = ast.literal_eval(node.value)
            for segment, weight in zip(segments, best['weights'], strict=True):
                segment['weight'] = weight
            node.value = ast.parse(repr(segments), mode='eval').body
    src = ast.unparse(tree)
    checks = '\n    '.join(
        f"matching_checks[ip_name + '_{kind}'] = bool(np.all("
        f'opt_{kind}.target_status(ret=True).tol_met))'
        for kind in ('orbit', 'optics'))
    src = src.replace('opt_orbit.generate_knob()',
                      checks + '\n    opt_orbit.generate_knob()')
    (directory / 'validation_source.py').write_text(src + '\n')
    ns = {'__name__': '__main__', 'matching_checks': {}}
    with (directory / 'validation.log').open('w') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            exec(compile(src, str(directory / 'validation_source.py'), 'exec'), ns)
    assert all(ns['matching_checks'].values()), ns['matching_checks']
    emittance = ns['out']['gemitt_y'] * 1e12
    np.testing.assert_allclose(emittance, best['emittance_pm_4ip'], rtol=2e-5)
    result = dict(ns['out'], weights=best['weights'], emittance_pm_4ip=emittance,
                  matching_checks=ns['matching_checks'],
                  matched_knobs={side: float(ns['env'][f'acbv1_sol_{side}_ipa'])
                                 for side in ('right', 'left')},
                  radiation_analysis_completed=True, status='PASS')
    (directory / 'validated.json').write_text(json.dumps(result, indent=2) + '\n')
    print('VALIDATED', args.field, emittance, 'pm', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--field', type=float, required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--seed', help='JSON result supplying initial weights')
    parser.add_argument('--eps', type=float, default=0.0003)
    parser.add_argument('--max-evaluations', type=int, default=150)
    run(parser.parse_args())
