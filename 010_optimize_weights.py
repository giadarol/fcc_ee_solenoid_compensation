"""Optimize relative first-corrector weights with the local solenoid off.

Run with the py313 Python from this directory. The last weight is fixed at 1
to remove the common-scale degeneracy with the matched acbv1 knob. Signed
weights are allowed. Results and matching logs go to optimization_010_weights.
The production script is not modified by this driver.
"""
import ast
import argparse
import contextlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'optimization_010_weights'


def configured_weights():
    tree = ast.parse((ROOT / '010_parametric_version.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], 'id', '') == 'first_corr_segments')
    weights = np.array([seg['weight'] for seg in ast.literal_eval(node.value)])
    if len(weights) < 2 or weights[-1] == 0:
        raise ValueError('Need at least two segments and a nonzero last weight')
    return weights / weights[-1]


def source(weights, reusable=False, radiation=False):
    tree = ast.parse((ROOT / '010_parametric_version.py').read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], 'id', None)
        overrides = dict(B0_local=0., plot=False, radiation_analysis=radiation)
        if name in overrides:
            node.value = ast.Constant(overrides[name])
        elif name == 'first_corr_segments':
            segments = ast.literal_eval(node.value)
            for segment, weight in zip(segments, weights, strict=True):
                segment.pop('strength', None)
                segment['weight'] = float(weight)
            node.value = ast.parse(repr(segments), mode='eval').body
    src = ast.unparse(tree)
    if reusable:
        src = src.replace('opt_orbit.generate_knob()', 'pass')
        src = src.replace('opt_optics.generate_knob()', 'pass')
        start = src.index("    line[f'on_sol_corr_{ip_name}'] = 0")
        end = src.index("line['on_sol_ipa'] = 1", start)
        src = src[:start] + src[end:]
    return compile(src, str(ROOT / '010_parametric_version.py'), 'exec')


class Evaluator:
    def __init__(self):
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        (DIRECTORY / 'pid').write_text(str(os.getpid()))
        self.weights = configured_weights()
        (DIRECTORY / '010_source.py').write_text(
            (ROOT / '010_parametric_version.py').read_text())
        self.log = (DIRECTORY / 'matching.log').open('w', buffering=1)
        self.history = (DIRECTORY / 'history.jsonl').open('w', buffering=1)
        self.ns = {'__name__': '__main__'}
        with contextlib.redirect_stdout(self.log), contextlib.redirect_stderr(self.log):
            exec(source(self.weights, reusable=True), self.ns)
        self.env = self.ns['env']
        self.line = self.ns['line']
        self.opts = [self.ns['opt_orbit'], self.ns['opt_optics']]
        self.initial = {v.name: float(self.env[v.name])
                        for opt in self.opts for v in opt.vary}
        self.best = np.inf
        self.count = 0
        self.cache = {}

    def __call__(self, ratios):
        if (DIRECTORY / 'STOP').exists():
            raise KeyboardInterrupt('Stopped by checkpoint request')
        key = tuple(map(float, ratios))
        if key in self.cache:
            return self.cache[key]
        weights = [*key, 1.]
        self.count += 1
        start = time.time()
        result = dict(index=self.count, weights=weights)
        try:
            with contextlib.redirect_stdout(self.log), contextlib.redirect_stderr(self.log):
                print('\nEVALUATION', self.count, weights, flush=True)
                for side in ['right', 'left']:
                    for i, weight in enumerate(weights, start=1):
                        self.env[f'acbv1_seg{i}_sol_{side}_ipa'] = (
                            weight * self.env.ref[f'acbv1_sol_{side}_ipa'])
                for name, value in self.initial.items():
                    self.env[name] = value
                for _ in range(3):
                    for opt in self.opts:
                        if not np.all(opt.target_status(ret=True).tol_met):
                            opt.solve()
                if not all(np.all(opt.target_status(ret=True).tol_met)
                           for opt in self.opts):
                    raise ValueError('Orbit/optics matching tolerances not met')
                tw = self.line.twiss4d(strengths=True, radiation_integrals=True)
            value = float(tw.rad_int_eq_gemitt_y) * 4e12
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid emittance: {value}')
            result['emittance_pm_4ip'] = value
            result['matched_knobs'] = {
                side: float(self.env[f'acbv1_sol_{side}_ipa'])
                for side in ['right', 'left']}
            if value < self.best:
                self.best = value
                (DIRECTORY / 'best.json').write_text(json.dumps(result, indent=2))
        except Exception:
            result['error'] = traceback.format_exc()
            value = 1e6
        result['seconds'] = time.time() - start
        self.history.write(json.dumps(result) + '\n')
        self.cache[key] = value
        print(f'{self.count}: weights={weights}, emittance={value:.10g} pm, '
              f'best={self.best:.10g}, seconds={result["seconds"]:.1f}', flush=True)
        return value


def fresh(weights, radiation=False):
    ns = {'__name__': '__main__'}
    with (DIRECTORY / 'fresh.log').open('w') as log:
        with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            exec(source(weights, radiation=radiation), ns)
    return dict(weights=weights, emittance_pm_4ip=ns['out']['gemitt_y'] * 1e12,
                matched_knobs={side: float(ns['env'][f'acbv1_sol_{side}_ipa'])
                               for side in ['right', 'left']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='optimization_010_weights')
    parser.add_argument('--start', nargs='+', type=float)
    parser.add_argument('--maxfev', type=int, default=300)
    parser.add_argument('--method', choices=['Nelder-Mead', 'BFGS'], default='Nelder-Mead')
    args = parser.parse_args()
    DIRECTORY = ROOT / args.directory
    evaluator = Evaluator()
    baseline = evaluator(evaluator.weights[:-1])
    check = fresh(evaluator.weights.tolist())
    np.testing.assert_allclose(baseline, check['emittance_pm_4ip'], rtol=2e-5)
    (DIRECTORY / 'baseline.json').write_text(json.dumps(check, indent=2))
    initial = np.asarray(args.start if args.start is not None else evaluator.weights[:-1])
    if len(initial) != len(evaluator.weights) - 1:
        raise ValueError('--start must have one ratio per segment except the last')
    if args.method == 'Nelder-Mead':
        options = dict(initial_simplex=np.vstack([
            initial, initial + 0.2 * np.eye(len(initial))]),
            maxfev=args.maxfev, xatol=2e-5, fatol=1e-7)
    else:
        options = dict(eps=1e-4, gtol=2e-5, maxiter=40)
    result = minimize(evaluator, initial, method=args.method, options=options)
    (DIRECTORY / 'optimizer.json').write_text(json.dumps(dict(
        success=bool(result.success), message=str(result.message),
        ratios=result.x.tolist(), emittance_pm_4ip=float(result.fun),
        nfev=int(result.nfev)), indent=2))
    best = json.loads((DIRECTORY / 'best.json').read_text())
    check = fresh(best['weights'], radiation=True)
    np.testing.assert_allclose(best['emittance_pm_4ip'],
                               check['emittance_pm_4ip'], rtol=2e-5)
    (DIRECTORY / 'validated.json').write_text(json.dumps(check, indent=2))
    print('VALIDATED', check, flush=True)
