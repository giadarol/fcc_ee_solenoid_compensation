"""Test zero integrated Bs with local coils and no first-corrector powering.

Use the geometry from 010_parametric_version.py without editing that file.
Cancel the integrated longitudinal field over the common modeled interval
[-2.399, 2.399] m, using the tracking slice grid and zeroed endpoint fields.
Remote compensation-solenoid field is explicitly zero. The other orbit and
optics correctors retain the matching procedure in 010.
"""
import ast
import contextlib
import json
from pathlib import Path
import traceback

import numpy as np
from scipy.integrate import quad

from tilted_solenoid import TiltedSolenoid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / '012_fully_local_results'


def main():
    OUT.mkdir(exist_ok=True)
    source = (ROOT / '010_parametric_version.py').read_text()
    (OUT / '010_source.py').write_text(source)
    tree = ast.parse(source)
    names = {'B0', 'r0', 'theta', 'sol_half_length', 'r0_local', 'l_local',
             'first_corr_segments'}
    params = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
              if isinstance(n, ast.Assign) and getattr(n.targets[0], 'id', '') in names}
    for segment in params['first_corr_segments']:
        segment.pop('strength', None)
        segment['weight'] = 0.0

    s = np.linspace(-2.399, 2.399, 201)
    edges = np.array([edge for seg in params['first_corr_segments'] for edge in seg['s_range']])
    s = s[~np.any(np.isclose(s[:, None], np.r_[-edges, edges], rtol=0, atol=1e-12), axis=1)]
    s = np.unique(np.r_[s, -edges, edges])
    main_sol = TiltedSolenoid(L=2 * params['sol_half_length'], a=params['r0'],
                             B0=params['B0'], theta=params['theta'])
    local = [TiltedSolenoid(L=params['l_local'], a=params['r0_local'], B0=1.,
                           theta=params['theta'],
                           z0=sign * (params['sol_half_length'] + params['l_local']/2))
             for sign in [-1, 1]]

    def fields(positions):
        positions = np.atleast_1d(positions)
        zero = np.zeros_like(positions)
        main = np.array(main_sol.get_field(zero, zero, positions))
        unit_local = sum(np.array(sol.get_field(zero, zero, positions)) for sol in local)
        return main, unit_local

    main_field, local_unit_field = fields(s)
    bs_main = main_field[2].copy()
    bs_local_unit = local_unit_field[2].copy()
    bs_main[[0, -1]] = 0.
    bs_local_unit[[0, -1]] = 0.
    integral_main = float(np.trapezoid(bs_main, s))
    integral_unit = float(np.trapezoid(bs_local_unit, s))
    b_local = -integral_main / integral_unit
    residual = float(np.trapezoid(bs_main + b_local * bs_local_unit, s))
    assert abs(residual) < 1e-12

    analytic_main = quad(lambda pos: fields(pos)[0][2, 0], s[0], s[-1], epsabs=1e-10)[0]
    analytic_unit = quad(lambda pos: fields(pos)[1][2, 0], s[0], s[-1], epsabs=1e-10)[0]
    result = dict(
        parameters={**params, 'B0_local': b_local},
        interval_m=[float(s[0]), float(s[-1])],
        main_integral_Tm=integral_main,
        local_pair_integral_per_T_m=integral_unit,
        local_pair_integral_Tm=b_local * integral_unit,
        grid_residual_integral_Tm=residual,
        analytic_unclipped_residual_integral_Tm=analytic_main + b_local * analytic_unit,
        analytic_unclipped_canceling_B0_local_T=-analytic_main / analytic_unit,
        weights=[0.] * len(params['first_corr_segments']),
        convention='Integrated Bs of the tracking model, including its zeroed endpoints.',
        endpoint_assignment='Explicit ks_profile[1] = 0 keeps the exit field zero when knobs are toggled.',
        remaining_corrections='Standard orbit and optics correctors are rematched; remote compensation solenoids are off.',
    )
    (OUT / 'field_solution.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'B0_local = {b_local:.12g} T; grid integral = {residual:.3g} T m', flush=True)

    overrides = dict(B0_local=b_local, first_corr_segments=params['first_corr_segments'],
                     plot=False, radiation_analysis=True)
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
            if name in overrides:
                node.value = ast.parse(repr(overrides[name]), mode='eval').body
    code = ast.unparse(tree)
    # Negative-index assignment leaves the original positive-index expression
    # active in this xdeps version: toggling on_sol would restore the exit field.
    # Explicit index 1 removes that dependency and preserves the zero endpoint.
    assert code.count('.ks_profile[-1] = 0') in (0, 2)
    code = code.replace('.ks_profile[-1] = 0', '.ks_profile[1] = 0')
    remote_assignment = "env[f'field_comp_sol_{ip_name}'] = -ksol_l_main_solenoid / ksol_l_comp_solenoid / 2"
    assert (code.count(remote_assignment) == 1
            or "env[f'field_comp_sol_{ip_name}'] = 0.0" in code)
    code = code.replace(remote_assignment, "env[f'field_comp_sol_{ip_name}'] = 0.0")
    needle = '    opt_orbit.generate_knob()'
    assert code.count(needle) == 1
    code = code.replace(needle,
        "    local_test_matching[ip_name] = {\n"
        "        'orbit': bool(np.all(opt_orbit.target_status(ret=True).tol_met)),\n"
        "        'optics': bool(np.all(opt_optics.target_status(ret=True).tol_met)),\n"
        "    }\n" + needle)
    (OUT / 'run_source.py').write_text(code + '\n')
    ns = {'__name__': '__main__', 'local_test_matching': {}}
    try:
        with (OUT / 'run.log').open('w', buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                exec(compile(code, str(OUT / 'run_source.py'), 'exec'), ns)
        env = ns['env']
        result.update(
            emittance_pm_4ip=float(ns['out']['gemitt_y']) * 1e12,
            tracking_integral_Tm=float(ns['ksol_l_main_solenoid'] * ns['rigidity0']),
            remote_compensation_field_T=float(env['field_comp_sol_ipa']),
            doublet_rotation_rad=float(env['phi_rot_doublet_ipa']),
            main_endpoint_Bs_T=[
                float(env.get(ns['ele_names'][0]).ks_profile[0] * ns['rigidity0']),
                float(env.get(ns['ele_names'][-1]).ks_profile[1] * ns['rigidity0'])],
            matching_checks=ns['local_test_matching'],
            radiation_analysis_completed='tw_rad' in ns,
            first_corrector_knobs={side: float(env[f'acbv1_sol_{side}_ipa'])
                                   for side in ['right', 'left']},
            max_abs_y_m=float(np.max(np.abs(ns['tw4d'].y))),
            max_abs_dy_m=float(np.max(np.abs(ns['tw4d'].dy))),
        )
        assert not ns['vary_first_corr']
        assert abs(result['tracking_integral_Tm']) < 1e-11
        assert result['remote_compensation_field_T'] == 0.
        assert result['main_endpoint_Bs_T'] == [0., 0.]
        assert all(all(check.values()) for check in result['matching_checks'].values())
        assert result['radiation_analysis_completed']
        for side in ['right', 'left']:
            for i in range(1, len(params['first_corr_segments']) + 1):
                assert float(env[f'acbv1_seg{i}_sol_{side}_ipa']) == 0.
        result['status'] = 'PASS'
    except Exception:
        result.update(status='FAIL', error=traceback.format_exc())
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    (OUT / 'result.txt').write_text(
        'Fully local solenoid compensation test\n'
        '=====================================\n'
        + json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if result['status'] != 'PASS':
        raise RuntimeError('Local compensation test failed; see result.json and run.log')


if __name__ == '__main__':
    main()
