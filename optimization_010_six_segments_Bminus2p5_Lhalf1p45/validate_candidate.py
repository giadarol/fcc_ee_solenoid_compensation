"""Fresh matching and radiation validation of a saved search candidate."""
import ast
import contextlib
import json
from pathlib import Path
import sys

import numpy as np


directory = Path(__file__).resolve().parent / sys.argv[1]
sys.path.insert(0, str(directory.parent.parent))
best = json.loads((directory / 'best.json').read_text())
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
source = ast.unparse(tree)
# Check both matches before generating either knob, which resets its corrections.
checks = '\n    '.join(
    f"matching_checks[ip_name + '_{kind}'] = bool(np.all("
    f'opt_{kind}.target_status(ret=True).tol_met))'
    for kind in ('orbit', 'optics'))
source = source.replace('opt_orbit.generate_knob()',
                        checks + '\n    opt_orbit.generate_knob()')
(directory / 'validation_source.py').write_text(source + '\n')
ns = {'__name__': '__main__', 'matching_checks': {}}
with (directory / 'validation.log').open('w') as log:
    with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        exec(compile(source, str(directory / 'validation_source.py'), 'exec'), ns)
assert all(ns['matching_checks'].values()), ns['matching_checks']
emittance = ns['out']['gemitt_y'] * 1e12
np.testing.assert_allclose(emittance, best['emittance_pm_4ip'], rtol=2e-5)
result = dict(ns['out'], emittance_pm_4ip=emittance,
              weights=best['weights'], matching_checks=ns['matching_checks'],
              matched_knobs={side: float(ns['env'][f'acbv1_sol_{side}_ipa'])
                             for side in ('right', 'left')},
              radiation_analysis_completed=True, status='PASS')
(directory / 'validated.json').write_text(json.dumps(result, indent=2) + '\n')
print(directory.name, emittance, 'pm; PASS', flush=True)
