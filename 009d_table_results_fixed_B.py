import xtrack as xt
import numpy as np

r0 = 0.13
ds_end = 2.13
B0 = 2.

s_screen_sol_list = np.array([1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6])
ds_start_list = np.array([1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8])

cols = {str(ds): 0 * s_screen_sol_list for ds in ds_start_list}
cols['s_screen_sol'] = np.array(list(map(str, s_screen_sol_list)))

tt = xt.Table(cols, index='s_screen_sol')

for ss in s_screen_sol_list:
    for ds_start in ds_start_list:
        fname = (f'B0_{B0:.3f}_r0_{r0:.3f}_sol_half_length_{ss:.3f}'
                 f'_ds_start_{ds_start:.3f}_ds_end_{ds_end:.3f}.json')
        ddd = xt.json.load('results/' + fname)
        tt[str(ds_start), str(ss)] = ddd['gemitt_y']

print()
print(f'{B0=:.3f} {r0=:.3f} {ds_end=:.3f}')
print()
tt.show(digits=3)
