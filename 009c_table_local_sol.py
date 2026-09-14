import xtrack as xt
import numpy as np

r0 = 0.13
ds_start = 1.7
ds_end = 2.13

sol_half_length = 1.6
r0_local = 0.13

# B0 = 3.
# B0_local_list = np.array([0.0001, 0.2, 0.4, 0.6, 0.8,
#                           1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
#                           2.2, 2.4, 2.6, 2.8, 3.0])
# l_local_list = np.array([0.4]) #, 0.6, 0.8])


B0 = 2.
B0_local_list = np.array([0.0001, 0.2, 0.4, 0.6, 0.8,
                          1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
                          2.2, 2.4, 2.6, 2.8, 3.0,
                          3.25, 3.5, 3.75, 4])
l_local_list = np.array([0.3, 0.4, 0.5])#, 0.6, 0.8])


cols = {str(ll): 0*B0_local_list for ll in l_local_list}
cols['B0_local'] = np.array(list(map(str, B0_local_list)))

tt = xt.Table(cols, index='B0_local')

for B0_local in B0_local_list:
    for ll in l_local_list:
        fname = (f'B0_{B0:.3f}_r0_{r0:.3f}_sol_half_length_{sol_half_length:.3f}'
                f'_ds_start_{ds_start:.3f}_ds_end_{ds_end:.3f}')
        if B0_local !=0:
            fname += f'_B0_local_{B0_local:.3f}_r0_local_{r0_local:.3f}_l_local_{ll:.3f}'
        fpath = 'results/' + fname + '.json'

        ddd = xt.json.load(fpath)
        tt[str(ll), str(B0_local)] = ddd['gemitt_y']
print()
print(f'{r0=:.3f} {ds_start=:.3f}')
print()
tt.show(digits=3)

import matplotlib.pyplot as plt
plt.close('all')

scan_title = (
    f'B0 = {B0:g} T, s_screen_sol = {sol_half_length:g} m, '
    f'ds_start = {ds_start:g} m, ds_end = {ds_end:g} m\n'
    f'r0 = {r0:g} m, r0_local = {r0_local:g} m'
)

# One curve
l_plot = 0.4
plt.figure(1)
plt.plot(B0_local_list, tt[str(l_plot)]*1e12)
plt.xlabel(r'$B_\text{local}$ [T]')
plt.ylabel('Vertical emittance [pm]')
plt.suptitle(f'{scan_title}, l_local = {l_plot:g} m', fontsize=8)
plt.ylim(0.4, None)
plt.show()

# All curves
plt.figure(2)
for ll in l_local_list:
    plt.plot(B0_local_list, tt[str(ll)]*1e12, label=f'l_local = {ll}')
plt.xlabel(r'$B_\text{local}$ [T]')
plt.ylabel('Vertical emittance [pm]')
plt.ylim(0.4, None)
plt.legend()
plt.suptitle(scan_title, fontsize=8)

plt.show()
