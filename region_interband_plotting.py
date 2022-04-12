import numpy as np
from LosslessFunctions import region0
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter

M = 400  # the grid size
m = 1  # it's normalized
Vx = np.linspace(0, 140e-3, M)
Ed = np.linspace(-10e-3, 140e-3, M)

# forming the region0 (without the interband)
R0 = np.zeros((M, M))
wp_min = np.empty((M, M))
wp_min[:] = np.nan

for i in range(M):
    print(i)
    for j in range(M):
        R0[i, j] = region0(Vx[j], Ed[i], m)
        if R0[i, j] == 1:
            wp_min[i, j] = (Ed[i] + np.sqrt(Ed[i]**2 + 4*Vx[j]**2)) / 2
    print(i)

# reading the three region for 3 normalized l
p = 5  # padding for smoothing out

f1 = open('data/regions/norm_R_L1e2.txt')
f2 = open('data/regions/norm_R_L5e2.txt')
f3 = open('data/regions/norm_R_L10e2.txt')

R1 = f1.read()
R1 = R1.split()
R1 = np.array(list(map(float, R1)))
R1 = np.reshape(R1, (M, M))
R1 = np.pad(R1, ((p, p), (p, p)), 'constant', constant_values=((-1, -1), (-1, -1)))
R1 = uniform_filter(R1, size=p)
R1 = R1[p:M+p, p:M+p]

R2 = f2.read()
R2 = R2.split()
R2 = np.array(list(map(float, R2)))
R2 = np.reshape(R2, (M, M))
R2 = np.pad(R2, ((p, p), (p, p)), 'constant', constant_values=((-1, -1), (-1, -1)))
R2 = uniform_filter(R2, size=p)
R2 = R2[p:M+p, p:M+p]

R3 = f3.read()
R3 = R3.split()
R3 = np.array(list(map(float, R3)))
R3 = np.reshape(R3, (M, M))
R3 = np.pad(R3, ((p, p), (p, p)), 'constant', constant_values=((-1, -1), (-1, -1)))
R3 = uniform_filter(R3, size=p)
R3 = R3[p:M+p, p:M+p]

f1.close()
f2.close()
f3.close()


# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

plt.figure(figsize=(7.6, 7.6))
# plt.plot(1e3*Vx, 1e3*Vx, ':', color='grey')  # the diagonal
# plot wp_min
plt.imshow(1e3*wp_min, extent=[1e3*Vx.min(), 1e3*Vx.max(), 1e3*Ed.min(), 1e3*Ed.max()],
           vmin=0, vmax=1e3*np.nanmax(wp_min),
           origin='lower')
#           , cmap='plasma')
cbar = plt.colorbar(pad=0.048, fraction=0.048)
cbar.ax.tick_params(labelsize=16)
cbar.set_label(r'$\min(\hbar\tilde{\omega}_p)$', fontsize=26, rotation=270, labelpad=27)
# plot the regions
CS3 = plt.contour(1e3*Vx, 1e3*Ed, R3, levels=[0], colors='C1', linewidths=2, linestyles=':')
CS3.collections[0].set_label(r'$\tilde{\ell}$ = 500')
CS2 = plt.contour(1e3*Vx, 1e3*Ed, R2, levels=[0], colors='C3', linewidths=2, linestyles='-.')
CS2.collections[0].set_label(r'$\tilde{\ell}$ = 250')
CS1 = plt.contour(1e3*Vx, 1e3*Ed, R1, levels=[0], colors='C4', linewidths=2, linestyles='--')
CS1.collections[0].set_label(r'$\tilde{\ell}$ = 50')
CS0 = plt.contour(1e3*Vx, 1e3*Ed, R0, levels=[0], colors='k', linewidths=2)
CS0.collections[0].set_label(r'$\tilde{\ell}$ = 0')
plt.axis('scaled')
plt.tick_params(labelsize=16)
plt.xlabel(r'$\tilde{V}\sqrt{x}$', fontsize=26)
plt.ylabel(r'$\tilde{E}_d$', fontsize=26)
plt.legend(frameon=False, fontsize=20,
           bbox_to_anchor=(0.58, 0.14), labelspacing=-2.5)

'''THE SLAB'''
# ZnCdTe:O parameters
m_zn = 0.117  # in me
m_cd = 0.09  # in me
V_zn = 2.8  # in eV
V_cd = 2.2  # in eV
Ed_zn = -0.27  # in eV
Ed_cd = 0.38  # in eV
C_zncd = 0.46  # bowing in eV

y = np.arange(0.26, 0.30, 1e-3)
x_6 = 1e-6
x_5 = 1e-5

m = m_cd*y + m_zn*(1-y)
vx = (V_cd*y + V_zn*(1-y)) / m
ed = ((Ed_cd*y + Ed_zn*(1-y)) + C_zncd*y*(1-y)) / m
# plot the slab
plt.plot(1e3*vx*np.sqrt(x_6), 1e3*ed, '--', color='C2')
plt.plot(1e3*vx*np.sqrt(x_5), 1e3*ed, '--', color='C2')
plt.text(69, -7, '$x = 10^{-5}$', rotation=90, fontsize=20, color='C2')
plt.text(25, 112, '$x = 10^{-6}$', rotation=-90, fontsize=20, color='C2')

plt.tight_layout()
plt.show()
