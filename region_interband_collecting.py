import numpy as np
from matplotlib import pyplot as plt
from LosslessFunctions import region0, region

res = 500
M = 400
m = 1
L0 = 1973e-6  # hbar / 2me*c in A
Lm = L0 / m
L = 1e2 * Lm
Vx = np.linspace(0, 140e-3, M)
Ed = np.linspace(-10e-3, 140e-3, M)

R0 = np.zeros((M, M))
R1 = np.zeros((M, M))
R2 = np.zeros((M, M))
R3 = np.zeros((M, M))

f1 = open('data/regions/norm_R_L1e2_appended.txt', 'a')
f2 = open('data/regions/norm_R_L5e2_appended.txt', 'a')
f3 = open('data/regions/norm_R_L10e2_appended.txt', 'a')
for i in range(235, M):
    print(i)
    for j in range(M):
        R0[i, j] = region0(Vx[j], Ed[i], m)
        R1[i, j] = region(Vx[j], Ed[i], m, L, res=res)
        R2[i, j] = region(Vx[j], Ed[i], m, 5*L, res=res)
        R3[i, j] = region(Vx[j], Ed[i], m, 10*L, res=res)
        f1.write(str(R1[i, j]) + '\n')
        f2.write(str(R2[i, j]) + '\n')
        f3.write(str(R3[i, j]) + '\n')

f1.close()
f2.close()
f3.close()

# setting the latex style
plt.rc('font', family='serif')
plt.rc('text', usetex=True)

plt.figure(figsize=(6.5, 6.5))
plt.plot(1e3*Vx, 1e3*Vx, '--', color='grey')
plt.contour(1e3*Vx, 1e3*Ed, R0, levels=[0], colors='C0')
plt.contour(1e3*Vx, 1e3*Ed, R1, levels=[0], colors='C1')
plt.contour(1e3*Vx, 1e3*Ed, R2, levels=[0], colors='C2')
plt.contour(1e3*Vx, 1e3*Ed, R3, levels=[0], colors='C3')
plt.axis('scaled')
plt.tick_params(labelsize=16)

plt.show()
