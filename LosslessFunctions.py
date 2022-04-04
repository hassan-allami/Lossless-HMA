import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.integrate import quad
import functools

# constants
cte = (10 ** 6 / 1973 ** 2) ** (3 / 2)  # (2me/hbar^2)^3/2 in ev^(-3/2)A^(-3)
C_wp = np.sqrt(8e3 / (137 * 3 * np.pi))  # sqrt(8*alpha * sqrt(2mec^2) / (3pi))  [eV^1/4]
C_eps = 8e9 / (np.pi * 137 * 1973 ** 2)  # 8*alpha*(2mec)^3/2 / (pi*(hbar.c)^2)  [eV^-1/2A-2]

'''without interband transitions'''


# h*wp0 as a function of mu in eV
@functools.lru_cache(maxsize=128)  # this is memoizing
def wp0(mu, vx, ed, m):
    if mu < ed:
        return C_wp * m ** (1 / 4) * (vx ** 2 / (ed - mu) + mu) ** (3 / 4) * (
                (ed - mu) ** 2 / ((ed - mu) ** 2 + vx ** 2)) ** (3 / 2)
    else:
        return 0


# density
def density(mu, vx, ed, m):
    # to make sure it spits out 'nan' when mu = nan
    if np.isnan(mu):
        return float('nan')
    else:
        # define the integrand
        def int_n(k, vx, ed):
            return k ** 2 * (1 - (k ** 2 - ed) / np.sqrt((k ** 2 - ed) ** 2 + 4 * vx ** 2))

        em0 = (ed - np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2
        if em0 < mu:
            if mu < ed:
                kf = np.sqrt(vx ** 2 / (ed - mu) + mu)  # sqrt(Ekf)
            else:
                kf = np.inf
            result, err = quad(int_n, 0, kf, args=(vx, ed))
            return cte * m ** (3 / 2) * result / (2 * np.pi ** 2)
        else:
            return 0


# mu_max for wp0
def mu_max0(vx, ed):
    if vx > 0:
        a = 2 * vx ** 4 * (ed ** 2 + 4 * vx ** 2)
        expr = 2 * np.sign(ed) * np.sqrt(2) * ed * vx ** 2 / np.sqrt(a ** (1 / 3) - 2 * vx ** 2) - a ** (
                1 / 3) - 4 * vx ** 2
        return ed - (np.sign(ed) * np.sqrt(a ** (1 / 3) - 2 * vx ** 2) + np.sqrt(expr)) / np.sqrt(2)
    else:
        return ed


# lossless region without interband transitions
def region0(vx, ed, m):
    if ed > 0 and (C_wp * m ** (1 / 4) * vx ** (3 / 2) > ed ** (3 / 4) * (1 + vx ** 2 / (ed ** 2)) ** (3 / 2) * (
            ed + np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2):
        return 1
    elif ed <= vx and (wp0(mu_max0(vx, ed), vx, ed, m) > (ed + np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2):
        return 1
    else:
        return -1


# lossless region without interband transitions (different approach)
def region00(vx, ed, m):
    # The W function (the positive root function) that gives the maximum of wp0
    def w(r):
        c = [1, -12 * r, 6 * (45 + 8 * r ** 2), -8 * r * (27 + 8 * r ** 2), -27]  # poly coefficients
        root = np.roots(c)
        return root[(root.imag == 0) & (root.real > 0)].real[0]

    # the first condition
    if ed > 0 and C_wp * m ** (1 / 4) * vx ** (3 / 2) > ed ** (3 / 4) * (1 + vx ** 2 / (ed ** 2)) ** (3 / 2) * (
            ed + np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2:
        return 2
    # the condition
    elif ed < vx and vx > 0 and (C_wp * m ** (1 / 4) * vx ** (3 / 4) * (w(ed / vx)) ** (3 / 4) >
                                 np.sqrt(2) * (ed + np.sqrt(ed ** 2 + 4 * vx ** 2))):
        return 1
    else:
        return -1


# finding the mu range(s)
def mu_range0(vx, ed, m, res=100):
    em0 = (ed - np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2  # Em bottom
    dem = (ed + np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2  # Em width
    mu = np.linspace(em0, 0, res)
    dmu = -em0 / res
    # the region where wp0(0) > Delta Em
    if ed > 0 and C_wp * m ** (1 / 4) * vx ** (3 / 2) > ed ** (3 / 4) * (1 + vx ** 2 / (ed ** 2)) ** (3 / 2) * (
            ed + np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2:
        r = [u if dem < wp0(u, vx, ed, m) < dem - u else float('nan') for u in mu]
        return np.nanmin(r), np.nanmax(r), float('nan'), float('nan')
    # where wp_max > DeltaEm and mu_max < 0
    elif ed < vx and wp0(mu_max0(vx, ed), vx, ed, m) > dem:
        r = [u for u in mu if dem < wp0(u, vx, ed, m) < dem - u]
        if len(r) > 0:
            index = np.where(np.diff(r) > 2 * dmu)[0]
            # this checks if there is gap in the mu region
            if len(index) == 0:
                return np.nanmin(r), np.nanmax(r), float('nan'), float('nan')
            else:
                i = index[0]
                return np.nanmin(r), np.nanmax(r), r[i], r[i + 1]
        else:
            return float('nan'), float('nan'), float('nan'), float('nan')
    else:
        return float('nan'), float('nan'), float('nan'), float('nan')


# density region
def n_range0(vx, ed, m, res=100):
    # just calculates the density for the results of mu_range0
    muse = mu_range0(vx, ed, m, res)
    if muse[0] != float('nan'):
        n0 = density(muse[0], vx, ed, m)
    else:
        n0 = float('nan')
    if muse[1] != float('nan'):
        n1 = density(muse[1], vx, ed, m)
    else:
        n1 = float('nan')
    if muse[2] != float('nan'):
        n2 = density(muse[2], vx, ed, m)
    else:
        n2 = float('nan')
    if muse[3] != float('nan'):
        n3 = density(muse[3], vx, ed, m)
    else:
        n3 = float('nan')
    return n0, n1, n2, n3


# wp range in the window
def wp0_range(vx, ed, m, res=100):
    if region0(vx, ed, m) == -1:
        # print('nothing is in the window')
        return float('nan'), float('nan')
    else:
        res = 100  # determines the resolution
        em0 = (ed - np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2
        dem = (ed + np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2
        mu = np.linspace(em0, min(0, ed), res)
        w = [wp0(u, vx, ed, m) if dem < wp0(u, vx, ed, m) < (dem - u) else float('nan') for u in mu]
        if np.isnan(w).all():
            return dem, dem
        else:
            return np.nanmin(w), np.nanmax(w)


''' including interband transitions'''


# E_cross
@functools.lru_cache(maxsize=128)  # this is memoizing
def e_x(mu, vx, ed):
    a = vx ** 2 / (ed - mu) + mu - ed
    if ed < 0:
        return np.sqrt(ed ** 2 + 4 * vx ** 2)
    elif a < 0:
        return np.sqrt(a ** 2 + 4 * vx ** 2)
    else:
        return 2 * vx


# epsilon cross
def eps_x(mu, vx, ed, m, l, w):
    lm = 1973e-6 / m  # the length scale in terms of m
    l = l / lm  # normalized l
    # normalize to 2mc^2
    mu = mu / (m * 1e6)
    vx = vx / (m * 1e6)
    ed = ed / (m * 1e6)
    w = w / (m * 1e6)
    kf = np.sqrt(vx ** 2 / (ed - mu) + mu)

    # forming the integrand
    def integrand(k, vx, ed, w):
        return k ** 2 / np.sqrt((k**2 - ed)**2 + 4*vx**2) / ((k**2 - ed)**2 + 4*vx**2 - w**2)
    output = quad(integrand, 0, kf, args=(vx, ed, w), full_output=1)

    if len(output) == 3:
        return 1 + 8 / (137 * np.pi) * vx ** 2 * l ** 2 * output[0]
    else:
        # approx if it didn't converge
        # print('using approx')
        delta = 1e-4
        w = (1 - delta) * e_x(mu, vx, ed)
        approx = quad(integrand, 0, kf, args=(vx, ed, w), full_output=1)
        return 1 + 8 / (137 * np.pi) * vx ** 2 * l ** 2 * approx[0]


# solve for wp
def wp(mu, vx, ed, m, l):
    delta = 1e-15
    [wl, wr] = [0, e_x(mu, vx, ed) - delta]
    wm = (wl + wr) / 2
    n = 1
    while wr - wl > 1e-5 and n < 100:
        diff = wm ** 2 * eps_x(mu, vx, ed, m, l, wm) - wp0(mu, vx, ed, m) ** 2
        if diff > 0:
            wr = wm
            wm = (wr + wl) / 2
        else:
            wl = wm
            wm = (wl + wr) / 2
            n += 1
    return wm


# lossless region including interband transitions
def region(vx, ed, m, l, res=100):
    if region0(vx, ed, m) == -1:
        return -1
    else:
        eps = 1e-12
        em0 = (ed - np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2
        dem = (ed + np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2
        mu = np.linspace(em0 + eps, ed - eps, res)  # res determines the resolution of the search
        r = 2 * any(dem < wp(u, vx, ed, m, l) < (dem - u) for u in mu) - 1
        return r


# wp range in the window
def wp_range(vx, ed, m, l, n=100):
    if region0(vx, ed, m) == -1:
        print('nothing is in the window')
        return float('nan')
    else:
        delta = 1e-10
        em0 = (ed - np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2
        dem = (ed + np.sqrt(ed ** 2 + 4 * vx ** 2)) / 2
        mu = np.linspace(em0 + delta, ed - delta, n)  # n determines the resolution of the search
        w = [wp(u, vx, ed, m, l) if dem < wp(u, vx, ed, m, l) < (dem - u) else float('nan') for u in mu]
        return np.nanmin(w), np.nanmax(w)


''' test
y = np.linspace(0.27, 0.29, 50)
Ed0 = -0.27
Ed1 = 0.38
V0 = 2.8
V1 = 2.2
m0 = 0.117
m1 = 0.09
x0 = 1e-6
x1 = 1e-5
Ed = (1-y)*Ed0 + y*Ed1 + y*(1-y)*0.46
V = (1-y)*V0 + y*V1
mm = (1-y)*m0 + y*m1

plt.plot(y, Ed)
plt.plot(y, V*np.sqrt(x0))
plt.plot(y, V*np.sqrt(x1))

R0 = np.zeros(len(y))
R1 = np.zeros(len(y))
Wp0_min = np.zeros(len(y))
Wp0_max = np.zeros(len(y))
Wp1_min = np.zeros(len(y))
Wp1_max = np.zeros(len(y))
for i in range(len(y)):
    R0[i] = region0(V[i]*np.sqrt(x0), Ed[i], mm[i])
    R1[i] = region0(V[i]*np.sqrt(x1), Ed[i], mm[i])
    Wp0_min[i], Wp0_max[i] = wp0_range(V[i]*np.sqrt(x0), Ed[i], mm[i])
    Wp1_min[i], Wp1_max[i] = wp0_range(V[i] * np.sqrt(x1), Ed[i], mm[i])

plt.plot(y, 0.01*R0)
plt.plot(y, 0.01*R1)

plt.plot(y, Wp0_min)
plt.plot(y, Wp0_max)
plt.plot(y, Wp1_min)
plt.plot(y, Wp1_max)


plt.show()
'''
