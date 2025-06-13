import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve
from numba import njit

# --------------------------- Paramètres --------------------------- #
N = 10000
D = 1.0
dt = 0.1
T = 5000
steps = int(T / dt)
L = 100.0

sub_volume_start = np.array([45.0, 45.0, 45.0])
sub_volume_end = np.array([55.0, 55.0, 55.0])
sub_volume_length = np.prod(sub_volume_end - sub_volume_start)
C = N / L**3

# --------------------------- Initialisation --------------------------- #
positions = np.random.uniform(0, L, (N, 3))
initial_positions = positions.copy()

# --------------------------- Fonctions Numba --------------------------- #
@njit(fastmath=True)
def simulate(positions, initial_positions, steps, sqrt_2Ddt, L, sub_volume_start, sub_volume_end, sub_volume_length):
    N = positions.shape[0]
    concentration = np.empty(steps)
    msd = np.empty(steps)

    for i in range(steps):
        displacements = np.random.normal(0, sqrt_2Ddt, (N, 3))
        positions = (positions + displacements) % L

        in_sub_volume = np.empty(N, dtype=np.bool_)
        for j in range(N):
            inside = True
            for d in range(3):
                if positions[j, d] < sub_volume_start[d] or positions[j, d] >= sub_volume_end[d]:
                    inside = False
                    break
            in_sub_volume[j] = inside

        concentration[i] = np.sum(in_sub_volume) / sub_volume_length

        displacement = (positions - initial_positions + L / 2) % L - L / 2
        msd[i] = np.mean(np.sum(displacement**2, axis=1))

    return concentration, msd

# --------------------------- Simulation rapide --------------------------- #
sqrt_2Ddt = np.sqrt(2 * D * dt)
concentration, msd = simulate(positions, initial_positions, steps, sqrt_2Ddt, L,
                               sub_volume_start, sub_volume_end, sub_volume_length)

# --------------------------- Tracé du MSD --------------------------- #
time = np.linspace(0, T, steps)
theoretical_msd = 6 * D * time
msd_max = L**2 / 4  # approximation en 3D sur tore

plt.figure()
plt.plot(time, msd, label="MSD simulé")
plt.plot(time, theoretical_msd, label="MSD théorique (6Dt)")
plt.axhline(msd_max, color='red', linestyle='--', label=f"MSD max théorique")
plt.xlabel("Temps")
plt.ylabel("MSD")
plt.title("MSD simulé vs théorique avec saturation (3D)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------- Histogramme --------------------------- #
I_t = concentration * sub_volume_length

plt.figure(figsize=(7, 5))
plt.hist(I_t, bins=50, density=False, alpha=0.6, edgecolor='black', label="Histogramme")
plt.axvline(I_t.mean(), color='red', linestyle='--', label=f"Moyenne = {I_t.mean():.2f}")
plt.xlabel("Intensité I(t)")
plt.ylabel("Fréquence")
plt.title("Histogramme de I(t) (intensité dans le sous-volume, 3D)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------- Autocorrélation rapide (FFT) --------------------------- #
def autocorr_fft(x):
    x = x - np.mean(x)
    result = fftconvolve(x, x[::-1], mode='full')
    result = result[result.size // 2:]
    norm = np.arange(len(x), 0, -1)
    return result / norm

C_emp = autocorr_fft(I_t)
tau = np.arange(len(C_emp)) * dt

# --------------------------- Fit modes propres (approximation 3D) --------------------------- #
nmax = 10
n = np.arange(1, nmax + 1)
pi = np.pi
k = n * pi / L

# Produit triple des Ivals
Ival_1D = np.sqrt(2 / L) / k * (np.sin(k * sub_volume_end[0]) - np.sin(k * sub_volume_start[0]))
Ival3D = np.array([Ival_1D[i] * Ival_1D[j] * Ival_1D[k_] for i in range(nmax) for j in range(nmax) for k_ in range(nmax)])
lam3D = np.array([k[i]**2 + k[j]**2 + k[k_]**2 for i in range(nmax) for j in range(nmax) for k_ in range(nmax)])


def C_theo_3D_vectorized(tau, D, c0):
    expo = np.exp(-np.outer(lam3D, tau) * D)
    C_sum = np.dot(Ival3D ** 2, expo)
    return c0 * C_sum

# Estimation initiale
fit_range = (time < L**2 / (6 * D))
def linear_msd(t, D): return 6 * D * t
popt_msd, _ = curve_fit(linear_msd, time[fit_range], msd[fit_range])
D_init = popt_msd[0]
C_init = I_t.mean() / sub_volume_length
p0 = [D_init, C_init]

# Fit de la corrélation
mask_fit = (tau > 0.1) & (tau < 200)
tau_fit = tau[mask_fit]
C_fitdata = C_emp[mask_fit]

bounds = ([0.0, 0.0], [np.inf, np.inf])
popt, _ = curve_fit(C_theo_3D_vectorized, tau_fit, C_fitdata, p0=p0, bounds=bounds)
D_modes_est, c0_modes_est = popt

# --------------------------- Tracé de la corrélation --------------------------- #
C_th = C_theo_3D_vectorized(tau_fit, D_modes_est, c0_modes_est)
C_th_true = C_theo_3D_vectorized(tau, D, C)

plt.figure()
plt.plot(tau_fit, C_fitdata, '-', label="Corrélation empirique")
plt.plot(tau_fit, C_th, '--', label=f"Fit modes propres (D={D_modes_est:.3f}, c₀={c0_modes_est:.1f})")
plt.xlabel("τ (s)")
plt.ylabel("C_I(τ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title("Autocorrélation avec fit par modes propres (3D)")
plt.show()

# --------------------------- Résultats --------------------------- #
print("\n--- Estimations (3D) ---")
print(f"• D vrai       : {D:.1f}")
print(f"• C vraie      : {C:.3f}")
print(f"• D estimé (modes propres) : {D_modes_est:.5f}")
print(f"• C estimé (modes propres) : {c0_modes_est:.5f}")
