#%%

#Code final qui estime p et alpha simultanément à partir de l'autocorrélation du signal I
from scipy.signal import correlate
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import math

# --- Paramètres du modèle balistique ---
alpha = 0.1
p = 10.0
N = 1200
xf = 900
d = 20
M = xf / d
t_N = N / p
tf = xf / p

# --- Simulation temporelle ---
T_end = 10000
dt = 0.01 # résolution plus fine
temps = np.arange(0, T_end, dt)

# --- Simulation du modèle balistique ---
entrees = []
t_actuel = 0
while t_actuel < T_end:
    inter_arrivee = np.random.exponential(1 / alpha)
    t_actuel += inter_arrivee
    if t_actuel < T_end:
        entrees.append(t_actuel)
entrees = np.array(entrees)

def generer_signal_I_vectorise(entrees, temps, N, p, xf, d):
    I = np.zeros_like(temps)
    t_N = N / p
    tf = xf / p
    M = xf / d
    for t_entree in entrees:
        start_idx = int(np.ceil(t_entree / dt))
        end_idx = int(np.floor((t_entree + t_N) / dt))
        if end_idx >= len(temps):
            end_idx = len(temps) - 1
        tau = temps[start_idx:end_idx] - t_entree
        tf_mask = tau < tf
        I[start_idx:end_idx][tf_mask] += M * tau[tf_mask] / tf
        I[start_idx:end_idx][~tf_mask] += M
    return I

I = generer_signal_I_vectorise(entrees, temps, N, p, xf, d)

# --- Autocorrélation numérique ---
def autocorr_numerique(signal, dt, Tmax=500):
    signal = signal - np.mean(signal)
    corr = correlate(signal, signal, mode='full')
    mid = len(corr) // 2
    corr = corr[mid:mid + int(Tmax / dt)]
    taus = np.arange(len(corr)) * dt
    return taus, corr / len(signal)

# --- Autocorrélation analytique non normalisée ---
def autocorr_analytique_non_norm(alpha, p, N, xf, d, M, dt, Tmax=300):
    tN = N / p
    tf = xf / p
    tf_tilde = tf / tN
    K = (alpha * N * M**2) / (6 * p * tf_tilde**2)
    taus = np.arange(0, Tmax, dt)
    tau_tilde = taus / tN
    C_tau = np.zeros_like(taus)
    mask1 = tau_tilde <= 1 - tf_tilde
    t = tau_tilde[mask1]
    C_tau[mask1] = K * (t**3 - 3 * tf_tilde * t**2 - 3 * tf_tilde**2 * t + (6 - 4 * tf_tilde) * tf_tilde**2)
    mask2 = (tau_tilde > 1 - tf_tilde) & (tau_tilde <= tf_tilde)
    t = tau_tilde[mask2]
    C_tau[mask2] = K * (t**3 + 3 * tf_tilde * (tf_tilde - 2) * t + (3 - tf_tilde**2) * tf_tilde)
    mask3 = (tau_tilde > tf_tilde) & (tau_tilde <= 1)
    t = tau_tilde[mask3]
    C_tau[mask3] = 3 * K * tf_tilde * (1 - t)**2
    return taus, C_tau

# --- Fonction pour curve_fit : dépend de alpha et p ---
def C_theorique_non_normalise(tau, alpha_val, p_val):
    _, C_th = autocorr_analytique_non_norm(alpha_val, p_val, N, xf, d, M, dt, Tmax=len(tau) * dt)
    return C_th[:len(tau)]

# --- Détection précise de tau_max ---
def detecter_tau_max_precis(taus, C_sim_norm, seuil=0.05):
    for i in range(1, len(C_sim_norm)):
        if C_sim_norm[i - 1] >= seuil > C_sim_norm[i]:
            # Interpolation linéaire
            t1, t2 = taus[i - 1], taus[i]
            c1, c2 = C_sim_norm[i - 1], C_sim_norm[i]
            slope = (c2 - c1) / (t2 - t1)
            tau_interp = t1 + (seuil - c1) / slope
            return tau_interp
    return taus[-1]

# --- Analyse de l'autocorrélation ---
taus_sim, C_sim = autocorr_numerique(I, dt)
C_sim_norm = C_sim / C_sim[0]
var_I = np.var(I)
C_sim_non_norm = C_sim_norm * var_I  # mise à l'échelle réelle

# Limite temporelle du fit
tau_max = detecter_tau_max_precis(taus_sim, C_sim_norm, seuil=0.05)
masque_fit = taus_sim <= tau_max
taus_fit = taus_sim[masque_fit]
C_fit = C_sim_non_norm[masque_fit]

# --- Estimation conjointe ---
p0 = [0.5, 12.0]  # valeurs initiales [alpha, p]
bounds = ([0.001, 1.0], [1.0, 50.0])  # bornes réalistes
popt2, pcov2 = curve_fit(C_theorique_non_normalise, taus_fit, C_fit, p0=p0, bounds=bounds, maxfev=10000)
alpha_joint, p_joint = popt2

# --- Résultats ---
print(f"\n--- Estimation conjointe alpha & p ---")
print(f"  α estimé : {alpha_joint:.5f} s⁻¹ (réel = {alpha})")
print(f"  p estimé : {p_joint:.5f} codons/s (réel = {p})")
print(f"  τ_max (interpolé) : {tau_max:.6f} s")

# --- Affichage ---
taus_th, C_th_joint = autocorr_analytique_non_norm(alpha_joint, p_joint, N, xf, d, M, dt, Tmax=len(taus_sim)*dt)
 
plt.figure(figsize=(12, 6))
plt.plot(taus_sim, C_sim_non_norm, label="Simulation (non normalisée)", color='blue')
plt.plot(taus_th[:len(taus_sim)], C_th_joint[:len(taus_sim)], '--', label="Fit théorique", color='red')
plt.axvline(tau_max, color='gray', linestyle='--', label=f"τ_max ≈ {tau_max:.3f} s")
plt.xlabel("Décalage temporel τ (s)")
plt.ylabel("C(τ)")
plt.title("Estimation conjointe de α et p par ajustement de l'autocorrélation")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%%

# Code qui estime C et D simultanément à partir de l'autocorrélation du signal I
# BRUIT DE FOND Version Numba 

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
L = 100
sub_volume_start = 45
sub_volume_end = 55
sub_volume_length = sub_volume_end - sub_volume_start
C = N / L

# --------------------------- Initialisation --------------------------- #
positions = np.random.uniform(0, L, N)
initial_positions = positions.copy()

# --------------------------- Fonctions Numba --------------------------- #
@njit(fastmath=True)
def simulate(positions, initial_positions, steps, sqrt_2Ddt, L, sub_volume_start, sub_volume_end, sub_volume_length):
    N = positions.size
    concentration = np.empty(steps)
    msd = np.empty(steps)
    for i in range(steps):
        dx = np.random.normal(0, sqrt_2Ddt, N)
        positions = (positions + dx) % L
        in_sub_volume = (positions >= sub_volume_start) & (positions < sub_volume_end)
        concentration[i] = np.sum(in_sub_volume) / sub_volume_length
        displacement = (positions - initial_positions + L/2) % L - L/2
        msd[i] = np.mean(displacement**2)
    return concentration, msd

# --------------------------- Simulation rapide --------------------------- #
sqrt_2Ddt = np.sqrt(2 * D * dt)

concentration, msd = simulate(positions, initial_positions, steps, sqrt_2Ddt, L,
                               sub_volume_start, sub_volume_end, sub_volume_length)

# --------------------------- Tracé du MSD --------------------------- #
time = np.linspace(0, T, steps)
theoretical_msd = 2 * D * time
msd_max = L**2 / 12


plt.figure()
plt.plot(time, msd, label="MSD simulé")
plt.plot(time, theoretical_msd, label="MSD théorique (2Dt)")
plt.axhline(msd_max, color='red', linestyle='--', label=f"MSD max théorique")
plt.xlabel("Temps")
plt.ylabel("MSD")
plt.title("MSD simulé vs théorique avec saturation")
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
plt.title("Histogramme de I(t) (intensité dans le sous-volume)")
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

# --------------------------- Préparation du fit modes propres --------------------------- #
nmax = 50
n = np.arange(1, nmax + 1)
k = n * np.pi / L
Ival = np.sqrt(2.0 / L) / k * (np.sin(k * sub_volume_end) - np.sin(k * sub_volume_start))

def C_theo_1D_vectorized(tau, D, c0):
    lam = (n * np.pi / L) ** 2
    expo = np.exp(-np.outer(lam, tau) * D)
    C_sum = np.dot(Ival ** 2, expo)
    return c0 * C_sum


# --------------------------- Estimation initiale pour D_MSD --------------------------- #
t_sat = L**2 / ((2 * np.pi)**2 * D)  # ≈ L² / (39.48 D)
fit_range = (time < t_sat)


def linear_msd(t, D): return 2 * D * t
popt_msd, _ = curve_fit(linear_msd, time[fit_range], msd[fit_range])
D_init = popt_msd[0]

C_init = I_t.mean() / sub_volume_length
p0 = [D_init, C_init]

# --------------------------- Fit modes propres --------------------------- #
mask_fit = (tau > 0.1) & (tau < 200)  # ajuster selon le bruit
tau_fit = tau[mask_fit]
C_fitdata = C_emp[mask_fit]

bounds = ([0.0, 0.0], [np.inf, np.inf])
popt, pcov = curve_fit(C_theo_1D_vectorized, tau_fit, C_fitdata, p0=p0, bounds=bounds)
D_modes_est, c0_modes_est = popt

# --------------------------- Tracé de la corrélation --------------------------- #
C_th = C_theo_1D_vectorized(tau_fit, D_modes_est, c0_modes_est)
C_th_true = C_theo_1D_vectorized(tau, D, C)

plt.figure()
plt.plot(tau_fit, C_fitdata, '-', ms=2, label="Corrélation empirique")
plt.plot(tau_fit, C_th, '--', label=f"Fit modes propres (D={D_modes_est:.3f}, c₀={c0_modes_est:.1f})")
plt.xlabel("τ (s)")
plt.ylabel("C_I(τ)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.title("Autocorrélation avec fit par modes propres")
plt.show()
 
# --------------------------- Résultats --------------------------- #
print("\n--- Estimations ---")
print(f"• D vrai       : {D:.1f}")
print(f"• C vraie      : {C:.1f}")
print(f"• D estimé (modes propres) : {D_modes_est:.5f}")
print(f"• C estimé (modes propres) : {c0_modes_est:.5f}")

# Histogramme des erreurs entre corrélation empirique et thoérique
