import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.signal import fftconvolve

# --- Paramètres du modèle ---
alpha = 0.07           # taux d'initiation (1/s)
p = 35                 # vitesse d'élongation (codons/s)
L = 1292               # longueur totale de l'ARNm (codons)
d = 24                 # espacement entre codons fluorescents
M = L // d             # nombre de codons fluorescents
i0 = 1                 # intensité unitaire par codon fluorescent

T = L / p              # durée de transit d’un ribosome
mu = alpha * T         # espérance du nombre de ribosomes simultanés

# --- Distribution de l’intensité d’un ribosome ---
valeurs_irib = np.arange(0, M + 1) * i0
proba_irib = [d / L] * M + [(L - d * M) / L]

def echantillonner_Irib(N_samples):
    return np.random.choice(valeurs_irib, size=N_samples, p=proba_irib)

# --- Simulation Monte Carlo ---
n_runs = 100000
resultats_sim = []

for _ in range(n_runs):
    n_ribo = np.random.poisson(mu)
    intensites = echantillonner_Irib(n_ribo)
    I_tot = np.sum(intensites)
    resultats_sim.append(I_tot)

# Histogramme empirique
comptage_sim = Counter(resultats_sim)
x_sim = np.array(sorted(comptage_sim.keys()))
P_sim = np.array([comptage_sim[k] / n_runs for k in x_sim])

# --- Calcul de la distribution théorique ---
max_k = max(x_sim) + 1
P0 = np.zeros(max_k)
P0[0] = 1  # Dirac initial

P_irib = np.zeros(max_k)
for i, v in enumerate(valeurs_irib):
    if v < max_k:
        P_irib[v] = proba_irib[i]

P_theo = np.zeros(max_k)
for n in range(0, 40):  # somme jusqu’à 40 ribosomes
    conv = P0
    for _ in range(n):
        conv = fftconvolve(conv, P_irib)[:max_k]
    poids = np.exp(-mu) * mu**n / np.math.factorial(n)
    P_theo += poids * conv

# --- Affichage graphique : deux lignes lissées ---
plt.figure(figsize=(8, 6))
plt.plot(x_sim, P_sim, color='red', linewidth=1.2, label='Simulation Monte Carlo')
plt.plot(np.arange(len(P_theo)), P_theo, color='blue', linewidth=1.2, label='Loi théorique')
plt.xlabel(r"$I$", fontsize=14)
plt.ylabel(r"$P_T(I)$", fontsize=14)
plt.title("Distribution du signal total de fluorescence – modèle balistique", fontsize=12)
plt.xlim(0, 225)
plt.ylim(0, 0.09)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
