import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import (
    CARTELLA_RISULTATI, CARTELLA_DATI, N_PERMUTAZIONI, ALPHA_ELASTIC_NET, SEME_CASUALE
)

print("== STEP 10: permutation test ==")
print(f"Numero permutazioni: {N_PERMUTAZIONI}")

# Carica dati
geni_pannello = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "geni_pannello_descoperta.txt"), header=None)[0].tolist()

train_expr = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "training_batch_corretto.csv"), index_col=0)
val_expr = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "validazione_batch_corretto.csv"), index_col=0)

meta_train = pd.read_csv(os.path.join(CARTELLA_DATI, "target_metadata_GPL96.csv"))
meta_val = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "metadata_GSE33440.csv"))

train_t1d = meta_train[meta_train["Group"] == "T1D"]["GSM_ID"].tolist()
train_ctrl = meta_train[meta_train["Group"] == "Control"]["GSM_ID"].tolist()
val_t1d = meta_val[meta_val["Group"] == "T1D"]["GSM_ID"].tolist()
val_ctrl = meta_val[meta_val["Group"] == "Control"]["GSM_ID"].tolist()

# Geni comuni
geni_comuni = [g for g in geni_pannello if g in train_expr.index and g in val_expr.index]
print(f"Geni del pannello utilizzabili: {len(geni_comuni)}/{len(geni_pannello)}")

# Prepara dati
X_train = train_expr.loc[geni_comuni, train_t1d + train_ctrl].T.values
y_train = np.array([1] * len(train_t1d) + [0] * len(train_ctrl))
X_val = val_expr.loc[geni_comuni, val_t1d + val_ctrl].T.values
y_val = np.array([1] * len(val_t1d) + [0] * len(val_ctrl))

# Standardizza
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# AUC osservata (reale)
parametri = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "parametri_modello_finale.csv"), header=None, index_col=0)
miglior_C = float(parametri.loc["miglior_C", 1])

modello_reale = LogisticRegression(
    penalty="elasticnet", solver="saga", C=miglior_C,
    l1_ratio=ALPHA_ELASTIC_NET, max_iter=10000, random_state=SEME_CASUALE
)
modello_reale.fit(X_train_scaled, y_train)
pred_reali = modello_reale.predict_proba(X_val_scaled)[:, 1]
auc_osservata = roc_auc_score(y_val, pred_reali)
print(f"AUC osservata: {auc_osservata:.4f}")

# Permutazioni: mescola le etichette del training
print(f"\nEsecuzione {N_PERMUTAZIONI} permutazioni...")
auc_permutate = []
rng = np.random.RandomState(SEME_CASUALE)

for i in range(N_PERMUTAZIONI):
    if (i + 1) % 100 == 0:
        print(f"  Permutazione {i + 1}/{N_PERMUTAZIONI}")

    y_perm = rng.permutation(y_train)

    modello_perm = LogisticRegression(
        penalty="elasticnet", solver="saga", C=miglior_C,
        l1_ratio=ALPHA_ELASTIC_NET, max_iter=10000, random_state=SEME_CASUALE + i
    )
    modello_perm.fit(X_train_scaled, y_perm)
    pred_perm = modello_perm.predict_proba(X_val_scaled)[:, 1]
    auc_permutate.append(roc_auc_score(y_val, pred_perm))

auc_permutate = np.array(auc_permutate)

# P-value: quante permutazioni hanno AUC >= AUC osservata?
n_estremi = np.sum(auc_permutate >= auc_osservata)
p_value = (1 + n_estremi) / (1 + N_PERMUTAZIONI)

print(f"\n{'=' * 50}")
print(f"RISULTATI TEST DI PERMUTAZIONE")
print(f"{'=' * 50}")
print(f"AUC osservata: {auc_osservata:.4f}")
print(f"AUC media permutazioni: {np.mean(auc_permutate):.4f}")
print(f"AUC std permutazioni: {np.std(auc_permutate):.4f}")
print(f"Permutazioni con AUC >= osservata: {n_estremi}/{N_PERMUTAZIONI}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print(f"\nRISULTATO: Il modello è statisticamente significativo (p < 0.05)")
else:
    print(f"\nRISULTATO: Il modello NON è statisticamente significativo (p >= 0.05)")

# Istogramma
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(auc_permutate, bins=30, alpha=0.7, color="#2196F3", edgecolor="white")
ax.axvline(auc_osservata, color="#FF5722", linewidth=2, linestyle="--",
           label=f"AUC osservata = {auc_osservata:.4f}")
ax.set_xlabel("AUC")
ax.set_ylabel("Frequenza")
ax.set_title(f"Test di Permutazione (B={N_PERMUTAZIONI})\np = {p_value:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CARTELLA_RISULTATI, "test_permutazione.png"), dpi=150)
print(f"Grafico salvato: test_permutazione.png")

pd.Series({
    "auc_osservata": auc_osservata,
    "media_auc_permutate": float(np.mean(auc_permutate)),
    "std_auc_permutate": float(np.std(auc_permutate)),
    "n_permutazioni": N_PERMUTAZIONI,
    "n_estremi": int(n_estremi),
    "p_value": float(p_value)
}).to_csv(os.path.join(CARTELLA_RISULTATI, "risultati_test_permutazione.csv"))

print("Permutation test completato.")
