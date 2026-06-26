import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import (
    CARTELLA_RISULTATI, CARTELLA_DATI, ALPHA_ELASTIC_NET, SEME_CASUALE
)

print("-- STEP 8: validazione esterna GSE33440 --")

# Carica pannello geni
file_pannello = os.path.join(CARTELLA_RISULTATI, "geni_pannello_descoperta.txt")
geni_pannello = pd.read_csv(file_pannello, header=None)[0].tolist()
print(f"Pannello geni: {len(geni_pannello)} geni")

# Carica dati validazione (batch-corrected se disponibili)
file_val_corretto = os.path.join(CARTELLA_RISULTATI, "validazione_batch_corretto.csv")
if os.path.exists(file_val_corretto):
    expr_val = pd.read_csv(file_val_corretto, index_col=0)
    print(f"Usando dati batch-corrected: {expr_val.shape}")
else:
    from config import FILE_VALIDAZIONE_PULITO
    expr_val = pd.read_csv(FILE_VALIDAZIONE_PULITO, index_col=0)
    print(f"Usando dati originali: {expr_val.shape}")

meta_val = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "metadata_GSE33440.csv"))
val_t1d = meta_val[meta_val["Group"] == "T1D"]["GSM_ID"].tolist()
val_ctrl = meta_val[meta_val["Group"] == "Control"]["GSM_ID"].tolist()
val_campioni = val_t1d + val_ctrl

# Geni del pannello presenti in validazione
geni_in_val = [g for g in geni_pannello if g in expr_val.index]
geni_mancanti_val = set(geni_pannello) - set(geni_in_val)
print(f"Geni pannello presenti in validazione: {len(geni_in_val)}/{len(geni_pannello)}")
if geni_mancanti_val:
    print(f"Geni mancanti: {geni_mancanti_val}")

if len(geni_in_val) < 2:
    print("ERRORE: Troppi pochi geni per fare predizioni.")
    exit(1)

# Prepara dati
X_val = expr_val.loc[geni_in_val, val_campioni].T.values
y_val = np.array([1] * len(val_t1d) + [0] * len(val_ctrl))
print(f"\nValidazione: {X_val.shape[0]} campioni, {X_val.shape[1]} feature")
print(f"  T1D: {np.sum(y_val)}, Control: {np.sum(y_val == 0)}")

# Carica parametri modello
parametri = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "parametri_modello_finale.csv"), header=None, index_col=0)
miglior_C = float(parametri.loc["miglior_C", 1])

# Standardizza usando i parametri del training
file_train_corretto = os.path.join(CARTELLA_RISULTATI, "training_batch_corretto.csv")
expr_train = pd.read_csv(file_train_corretto, index_col=0)

meta_train = pd.read_csv(os.path.join(CARTELLA_DATI, "target_metadata_GPL96.csv"))
train_t1d = meta_train[meta_train["Group"] == "T1D"]["GSM_ID"].tolist()
train_ctrl = meta_train[meta_train["Group"] == "Control"]["GSM_ID"].tolist()
train_campioni = train_t1d + train_ctrl

X_train_full = expr_train.loc[geni_in_val, train_campioni].T.values
y_train_full = np.array([1] * len(train_t1d) + [0] * len(train_ctrl))

# Standardizza su training, trasforma validation
scaler_full = StandardScaler()
X_train_scaled = scaler_full.fit_transform(X_train_full)
X_val_scaled = scaler_full.transform(X_val)

# Ri-allena modello su training completo e testa su validation
modello = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    C=miglior_C,
    l1_ratio=ALPHA_ELASTIC_NET,
    max_iter=10000,
    random_state=SEME_CASUALE
)
modello.fit(X_train_scaled, y_train_full)
predizioni = modello.predict_proba(X_val_scaled)[:, 1]

# Calcola AUC
auc_val = roc_auc_score(y_val, predizioni)
print(f"\n{'=' * 50}")
print(f"AUC su validazione esterna (GSE33440): {auc_val:.4f}")
print(f"{'=' * 50}")

# Curva ROC
fpr_val, tpr_val, _ = roc_curve(y_val, predizioni)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr_val, tpr_val, label=f"Elastic Net (AUC = {auc_val:.4f})", linewidth=2, color="#FF5722")
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Casuale (AUC = 0.5)")
ax.set_xlabel("Tasso Falsi Positivi")
ax.set_ylabel("Tasso Veri Positivi")
ax.set_title("Curva ROC - Validazione Esterna GSE33440")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CARTELLA_RISULTATI, "curva_roc_validazione_esterna.png"), dpi=150)
print(f"Curva ROC salvata: curva_roc_validazione_esterna.png")

# Salva predizioni
pred_df = pd.DataFrame({
    "Campione": val_campioni,
    "Etichetta_Reale": y_val,
    "Probabilita_Predetta": predizioni
})
pred_df.to_csv(os.path.join(CARTELLA_RISULTATI, "predizioni_validazione.csv"), index=False)

print(f"\nSTEP 8 COMPLETATO.")
print(f"AUC validazione esterna: {auc_val:.4f}")
