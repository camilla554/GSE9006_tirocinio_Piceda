import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, brier_score_loss, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import (
    CARTELLA_RISULTATI, BOOTSTRAP_AUC, PREVALENZA_SCREENING, SEME_CASUALE
)

print(">> STEP 9: metriche cliniche")

# Carica predizioni
pred_df = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "predizioni_validazione.csv"))
y_val = pred_df["Etichetta_Reale"].values
y_pred = pred_df["Probabilita_Predetta"].values

# Leggi AUC salvata
parametri_val = pd.read_csv(
    os.path.join(CARTELLA_RISULTATI, "parametri_modello_finale.csv"),
    header=None, index_col=0
)
# AUC dalla validazione esterna viene ricalcolata

auc_val = roc_auc_score(y_val, y_pred)
print(f"AUC validazione: {auc_val:.4f}")

# Brier Score (quanto sono calibrate le probabilità)
brier = brier_score_loss(y_val, y_pred)
print(f"Brier Score: {brier:.4f}")  # 0 = perfetto, 1 = pessimo

# Youden Index: trova la soglia che massimizza (sensibilità + specificità - 1)
fpr, tpr, soglie = roc_curve(y_val, y_pred)
indice_youden = np.argmax(tpr - fpr)
soglia_ottimale = soglie[indice_youden]
sensibilita = tpr[indice_youden]
specificita = 1 - fpr[indice_youden]
youden = sensibilita + specificita - 1

print(f"\n--- INDICE DI YOUDEN ---")
print(f"Soglia ottimale: {soglia_ottimale:.4f}")
print(f"Sensibilità: {sensibilita:.4f}")
print(f"Specificità: {specificita:.4f}")
print(f"Youden Index: {youden:.4f}")

# Matrice di confusione
pred_class = (y_pred >= soglia_ottimale).astype(int)
tn = np.sum((y_val == 0) & (pred_class == 0))
fp = np.sum((y_val == 0) & (pred_class == 1))
fn = np.sum((y_val == 1) & (pred_class == 0))
tp = np.sum((y_val == 1) & (pred_class == 1))

print(f"\n--- MATRICE DI CONFUSIONE (soglia = {soglia_ottimale:.4f}) ---")
print(f"                  Predetto Neg   Predetto Pos")
print(f"Reale Negativo     {tn}               {fp}")
print(f"Reale Positivo     {fn}               {tp}")

accuratezza = (tp + tn) / (tp + tn + fp + fn)
acc_bilanciata = (sensibilita + specificita) / 2
print(f"\nAccuratezza: {accuratezza:.4f}")
print(f"Accuratezza bilanciata: {acc_bilanciata:.4f}")

# PPV e NPV a prevalenza di screening (11%)
print(f"\n--- PPV / NPV (prevalenza screening = {PREVALENZA_SCREENING:.0%}) ---")
prev = PREVALENZA_SCREENING
se = sensibilita
sp = specificita

ppv = (se * prev) / (se * prev + (1 - sp) * (1 - prev))
npv = (sp * (1 - prev)) / ((1 - se) * prev + sp * (1 - prev))
print(f"PPV @ {prev:.0%}: {ppv:.4f}  (probabilità che un positivo sia davvero malato)")
print(f"NPV @ {prev:.0%}: {npv:.4f}  (probabilità che un negativo sia davvero sano)")

# Bootstrap per intervallo di confidenza AUC
print(f"\n--- BOOTSTrap AUC 95% CI ---")
rng = np.random.RandomState(SEME_CASUALE)
auc_boot = []
for i in range(BOOTSTRAP_AUC):
    idx = rng.choice(len(y_val), size=len(y_val), replace=True)
    if len(np.unique(y_val[idx])) < 2:
        continue
    try:
        auc_boot.append(roc_auc_score(y_val[idx], y_pred[idx]))
    except:
        continue

auc_boot = np.array(auc_boot)
ci_basso = np.percentile(auc_boot, 2.5)
ci_alto = np.percentile(auc_boot, 97.5)
print(f"AUC 95% CI: {ci_basso:.4f} - {ci_alto:.4f}")

# Calibration plot
print(f"\n--- CALIBRATION PLOT ---")
fig, ax = plt.subplots(figsize=(8, 6))
n_bins = 10
bordi = np.linspace(0, 1, n_bins + 1)
centri = (bordi[:-1] + bordi[1:]) / 2

tassi_osservati = []
for i in range(n_bins):
    mask = (y_pred >= bordi[i]) & (y_pred < bordi[i + 1])
    tassi_osservati.append(np.mean(y_val[mask]) if np.sum(mask) > 0 else 0)

ax.plot(centri, tassi_osservati, "o-", label="Osservato", color="#FF5722", linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Calibrazione perfetta")
ax.set_xlabel("Probabilità Predetta Media")
ax.set_ylabel("Frequenza Osservata")
ax.set_title(f"Calibrazione - Validazione Esterna\nBrier Score = {brier:.4f}")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(CARTELLA_RISULTATI, "calibrazione_validazione.png"), dpi=150)
print(f"Calibration plot salvato: calibrazione_validazione.png")

# Salva tutte le metriche
metriche = {
    "auc_validazione": float(auc_val),
    "auc_95_ci_basso": float(ci_basso),
    "auc_95_ci_alto": float(ci_alto),
    "brier_score": float(brier),
    "soglia_youden": float(soglia_ottimale),
    "sensibilita": float(sensibilita),
    "specificita": float(specificita),
    "indice_youden": float(youden),
    "accuratezza": float(accuratezza),
    "acc_bilanciata": float(acc_bilanciata),
    "ppv_11pct": float(ppv),
    "npv_11pct": float(npv),
    "tn": int(tn),
    "fp": int(fp),
    "fn": int(fn),
    "tp": int(tp)
}
pd.Series(metriche).to_csv(os.path.join(CARTELLA_RISULTATI, "metriche_cliniche.csv"))

print("Metriche calcolate.")
