import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import (
    CARTELLA_RISULTATI, FILE_TRAINING, FILE_META_TRAINING,
    ALPHA_ELASTIC_NET, SEME_CASUALE, NUM_FOLD_OUTER
)

print("=== STEP 7: Elastic Net finale ===")

# Carica pannello geni
file_pannello = os.path.join(CARTELLA_RISULTATI, "geni_pannello_descoperta.txt")
if not os.path.exists(file_pannello):
    print(f"ERRORE: File pannello geni non trovato. Esegui prima Step 5.")
    exit(1)

geni_pannello = pd.read_csv(file_pannello, header=None)[0].tolist()
print(f"Pannello geni caricato: {len(geni_pannello)} geni")
print(f"Geni: {geni_pannello}")

# Carica dati training (corretti per batch se disponibili)
file_train_corretto = os.path.join(CARTELLA_RISULTATI, "training_batch_corretto.csv")
if os.path.exists(file_train_corretto):
    expr_train = pd.read_csv(file_train_corretto, index_col=0)
    print(f"Usando dati batch-corrected: {expr_train.shape}")
else:
    expr_train = pd.read_csv(FILE_TRAINING, index_col=0)
    print(f"Usando dati originali: {expr_train.shape}")

meta_train = pd.read_csv(FILE_META_TRAINING)
campioni_t1d = meta_train[meta_train["Group"] == "T1D"]["GSM_ID"].tolist()
campioni_ctrl = meta_train[meta_train["Group"] == "Control"]["GSM_ID"].tolist()
tutti_campioni = campioni_t1d + campioni_ctrl

# Geni del pannello presenti nella matrice
geni_disponibili = [g for g in geni_pannello if g in expr_train.index]
geni_mancanti = set(geni_pannello) - set(geni_disponibili)
if geni_mancanti:
    print(f"ATTENZIONE: {len(geni_mancanti)} geni del pannello non trovati: {geni_mancanti}")

X = expr_train.loc[geni_disponibili, tutti_campioni].T.values
y = np.array([1] * len(campioni_t1d) + [0] * len(campioni_ctrl))

print(f"\nDati: {X.shape[0]} campioni, {X.shape[1]} feature (geni del pannello)")

# Standardizza
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Tuning del parametro C (regolarizzazione inversa)
print(f"\nTuning parametro di regolarizzazione C...")
cv = StratifiedKFold(n_splits=NUM_FOLD_OUTER, shuffle=True, random_state=SEME_CASUALE)
valori_C = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

miglior_C = None
miglior_score = -1

for C in valori_C:
    modello = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        C=C,
        l1_ratio=ALPHA_ELASTIC_NET,
        max_iter=10000,
        random_state=SEME_CASUALE
    )
    punteggi = cross_val_score(modello, X_scaled, y, cv=cv, scoring="roc_auc")
    media = punteggi.mean()
    print(f"  C={C:.4f}: AUC = {media:.4f} (+/- {punteggi.std():.4f})")
    if media > miglior_score:
        miglior_score = media
        miglior_C = C

print(f"\nMiglior C: {miglior_C} (AUC = {miglior_score:.4f})")

# Allena modello finale con il miglior C
modello_finale = LogisticRegression(
    penalty="elasticnet",
    solver="saga",
    C=miglior_C,
    l1_ratio=ALPHA_ELASTIC_NET,
    max_iter=10000,
    random_state=SEME_CASUALE
)
modello_finale.fit(X_scaled, y)

# Coefficienti del modello
coeff_df = pd.DataFrame({
    "Gene": geni_disponibili,
    "Coefficiente": modello_finale.coef_[0]
}).sort_values("Coefficiente", ascending=False)
coeff_df.to_csv(os.path.join(CARTELLA_RISULTATI, "coefficienti_pannello.csv"), index=False)

print(f"\nCoefficienti del modello:")
for _, riga in coeff_df.iterrows():
    direzione = "Più espresso in T1D" if riga["Coefficiente"] > 0 else "Meno espresso in T1D"
    print(f"  {riga['Gene']:20s}  coeff = {riga['Coefficiente']:+.4f}  ({direzione})")

# AUC training e ROC curve
pred_proba = modello_finale.predict_proba(X_scaled)[:, 1]
auc_train = roc_auc_score(y, pred_proba)
print(f"\nAUC Training: {auc_train:.4f}")

fpr, tpr, _ = roc_curve(y, pred_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, label=f"Elastic Net (AUC = {auc_train:.4f})", linewidth=2)
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Casuale (AUC = 0.5)")
ax.set_xlabel("Tasso Falsi Positivi (1 - Specificità)")
ax.set_ylabel("Tasso Veri Positivi (Sensibilità)")
ax.set_title("Curva ROC - Training (Elastic Net su geni pannello)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(CARTELLA_RISULTATI, "curva_roc_training.png"), dpi=150)
print(f"Curva ROC salvata: curva_roc_training.png")

# Salva parametri modello
pd.Series({
    "miglior_C": miglior_C,
    "auc_train": auc_train,
    "geni_pannello": ";".join(geni_disponibili),
    "coefficienti": ";".join([f"{r['Gene']}:{r['Coefficiente']:.4f}" for _, r in coeff_df.iterrows()])
}).to_csv(os.path.join(CARTELLA_RISULTATI, "parametri_modello_finale.csv"))

print(f"\nSTEP 7 COMPLETATO.")
print(f"Modello Elastic Net con {len(geni_disponibili)} geni, AUC training: {auc_train:.4f}")
