import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from boruta import BorutaPy
import os
import json

from config import (
    FILE_TRAINING, FILE_META_TRAINING, CARTELLA_RISULTATI,
    BOOTSTRAP_ELASTIC_NET, ALPHA_ELASTIC_NET,
    BORUTA_MAX_ITER, SEME_CASUALE
)

print("--- STEP 4: Boruta + Elastic Net ---")

# Carica dati
matrice_geni = pd.read_csv(FILE_TRAINING, index_col=0)
metadati = pd.read_csv(FILE_META_TRAINING)

campioni_t1d = metadati[metadati["Group"] == "T1D"]["GSM_ID"].tolist()
campioni_ctrl = metadati[metadati["Group"] == "Control"]["GSM_ID"].tolist()
tutti_campioni = campioni_t1d + campioni_ctrl

X = matrice_geni[tutti_campioni].T.values
y = np.array([1] * len(campioni_t1d) + [0] * len(campioni_ctrl))
nomi_geni = matrice_geni.index.tolist()

print(f"Dati: {X.shape[0]} campioni, {X.shape[1]} geni")
print(f"Classi: {np.sum(y)} T1D, {np.sum(y == 0)} Control")

# Standardizzazione
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================
# METODO 1: BORUTA
# ============================================================
print(f"\n{'=' * 50}")
print("METODO 1: BORUTA")
print(f"{'=' * 50}")
print(f"Avvio Boruta con {BORUTA_MAX_ITER} iterazioni...")

rf_boruta = RandomForestClassifier(
    n_estimators=500,
    max_depth=min(10, X.shape[1]),
    random_state=SEME_CASUALE,
    n_jobs=-1
)

boruta = BorutaPy(
    rf_boruta,
    n_estimators="auto",
    max_iter=BORUTA_MAX_ITER,
    random_state=SEME_CASUALE,
    verbose=0
)
boruta.fit(X_scaled, y)

geni_confermati = [nomi_geni[i] for i in range(len(nomi_geni)) if boruta.support_[i]]
geni_tentativi = [nomi_geni[i] for i in range(len(nomi_geni)) if boruta.support_weak_[i]]
geni_rigettati = [nomi_geni[i] for i in range(len(nomi_geni))
                  if not boruta.support_[i] and not boruta.support_weak_[i]]

print(f"  Geni CONFERMATI: {len(geni_confermati)}")
print(f"  Geni TENTATIVI: {len(geni_tentativi)}")
print(f"  Geni RIGETTATI: {len(geni_rigettati)}")

# Salva risultati Boruta
classifica_boruta = pd.DataFrame({
    "Gene_Symbol": nomi_geni,
    "Boruta_Selezionato": boruta.support_.tolist(),
    "Boruta_Tentativo": boruta.support_weak_.tolist(),
    "Boruta_Rank": boruta.ranking_.tolist()
})
classifica_boruta.to_csv(os.path.join(CARTELLA_RISULTATI, "risultati_boruta.csv"), index=False)

# ============================================================
# METODO 2: ELASTIC NET BOOTSTRAP
# ============================================================
print(f"\n{'=' * 50}")
print("METODO 2: ELASTIC NET BOOTSTRAP")
print(f"{'=' * 50}")
print(f"Bootstrap con {BOOTSTRAP_ELASTIC_NET} iterazioni...")
print(f"DEBUG: X shape = {X.shape}")

probabilita_selezione = {gene: 0 for gene in nomi_geni}

for i in range(BOOTSTRAP_ELASTIC_NET):
    if (i + 1) % 100 == 0:
        print(f"  Iterazione {i + 1}/{BOOTSTRAP_ELASTIC_NET}")

    # Campionamento bootstrap (con reimmissione)
    indici_boot = np.random.choice(len(X), size=len(X), replace=True)
    X_boot = X_scaled[indici_boot]
    y_boot = y[indici_boot]

    # Elastic Net
    en = ElasticNet(
        alpha=0.1,
        l1_ratio=ALPHA_ELASTIC_NET,
        max_iter=10000,
        random_state=SEME_CASUALE + i,
        selection="random"
    )
    en.fit(X_boot, y_boot)

    # Conta i geni con coefficiente non nullo
    coefficienti_non_nulli = np.where(np.abs(en.coef_) > 1e-6)[0]
    for idx in coefficienti_non_nulli:
        probabilita_selezione[nomi_geni[idx]] += 1

# Converti conteggi in probabilità
for gene in probabilita_selezione:
    probabilita_selezione[gene] /= BOOTSTRAP_ELASTIC_NET

en_df = pd.DataFrame(
    list(probabilita_selezione.items()),
    columns=["Gene_Symbol", "Prob_EN"]
)
en_df = en_df.sort_values("Prob_EN", ascending=False)
en_df.to_csv(os.path.join(CARTELLA_RISULTATI, "risultati_elastic_net_bootstrap.csv"), index=False)

print(f"\nTop 15 geni per probabilità Elastic Net:")
print(en_df.head(15).to_string(index=False))

# Riepilogo
riepilogo = {
    "boruta_confermati": len(geni_confermati),
    "boruta_rigettati": len(geni_rigettati),
    "bootstrap_elastic_net": BOOTSTRAP_ELASTIC_NET,
    "top_geni_en": en_df.head(20)["Gene_Symbol"].tolist()
}
with open(os.path.join(CARTELLA_RISULTATI, "riepilogo_selezione_feature.json"), "w") as f:
    json.dump(riepilogo, f, indent=2)

print(f"\nSTEP 4 COMPLETATO.")
print(f"Boruta: {len(geni_confermati)} geni confermati")
print(f"Elastic Net: {(en_df['Prob_EN'] > 0.5).sum()} geni con prob > 0.5")
