import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from config import (
    CARTELLA_RISULTATI, FILE_TRAINING, FILE_META_TRAINING,
    FILE_VALIDAZIONE_PULITO
)

print("== STEP 6: batch correction + PCA ==")

# Carica training set (GSE9006)
print("Caricamento training set (GSE9006 GPL96)...")
expr_train = pd.read_csv(FILE_TRAINING, index_col=0)
meta_train = pd.read_csv(FILE_META_TRAINING)

# Carica validation set (GSE33440)
print("Caricamento validation set (GSE33440)...")
expr_val = pd.read_csv(FILE_VALIDAZIONE_PULITO, index_col=0)
meta_val = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "metadata_GSE33440.csv"))

# Geni in comune tra training e validation
geni_comuni = expr_train.index.intersection(expr_val.index)

train_comuni = expr_train.loc[geni_comuni]
val_comuni = expr_val.loc[geni_comuni]

# Rimuovi geni con NaN in validation set
geni_nan = val_comuni.index[val_comuni.isna().any(axis=1)]
if len(geni_nan) > 0:
    print(f"Rimossi {len(geni_nan)} geni con NaN nel validation set.")
    geni_comuni = geni_comuni.drop(geni_nan)
    train_comuni = expr_train.loc[geni_comuni]
    val_comuni = expr_val.loc[geni_comuni]

print(f"\nGeni condivisi (dopo pulizia NaN): {len(geni_comuni)}")

# Ordina campioni
t1d_samples = meta_train[meta_train["Group"] == "T1D"]["GSM_ID"].tolist()
ctrl_samples = meta_train[meta_train["Group"] == "Control"]["GSM_ID"].tolist()
train_ordinati = t1d_samples + ctrl_samples

val_t1d = meta_val[meta_val["Group"] == "T1D"]["GSM_ID"].tolist()
val_ctrl = meta_val[meta_val["Group"] == "Control"]["GSM_ID"].tolist()
val_ordinati = val_t1d + val_ctrl

# Allinea i dati
train_align = train_comuni[train_ordinati].T.values
val_align = val_comuni[val_ordinati].T.values

etichette_batch = np.array([0] * train_align.shape[0] + [1] * val_align.shape[0])

# Combina i due dataset
espress_combinata = np.vstack([train_align, val_align])

print(f"\nMatrice combinata: {espress_combinata.shape}")
print(f"  Training: {train_align.shape[0]} campioni")
print(f"  Validation: {val_align.shape[0]} campioni")

# Correzione batch (mean-centering per batch)
def correggi_batch(expr, batch):
    expr_corretta = expr.copy()
    batch_unici = np.unique(batch)
    media_totale = np.mean(expr, axis=0)

    effetti_batch = {}
    for b in batch_unici:
        mask = batch == b
        if np.sum(mask) > 1:
            media_batch = np.mean(expr[mask], axis=0)
            effetti_batch[b] = media_batch - media_totale

    for b in batch_unici:
        mask = batch == b
        if b in effetti_batch:
            expr_corretta[mask] = expr_corretta[mask] - effetti_batch[b]

    return expr_corretta + media_totale

print("\nApplicazione correzione batch...")
espress_corretta = correggi_batch(espress_combinata, etichette_batch)

n_train = train_align.shape[0]
train_corretto = espress_corretta[:n_train]
val_corretto = espress_corretta[n_train:]

# PCA prima e dopo correzione
print("\nGenerazione PCA plots...")

fig, assi = plt.subplots(1, 2, figsize=(14, 6))

# PCA PRIMA della correzione
scaler_prima = StandardScaler()
combinata_scaled_prima = scaler_prima.fit_transform(espress_combinata)
pca_prima = PCA(n_components=2)
pc_prima = pca_prima.fit_transform(combinata_scaled_prima)

colori = {0: "#2196F3", 1: "#FF5722"}
for b in [0, 1]:
    mask = etichette_batch == b
    label = "Training (GSE9006)" if b == 0 else "Validazione (GSE33440)"
    assi[0].scatter(pc_prima[mask, 0], pc_prima[mask, 1],
                c=colori[b], label=label, alpha=0.7, edgecolors="k", s=50)

assi[0].set_title(f"PCA PRIMA correzione batch\nPC1: {pca_prima.explained_variance_ratio_[0]:.1%}, PC2: {pca_prima.explained_variance_ratio_[1]:.1%}")
assi[0].set_xlabel("PC1")
assi[0].set_ylabel("PC2")
assi[0].legend()
assi[0].grid(True, alpha=0.3)

# PCA DOPO la correzione
scaler_dopo = StandardScaler()
combinata_scaled_dopo = scaler_dopo.fit_transform(espress_corretta)
pca_dopo = PCA(n_components=2)
pc_dopo = pca_dopo.fit_transform(combinata_scaled_dopo)

for b in [0, 1]:
    mask = etichette_batch == b
    label = "Training (GSE9006)" if b == 0 else "Validazione (GSE33440)"
    assi[1].scatter(pc_dopo[mask, 0], pc_dopo[mask, 1],
                c=colori[b], label=label, alpha=0.7, edgecolors="k", s=50)

assi[1].set_title(f"PCA DOPO correzione batch\nPC1: {pca_dopo.explained_variance_ratio_[0]:.1%}, PC2: {pca_dopo.explained_variance_ratio_[1]:.1%}")
assi[1].set_xlabel("PC1")
assi[1].set_ylabel("PC2")
assi[1].legend()
assi[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(CARTELLA_RISULTATI, "pca_correzione_batch.png"), dpi=150)
print(f"Grafico PCA salvato: pca_correzione_batch.png")

# Salva dati corretti
train_df = pd.DataFrame(train_corretto, index=train_ordinati, columns=geni_comuni).T
train_df.to_csv(os.path.join(CARTELLA_RISULTATI, "training_batch_corretto.csv"))

val_df = pd.DataFrame(val_corretto, index=val_ordinati, columns=geni_comuni).T
val_df.to_csv(os.path.join(CARTELLA_RISULTATI, "validazione_batch_corretto.csv"))

print(f"\nDati corretti salvati:")
print(f"  Training: {train_df.shape}")
print(f"  Validazione: {val_df.shape}")

print("Step 6 fatto. PCA plot salvato.")
