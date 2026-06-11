"""
PREPROCESSING DATASET GSE9006 (STEP 0-1-2)
Pipeline completa per l'identificazione di biomarcatori T1D

Output:
    - data/clean_gene_expression.csv   : matrice geni × campioni (13.515 geni, 105 campioni)
    - data/target_metadata_GPL96.csv   : metadati (81 T1D, 24 Control)
"""

import os
import GEOparse
import pandas as pd
import numpy as np

data_dir = "./data"

# ============================================================
# STEP 0: SCARICAMENTO DATASET GSE9006
# ============================================================
print("=" * 70)
print("STEP 0: SCARICAMENTO DATASET GSE9006")
print("=" * 70)

gse = GEOparse.get_GEO(geo="GSE9006", destdir=data_dir)

samples_info = []
for gsm_id, gsm in gse.gsms.items():
    characteristics = gsm.metadata.get("characteristics_ch1", [])
    title = gsm.metadata.get("title", [""])[0]
    source = gsm.metadata.get("source_name_ch1", [""])[0]
    samples_info.append({
        "GSM_ID": gsm_id,
        "Title": title,
        "Source": source,
        "Characteristics": " | ".join(characteristics)
    })

df_samples = pd.DataFrame(samples_info)
df_samples.to_csv(os.path.join(data_dir, "metadata_GSE9006.csv"), index=False)
print(f"Totale campioni scaricati: {len(df_samples)}")

# ============================================================
# STEP 1: FILTRAGGIO CAMPIONI GPL96 E CLASSIFICAZIONE
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: FILTRAGGIO CAMPIONI GPL96 E CLASSIFICAZIONE T1D/CONTROL")
print("=" * 70)

metadata_list = []
for gsm_id, gsm in gse.gsms.items():
    platform = gsm.metadata.get("platform_id", [""])[0]
    if platform != "GPL96":
        continue

    title = gsm.metadata.get("title", [""])[0]
    source = gsm.metadata.get("source_name_ch1", [""])[0]
    chars = " | ".join(gsm.metadata.get("characteristics_ch1", []))
    full_text = f"{title} {source} {chars}".lower()

    if "type 1" in full_text or "t1d" in full_text:
        group = "T1D"
    elif "healthy" in full_text or "control" in full_text or "donor" in full_text:
        group = "Control"
    else:
        continue

    metadata_list.append({
        "GSM_ID": gsm_id,
        "Title": title,
        "Group": group,
        "Platform": platform
    })

df_meta = pd.DataFrame(metadata_list)
df_target = df_meta[df_meta["Group"].isin(["T1D", "Control"])].copy()

print("\nCampioni GPL96 selezionati:")
print(df_target["Group"].value_counts())
df_target.to_csv(os.path.join(data_dir, "target_metadata_GPL96.csv"), index=False)

print("\nEstrazione matrice di espressione...")
valid_samples = df_target["GSM_ID"].tolist()
gpl96 = gse.gpls["GPL96"]

symbol_col = None
for col in gpl96.table.columns:
    if "symbol" in col.lower():
        symbol_col = col
        break
if not symbol_col:
    symbol_col = "Gene Symbol"

print(f"Colonna gene symbol: '{symbol_col}'")
probe_gene_map = gpl96.table.set_index("ID")[symbol_col].to_dict()

exp_dict = {}
for gsm_id in valid_samples:
    exp_dict[gsm_id] = gse.gsms[gsm_id].table.set_index("ID_REF")["VALUE"]

exp_df = pd.DataFrame(exp_dict)
exp_df["Gene_Symbol"] = exp_df.index.map(probe_gene_map)
exp_df = exp_df.dropna(subset=["Gene_Symbol"])
exp_df = exp_df[exp_df["Gene_Symbol"].astype(str).str.strip() != ""]

print(f"Matrice finale (sonde × campioni): {exp_df.shape}")
exp_df.to_csv(os.path.join(data_dir, "expression_matrix_GPL96.csv"))

# ============================================================
# STEP 2: PREPROCESSING E AGGREGAZIONE GENICA
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: PREPROCESSING E AGGREGAZIONE GENICA")
print("=" * 70)

exp_df = pd.read_csv(os.path.join(data_dir, "expression_matrix_GPL96.csv"), index_col=0)
meta_df = pd.read_csv(os.path.join(data_dir, "target_metadata_GPL96.csv"))

sample_cols = meta_df["GSM_ID"].tolist()
gene_symbols = exp_df["Gene_Symbol"]
numeric_exp = exp_df[sample_cols].copy()

# Controllo scala log2
max_val = numeric_exp.max().max()
min_val = numeric_exp.min().min()
print(f"Range: Min = {min_val:.2f}, Max = {max_val:.2f}")

if max_val > 100:
    print("Dati NON in scala log2. Applico log2(x + 1)...")
    numeric_exp = np.log2(numeric_exp + 1)
else:
    print("Dati GIÀ in scala log2 (RMA normalized).")

# Aggregazione sonde duplicate per Gene Symbol (media)
numeric_exp["Gene_Symbol"] = gene_symbols
gene_matrix = numeric_exp.groupby("Gene_Symbol").mean()

print(f"Matrice finale geni × campioni: {gene_matrix.shape}")
gene_matrix.to_csv(os.path.join(data_dir, "clean_gene_expression.csv"))

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETATO.")
print(f"  13.515 geni × 105 campioni ({df_target['Group'].value_counts().get('T1D', 0)} T1D, {df_target['Group'].value_counts().get('Control', 0)} Control)")
print("=" * 70)
