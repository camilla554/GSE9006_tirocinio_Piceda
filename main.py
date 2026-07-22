import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

data_dir = "./data"

print("--- STEP 3: ANALISI DI ESPRESSIONE DIFFERENZIALE (DEG) ---")

# 1. Caricamento dati generati nello Step 2
gene_matrix = pd.read_csv(os.path.join(data_dir, "clean_gene_expression.csv"), index_col=0)
meta_df = pd.read_csv(os.path.join(data_dir, "target_metadata_GPL96.csv"))

# Identificazione dei campioni per ciascun gruppo
t1d_samples = meta_df[meta_df["Group"] == "T1D"]["GSM_ID"].tolist()
ctrl_samples = meta_df[meta_df["Group"] == "Control"]["GSM_ID"].tolist()

t1d_data = gene_matrix[t1d_samples]
ctrl_data = gene_matrix[ctrl_samples]

# 2. Calcolo Log2 Fold Change e T-Test per ciascun gene
results = []

for gene in gene_matrix.index:
    t1d_vals = t1d_data.loc[gene]
    ctrl_vals = ctrl_data.loc[gene]

    mean_t1d = np.mean(t1d_vals)
    mean_ctrl = np.mean(ctrl_vals)

    # Log2 Fold Change (differenza tra le medie in scala logaritmica)
    log2fc = mean_t1d - mean_ctrl

    # Welch's t-test (non assume uguali varianze tra i due gruppi)
    t_stat, p_val = stats.ttest_ind(t1d_vals, ctrl_vals, equal_var=False)

    results.append({
        "Gene_Symbol": gene,
        "Mean_T1D": mean_t1d,
        "Mean_Control": mean_ctrl,
        "Log2FC": log2fc,
        "P_Value": p_val
    })

df_deg = pd.DataFrame(results)

# 3. Correzione per test multipli (Benjamini-Hochberg FDR)
df_deg = df_deg.dropna(subset=["P_Value"]).copy()
df_deg["FDR"] = multipletests(df_deg["P_Value"], method="fdr_bh")[1]

# 4. Ordinamento per significatività (FDR crescente)
df_deg = df_deg.sort_values(by="FDR", ascending=True)

# 5. Filtraggio dei geni significativi
# Soglie standard: FDR < 0.05 e |Log2FC| > 0.58 (almeno ~1.5x di variazione)
log2fc_threshold = 0.58
fdr_threshold = 0.05

deg_sig = df_deg[(df_deg["FDR"] < fdr_threshold) & (df_deg["Log2FC"].abs() > log2fc_threshold)]

# Classificazione in UP e DOWN regulated
deg_up = deg_sig[deg_sig["Log2FC"] > log2fc_threshold]
deg_down = deg_sig[deg_sig["Log2FC"] < -log2fc_threshold]

print(f"\n--- RISULTATI ANALISI DEG ---")
print(f"Geni totali analizzati: {len(df_deg)}")
print(f"Geni significativi totali (FDR < {fdr_threshold}, |Log2FC| > {log2fc_threshold}): {len(deg_sig)}")
print(f"  -> Geni Sovra-espressi (UP in T1D): {len(deg_up)}")
print(f"  -> Geni Sotto-espressi (DOWN in T1D): {len(deg_down)}")

# 6. Salvataggio su file CSV
df_deg.to_csv(os.path.join(data_dir, "deg_all_genes.csv"), index=False)
deg_sig.to_csv(os.path.join(data_dir, "deg_significant_genes.csv"), index=False)

print(f"\nPrimi 10 geni più significativi:")
print(deg_sig[["Gene_Symbol", "Log2FC", "P_Value", "FDR"]].head(10).to_string(index=False))

print("\n--- STEP 3 COMPLETATO CON SUCCESSO! ---")