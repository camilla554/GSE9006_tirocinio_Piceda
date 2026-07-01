import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
import os
import json

from config import (
    CARTELLA_RISULTATI, GENI_T2D_RANA, FILE_TRAINING, CARTELLA_DATI
)

print("--- STEP 11: confronto T2D (RQ3) ---")

# ============================================================
# PARTE 1: PANNELLO T1D IDENTIFICATO
# ============================================================
file_pannello = os.path.join(CARTELLA_RISULTATI, "geni_pannello_descoperta.txt")
geni_t1d = pd.read_csv(file_pannello, header=None)[0].tolist()
print(f"\nPANNELLO T1D IDENTIFICATO: {len(geni_t1d)} geni")
for i, g in enumerate(geni_t1d, 1):
    print(f"  {i:2d}. {g}")

# Coefficienti
coeff_df = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "coefficienti_pannello.csv"))
print(f"\nDirezione espressione (T1D vs Control):")
for _, riga in coeff_df.iterrows():
    direzione = "Più espresso in T1D" if riga["Coefficiente"] > 0 else "Meno espresso in T1D"
    print(f"  {riga['Gene']:20s}  {direzione}")

# ============================================================
# PARTE 2: CONFRONTO CON T2D DI RANA (RQ3)
# ============================================================
print(f"\n{'=' * 50}")
print("CONFRONTO CON PANNELLO T2D DI RANA (2025)")
print(f"{'=' * 50}")
print(f"Geni T2D (Rana): {len(GENI_T2D_RANA)}")
for i, g in enumerate(GENI_T2D_RANA, 1):
    print(f"  {i:2d}. {g}")

geni_comuni = set(geni_t1d) & set(GENI_T2D_RANA)
geni_specifici_t1d = set(geni_t1d) - set(GENI_T2D_RANA)
geni_specifici_t2d = set(GENI_T2D_RANA) - set(geni_t1d)

print(f"\n--- RISULTATI RQ3 ---")
print(f"Geni COMUNI (T1D & T2D): {len(geni_comuni)}")
for g in geni_comuni:
    print(f"  {g}")

print(f"\nGeni SPECIFICI T1D: {len(geni_specifici_t1d)}")
for g in sorted(geni_specifici_t1d):
    print(f"  {g}")

print(f"\nGeni SPECIFICI T2D (Rana): {len(geni_specifici_t2d)}")
for g in sorted(geni_specifici_t2d):
    print(f"  {g}")

# ============================================================
# PARTE 3: DEG SEPARATA PER T2D SU GSE9006
# ============================================================
print(f"\n{'=' * 50}")
print("ANALISI DEG SEPARATA PER T2D SU GSE9006")
print(f"{'=' * 50}")

matrice_geni = pd.read_csv(FILE_TRAINING, index_col=0)
meta_completo = pd.read_csv(os.path.join(CARTELLA_DATI, "target_metadata.csv"))

campioni_t2d = meta_completo[meta_completo["Group"] == "T2D"]["GSM_ID"].tolist()
campioni_ctrl = meta_completo[meta_completo["Group"] == "Control"]["GSM_ID"].tolist()
campioni_t1d = meta_completo[meta_completo["Group"] == "T1D"]["GSM_ID"].tolist()

print(f"\nCampioni GSE9006 (tutti):")
print(f"  T1D: {len(campioni_t1d)}")
print(f"  T2D: {len(campioni_t2d)}")
print(f"  Control: {len(campioni_ctrl)}")

# Verifica quali campioni T2D sono nella matrice GPL96
t2d_in_mat = [s for s in campioni_t2d if s in matrice_geni.columns]
ctrl_in_mat = [s for s in campioni_ctrl if s in matrice_geni.columns]

if len(t2d_in_mat) > 0 and len(ctrl_in_mat) > 1:
    print(f"\nEsecuzione DEG T2D vs Control su GSE9006...")
    print(f"  T2D: {len(t2d_in_mat)}, Control: {len(ctrl_in_mat)}")

    dati_t2d = matrice_geni[t2d_in_mat]
    dati_ctrl = matrice_geni[ctrl_in_mat]

    risultati_t2d = []
    for gene in matrice_geni.index:
        valori_t2d = dati_t2d.loc[gene].values.astype(float)
        valori_ctrl = dati_ctrl.loc[gene].values.astype(float)
        t_stat, p_val = stats.ttest_ind(valori_t2d, valori_ctrl, equal_var=False)
        log2fc = np.mean(valori_t2d) - np.mean(valori_ctrl)
        risultati_t2d.append({"Gene_Symbol": gene, "Log2FC": log2fc, "P_Valore": p_val})

    df_t2d = pd.DataFrame(risultati_t2d).dropna(subset=["P_Valore"])
    _, fdr_vals, _, _ = multipletests(df_t2d["P_Valore"], method="fdr_bh")
    df_t2d["FDR"] = fdr_vals
    df_t2d = df_t2d.sort_values("FDR")
    df_t2d.to_csv(os.path.join(CARTELLA_RISULTATI, "deg_t2d_vs_control.csv"), index=False)

    t2d_sig = df_t2d[(df_t2d["FDR"] < 0.05) & (df_t2d["Log2FC"].abs() > 0.58)]
    print(f"\nGeni differenzialmente espressi T2D: {len(t2d_sig)}")
    for _, riga in t2d_sig.head(20).iterrows():
        print(f"  {riga['Gene_Symbol']:20s}  Log2FC={riga['Log2FC']:+.3f}  FDR={riga['FDR']:.6f}")

    sovrapposizione = set(t2d_sig.head(20)["Gene_Symbol"]) & set(geni_t1d)
    print(f"\nSovrapposizione T1D panel con DEG T2D: {len(sovrapposizione)}")
    for g in sovrapposizione:
        print(f"  {g}")
else:
    print(f"\nCampioni T2D non disponibili nella matrice GPL96.")
    print(f"T2D in matrice: {len(t2d_in_mat)}, Control in matrice: {len(ctrl_in_mat)}")

# Salva risultati RQ3
risultati_rq3 = {
    "n_geni_t1d": len(geni_t1d),
    "geni_t1d": geni_t1d,
    "n_geni_t2d_rana": len(GENI_T2D_RANA),
    "geni_t2d_rana": GENI_T2D_RANA,
    "geni_comuni": sorted(list(geni_comuni)) if geni_comuni else [],
    "geni_specifici_t1d": sorted(list(geni_specifici_t1d)),
    "geni_specifici_t2d": sorted(list(geni_specifici_t2d))
}
with open(os.path.join(CARTELLA_RISULTATI, "risultati_rq3.json"), "w") as f:
    json.dump(risultati_rq3, f, indent=2)

print(f"\nSTEP 11 COMPLETATO.")
