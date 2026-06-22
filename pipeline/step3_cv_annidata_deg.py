import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import os
import json

from config import (
    FILE_TRAINING, FILE_META_TRAINING, CARTELLA_RISULTATI,
    NUM_FOLD_OUTER, NUM_FOLD_INNER, SOGLIA_LOG2FC, SOGLIA_FDR,
    MAX_DEG_PER_FOLD, MIN_DEG_PER_FOLD, SEME_CASUALE
)

print("=== STEP 3: nested CV + DEG ===")
print(f"Fold esterni: {NUM_FOLD_OUTER} (valutazione)")
print(f"Fold interni: {NUM_FOLD_INNER} (tuning Random Forest)")
print(f"Soglie DEG: |Log2FC| > {SOGLIA_LOG2FC}, FDR < {SOGLIA_FDR}")
print(f"Max DEG per fold: {MAX_DEG_PER_FOLD}")

# Carica dati
matrice_geni = pd.read_csv(FILE_TRAINING, index_col=0)
metadati = pd.read_csv(FILE_META_TRAINING)

campioni_t1d = metadati[metadati["Group"] == "T1D"]["GSM_ID"].tolist()
campioni_ctrl = metadati[metadati["Group"] == "Control"]["GSM_ID"].tolist()
tutti_campioni = campioni_t1d + campioni_ctrl

X = matrice_geni[tutti_campioni].T.values
y = np.array([1] * len(campioni_t1d) + [0] * len(campioni_ctrl))
nomi_geni = matrice_geni.index.tolist()

# Cross-validazione esterna (5 fold)
cv_esterna = StratifiedKFold(n_splits=NUM_FOLD_OUTER, shuffle=True, random_state=SEME_CASUALE)

risultati_fold = {}
liste_deg = {}
auc_fold = {}

for indice_esterno, (indici_train, indici_test) in enumerate(cv_esterna.split(X, y)):
    nome_fold = f"Fold_{indice_esterno + 1}"
    print(f"\n{'=' * 50}")
    print(f"{nome_fold}: {len(indici_train)} train, {len(indici_test)} test")
    print(f"  Train: {np.sum(y[indici_train])} T1D, {np.sum(y[indici_train] == 0)} Control")
    print(f"  Test:  {np.sum(y[indici_test])} T1D, {np.sum(y[indici_test] == 0)} Control")

    # Campioni di training e test per questo fold
    campioni_train = np.array(tutti_campioni)[indici_train]
    campioni_test = np.array(tutti_campioni)[indici_test]

    train_t1d = [s for s in campioni_train if s in campioni_t1d]
    train_ctrl = [s for s in campioni_train if s in campioni_ctrl]

    dati_t1d = matrice_geni[train_t1d]
    dati_ctrl = matrice_geni[train_ctrl]

    # DEG analysis: confronta T1D vs Control SOLO sul training fold
    risultati_deg = []
    for gene in nomi_geni:
        valori_t1d = dati_t1d.loc[gene].values.astype(float)
        valori_ctrl = dati_ctrl.loc[gene].values.astype(float)

        t_stat, p_val = stats.ttest_ind(valori_t1d, valori_ctrl, equal_var=False)
        log2fc = np.mean(valori_t1d) - np.mean(valori_ctrl)

        risultati_deg.append({
            "Gene_Symbol": gene,
            "Log2FC": log2fc,
            "T_Statistica": t_stat,
            "P_Valore": p_val
        })

    df_deg = pd.DataFrame(risultati_deg)
    df_deg = df_deg.dropna(subset=["P_Valore"]).copy()

    # Correzione per test multipli (Benjamini-Hochberg FDR)
    _, fdr_valori, _, _ = multipletests(df_deg["P_Valore"], method="fdr_bh")
    df_deg["FDR"] = fdr_valori
    df_deg = df_deg.sort_values("FDR")

    # Filtra geni significativi
    deg_significativi = df_deg[
        (df_deg["FDR"] < SOGLIA_FDR) &
        (df_deg["Log2FC"].abs() > SOGLIA_LOG2FC)
    ].copy()

    n_deg = len(deg_significativi)
    print(f"  DEG trovati: {n_deg}")

    if n_deg >= MIN_DEG_PER_FOLD:
        deg_selezionati = deg_significativi.head(MAX_DEG_PER_FOLD)
    else:
        print(f"  ATTENZIONE: {n_deg} DEG < soglia minima {MIN_DEG_PER_FOLD}")
        print(f"  Uso i top {MAX_DEG_PER_FOLD} geni per FDR come fallback.")
        deg_selezionati = df_deg.head(MAX_DEG_PER_FOLD)

    liste_deg[nome_fold] = deg_selezionati["Gene_Symbol"].tolist()

    # INNER CV (3-fold): tuning del parametro mtry per Random Forest
    # Come in Rana (2025): caret con 3-fold CV interno per trovare il miglior mtry
    from sklearn.model_selection import GridSearchCV

    rf_base = RandomForestClassifier(n_estimators=500, random_state=SEME_CASUALE)
    griglia_mtry = {"max_features": [int(len(deg_selezionati) ** 0.5),
                                      max(1, len(deg_selezionati) // 3),
                                      max(1, len(deg_selezionati) // 2)]}
    griglia_mtry["max_features"] = [v for v in set(griglia_mtry["max_features"]) if v > 0]

    inner_cv = StratifiedKFold(n_splits=NUM_FOLD_INNER, shuffle=True, random_state=SEME_CASUALE)
    ricerca_inner = GridSearchCV(rf_base, griglia_mtry, cv=inner_cv, scoring="roc_auc")
    dati_train_fold = matrice_geni[list(campioni_train)].loc[deg_selezionati["Gene_Symbol"]].T.values
    etichette_train = np.array([1 if s in campioni_t1d else 0 for s in campioni_train])

    ricerca_inner.fit(dati_train_fold, etichette_train)
    rf_ottimizzato = ricerca_inner.best_estimator_
    print(f"  Inner CV: miglior mtry = {ricerca_inner.best_params_['max_features']} (AUC inner = {ricerca_inner.best_score_:.4f})")

    # Valutazione out-of-fold con Random Forest ottimizzato
    dati_test_fold = matrice_geni[list(campioni_test)].loc[deg_selezionati["Gene_Symbol"]].T.values
    etichette_test = np.array([1 if s in campioni_t1d else 0 for s in campioni_test])

    predizioni = rf_ottimizzato.predict_proba(dati_test_fold)[:, 1]
    auc = roc_auc_score(etichette_test, predizioni)
    auc_fold[nome_fold] = auc
    print(f"  AUC Random Forest (out-of-fold, con inner CV): {auc:.4f}")

    risultati_fold[nome_fold] = {
        "n_train": len(indici_train),
        "n_test": len(indici_test),
        "n_deg": n_deg,
        "geni_deg": deg_selezionati["Gene_Symbol"].tolist(),
        "auc_rf": auc
    }

# Riepilogo
print(f"\n{'=' * 50}")
print(f"RIEPILOGO FOLD:")
valori_auc = list(auc_fold.values())
for fold, dati in risultati_fold.items():
    print(f"  {fold}: {dati['n_deg']} DEG, AUC RF = {dati['auc_rf']:.4f}")
print(f"Media AUC: {np.mean(valori_auc):.4f} (+/- {np.std(valori_auc):.4f})")

# Salva risultati
with open(os.path.join(CARTELLA_RISULTATI, "risultati_deg_fold.json"), "w") as f:
    salva = {k: {"n_deg": v["n_deg"], "auc_rf": float(v["auc_rf"]), "geni_deg": v["geni_deg"]}
             for k, v in risultati_fold.items()}
    json.dump(salva, f, indent=2)

# Calcola la frequenza di selezione di ogni gene attraverso i fold
tutti_geni_deg = []
for geni in liste_deg.values():
    tutti_geni_deg.extend(geni)
frequenza_geni = pd.Series(tutti_geni_deg).value_counts()
frequenza_geni.to_csv(os.path.join(CARTELLA_RISULTATI, "frequenza_geni_attraverso_fold.csv"))
print(f"\nGeni unici selezionati: {len(frequenza_geni)}")
print(f"Top 15 geni per frequenza:")
print(frequenza_geni.head(15))

print(f"\nSTEP 3 COMPLETATO.")
