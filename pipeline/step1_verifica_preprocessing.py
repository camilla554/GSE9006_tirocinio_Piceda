import pandas as pd
import numpy as np
from config import FILE_TRAINING, FILE_META_TRAINING

print(">> STEP 1: verifica preprocessing GSE9006")

matrice_geni = pd.read_csv(FILE_TRAINING, index_col=0)
metadati = pd.read_csv(FILE_META_TRAINING)

print(f"\nMatrice di espressione genica:")
print(f"  Geni (righe): {matrice_geni.shape[0]}")
print(f"  Campioni (colonne): {matrice_geni.shape[1]}")

campioni_t1d = metadati[metadati["Group"] == "T1D"]["GSM_ID"].tolist()
campioni_ctrl = metadati[metadati["Group"] == "Control"]["GSM_ID"].tolist()

print(f"\nMetadati campioni:")
print(f"  T1D: {len(campioni_t1d)}")
print(f"  Control: {len(campioni_ctrl)}")
print(f"  Totale: {len(campioni_t1d) + len(campioni_ctrl)}")

# Verifica che tutti i campioni siano presenti nella matrice
campioni_totali = campioni_t1d + campioni_ctrl
mancanti = set(campioni_totali) - set(matrice_geni.columns)
if mancanti:
    print(f"  ATTENZIONE: {len(mancanti)} campioni non trovati nella matrice!")
else:
    print(f"  OK: tutti i campioni presenti nella matrice.")

# Verifica scala log2
valori = matrice_geni.values.flatten()
valori = valori[~np.isnan(valori)]
q1, med, q3 = np.percentile(valori, [25, 50, 75])
print(f"\nVerifica scala log2:")
print(f"  Range valori: {valori.min():.2f} - {valori.max():.2f}")
print(f"  Mediana: {med:.2f}, Q1: {q1:.2f}, Q3: {q3:.2f}")
if q3 > 100:
    print(f"  I valori NON sono in scala log2 (Q3={q3:.2f} > 100). Serve trasformazione log2.")
else:
    print(f"  Dati probabilmente in scala log2 (Q3={q3:.2f} < 100).")

# Geni con varianza zero
geni_varianza_zero = matrice_geni.index[matrice_geni.std(axis=1) == 0].tolist()
print(f"\nGeni con varianza zero: {len(geni_varianza_zero)}")
if geni_varianza_zero:
    print(f"  Esempi: {geni_varianza_zero[:5]}")

print(f"\nSTEP 1 COMPLETATO.")
print(f"Dataset pronto: {matrice_geni.shape[0]} geni x {matrice_geni.shape[1]} campioni "
      f"({len(campioni_t1d)} T1D, {len(campioni_ctrl)} Control)")
