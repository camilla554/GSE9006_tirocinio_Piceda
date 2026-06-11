"""
GSE9006 per training, GSE33440 per validazione esterna.
"""
import os
import pandas as pd
from config import FILE_TRAINING, FILE_META_TRAINING, FILE_VALIDAZIONE_RAW

print("--- STEP 0: GSE9006 + GSE33440 ---")

print("""
DATASET DI TRAINING: GSE9006
  - Piattaforma: Affymetrix GPL96 (HG-U133A)
  - Tipo cellulare: PBMC (Peripheral Blood Mononuclear Cells)
  - Scaricato da: Gene Expression Omnibus (GEO)
  - Campioni: 105 (81 T1D, 24 Control)
  - Reference: Kaizer et al., 2007

DATASET DI VALIDAZIONE: GSE33440
  - Piattaforma: Illumina GPL6947 (HumanHT-12 v3.0)
  - Tipo cellulare: Monociti CD14+
  - Scaricato da: Gene Expression Omnibus (GEO)
  - Campioni: 22 (16 T1D, 6 Control)
  - Reference: Irvine et al., 2012
""")

if os.path.exists(FILE_TRAINING):
    df = pd.read_csv(FILE_TRAINING, index_col=0)
    print(f"File training trovato: {df.shape[0]} geni x {df.shape[1]} campioni")
else:
    print(f"ATTENZIONE: File training non trovato in {FILE_TRAINING}")

if os.path.exists(FILE_META_TRAINING):
    df = pd.read_csv(FILE_META_TRAINING)
    print(f"Metadati training: {df['Group'].value_counts().to_dict()}")

if os.path.exists(FILE_VALIDAZIONE_RAW):
    print(f"File validazione trovato: {FILE_VALIDAZIONE_RAW}")
else:
    print(f"ATTENZIONE: File validazione non trovato")

print(f"\nFatto. Dataset pronti.")
