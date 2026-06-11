import pandas as pd
import numpy as np
import gzip
import re
import os

from config import (
    FILE_VALIDAZIONE_RAW, FILE_META_VALIDAZIONE, FILE_EXPR_VALIDAZIONE,
    FILE_VALIDAZIONE_PULITO, FILE_TRAINING, CARTELLA_DATI
)

print("== STEP 2: preparazione GSE33440 ==")

if not os.path.exists(FILE_VALIDAZIONE_RAW):
    print(f"ERRORE: File {FILE_VALIDAZIONE_RAW} non trovato.")
    exit(1)

# Percorso del file di mappatura sonde Illumina -> geni
FILE_MAPPA_SONDE = os.path.join(CARTELLA_DATI, "..", "..", "datasets", "illumina_gpl6947_probe_gene.csv")

print("Lettura file series matrix GSE33440...")
with gzip.open(FILE_VALIDAZIONE_RAW, "rt", encoding="latin1") as f:
    righe = f.readlines()

id_campioni = []
gruppi_campioni = []
titoli_campioni = []
in_tabella = False
linea_intestazione = None
linee_dati = []

for riga in righe:
    s = riga.strip()

    if s.startswith("!Sample_geo_accession"):
        parti = s.split("\t")
        id_campioni = [p.strip().strip('"') for p in parti[1:]]

    elif s.startswith("!Sample_title"):
        parti = s.split("\t")
        titoli_campioni = [p.strip().strip('"') for p in parti[1:]]

    elif s.startswith("!Sample_characteristics_ch1") and "group:" in s:
        parti = s.split("\t")
        for p in parti[1:]:
            valore = p.strip().strip('"')
            m = re.search(r"group:\s*(.+)", valore)
            if m:
                g = m.group(1).strip()
                gruppi_campioni.append("Control" if g == "Healthy" else "T1D")

    elif s.startswith("!series_matrix_table_begin"):
        in_tabella = True
        continue
    elif s.startswith("!series_matrix_table_end"):
        break
    elif in_tabella:
        if linea_intestazione is None:
            linea_intestazione = s
        else:
            linee_dati.append(s)

n = min(len(id_campioni), len(gruppi_campioni), len(titoli_campioni))
id_campioni = id_campioni[:n]
gruppi_campioni = gruppi_campioni[:n]
titoli_campioni = titoli_campioni[:n]

metadati = pd.DataFrame({"GSM_ID": id_campioni, "Titolo": titoli_campioni, "Group": gruppi_campioni})
metadati.to_csv(FILE_META_VALIDAZIONE, index=False)
print(f"\nMetadati ({len(metadati)} campioni):")
print(metadati["Group"].value_counts())

print("Estrazione matrice di espressione...")
tutte_le_righe = []
for riga in linee_dati:
    parti = riga.split("\t")
    id_sonda = parti[0].strip().strip('"')
    valori = [float(x) if x.strip().strip('"') != "" else np.nan for x in parti[1:n+1]]
    tutte_le_righe.append([id_sonda] + valori)

expr_df = pd.DataFrame(tutte_le_righe, columns=["ID_SONDA"] + id_campioni).set_index("ID_SONDA")[id_campioni].astype(float)
expr_df.to_csv(FILE_EXPR_VALIDAZIONE)
print(f"Matrice: {expr_df.shape[0]} sonde x {expr_df.shape[1]} campioni")

# Verifica scala log2
valori = expr_df.values.flatten()
valori = valori[~np.isnan(valori)]
q3 = np.percentile(valori, 75)
print(f"Q3 = {q3:.2f}")
if q3 > 100:
    print("Applico log2(x+1)...")
    expr_df = np.log2(expr_df + 1)

print("\nMappatura sonda -> gene...")
if os.path.exists(FILE_MAPPA_SONDE):
    mappa_df = pd.read_csv(FILE_MAPPA_SONDE)
    mappa_sonda_gene = dict(zip(mappa_df["ProbeID"], mappa_df["GeneSymbol"]))
    if "ID" in mappa_sonda_gene:
        del mappa_sonda_gene["ID"]
    print(f"Mappature caricate: {len(mappa_sonda_gene)}")
else:
    print(f"ERRORE: File mappa non trovato: {FILE_MAPPA_SONDE}")
    exit(1)

expr_df["Gene_Symbol"] = expr_df.index.map(mappa_sonda_gene)
expr_df = expr_df[expr_df["Gene_Symbol"].notna()].copy()

# Aggrega più sonde per lo stesso gene (media)
geni_aggregati = expr_df.groupby("Gene_Symbol")[id_campioni].mean()
geni_aggregati = geni_aggregati.loc[geni_aggregati.std(axis=1) > 0]

geni_aggregati.to_csv(FILE_VALIDAZIONE_PULITO)
print(f"\nGeni: {len(geni_aggregati)}, Campioni: {geni_aggregati.shape[1]}")

# Geni condivisi con il training set
geni_training = pd.read_csv(FILE_TRAINING, index_col=0).index
condivisi = geni_aggregati.index.intersection(geni_training)
print(f"Geni condivisi con training: {len(condivisi)}/{len(geni_aggregati)}")

print(f"\nSTEP 2 COMPLETATO.")
print(f"Validation set pronto: {geni_aggregati.shape[0]} geni x {geni_aggregati.shape[1]} campioni")
