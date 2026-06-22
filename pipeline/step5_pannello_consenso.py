import pandas as pd
import numpy as np
import os
import json

from config import (
    CARTELLA_RISULTATI, CARTELLA_DATI,
    SOGLIA_FREQ_FOLD, SOGLIA_EN,
    PESO_FREQ, PESO_EN, PESO_FC,
    MIN_PANNELLO, MAX_PANNELLO
)

print(">> STEP 5: consensus e pannello finale")

# Carica i risultati degli step precedenti
frequenza_deg = pd.read_csv(
    os.path.join(CARTELLA_RISULTATI, "frequenza_geni_attraverso_fold.csv"),
    index_col=0
)
frequenza_deg.index.name = "Gene_Symbol"
frequenza_deg.columns = ["Frequenza_Selezione"]

# Calcola fold totali
try:
    with open(os.path.join(CARTELLA_RISULTATI, "risultati_deg_fold.json")) as f:
        dati_fold = json.load(f)
    totale_fold = len(dati_fold)
except:
    totale_fold = 5  # default

frequenza_deg["Freq_Normalizzata"] = frequenza_deg["Frequenza_Selezione"] / totale_fold

boruta_df = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "risultati_boruta.csv"))
en_df = pd.read_csv(os.path.join(CARTELLA_RISULTATI, "risultati_elastic_net_bootstrap.csv"))

print(f"\nFrequenza DEG attraverso fold: {len(frequenza_deg)} geni unici")
print(f"Risultati Boruta: {len(boruta_df)} geni")
print(f"Risultati Elastic Net: {len(en_df)} geni")

# Unisci tutti i risultati
combinato = frequenza_deg.reset_index().merge(
    boruta_df, on="Gene_Symbol", how="outer"
).merge(
    en_df, on="Gene_Symbol", how="outer"
)

# Valori di default per i geni che mancano in qualche analisi
combinato = combinato.fillna({
    "Frequenza_Selezione": 0,
    "Freq_Normalizzata": 0,
    "Boruta_Selezionato": False,
    "Boruta_Rank": combinato["Boruta_Rank"].max() + 1 if "Boruta_Rank" in combinato.columns else 999,
    "Prob_EN": 0
})

# Carica |Log2FC| dal file DEG completo
percorso_deg = os.path.join(CARTELLA_DATI, "deg_all_genes.csv")
if os.path.exists(percorso_deg):
    deg_full = pd.read_csv(percorso_deg)
    mappa_fc = dict(zip(deg_full["Gene_Symbol"], deg_full["Log2FC"].abs()))
    combinato["Log2FC_abs"] = combinato["Gene_Symbol"].map(mappa_fc).fillna(0)
else:
    combinato["Log2FC_abs"] = 0

# Standardizza Log2FC
media_fc = combinato["Log2FC_abs"].mean()
std_fc = combinato["Log2FC_abs"].std()
combinato["Log2FC_Std"] = (combinato["Log2FC_abs"] - media_fc) / std_fc if std_fc > 0 else 0

print(f"\nCalcolo punteggio di consenso per ogni gene...")

# Punteggio pesato: 50% frequenza, 40% Elastic Net, 10% Log2FC
combinato["Punteggio_Consenso"] = (
    PESO_FREQ * combinato["Freq_Normalizzata"] +
    PESO_EN * combinato["Prob_EN"] +
    PESO_FC * combinato["Log2FC_Std"]
)

# Criteri per essere candidato:
# - Frequenza fold >= 60%
# - (Probabilità EN >= 60% OPPURE confermato da Boruta)
combinato["Soddisfa_Criteri"] = (
    (combinato["Freq_Normalizzata"] >= SOGLIA_FREQ_FOLD) &
    ((combinato["Prob_EN"] >= SOGLIA_EN) | combinato["Boruta_Selezionato"])
)

geni_candidati = combinato[combinato["Soddisfa_Criteri"]].sort_values("Punteggio_Consenso", ascending=False)

print(f"Geni che soddisfano i criteri: {len(geni_candidati)}")

# Se non ci sono abbastanza candidati, usa fallback
if len(geni_candidati) < MIN_PANNELLO:
    print(f"ATTENZIONE: {len(geni_candidati)} geni < soglia minima {MIN_PANNELLO}")
    print(f"Uso top {MIN_PANNELLO * 3} geni per punteggio consenso come fallback.")
    geni_candidati = combinato.sort_values("Punteggio_Consenso", ascending=False).head(MIN_PANNELLO * 3)

# Pannello finale (max 20 geni)
pannello_finale = geni_candidati.head(MAX_PANNELLO)

print(f"\n{'-' * 50}")
print(f"PANNELLO FINALE: {len(pannello_finale)} GENI")
print(f"{'-' * 50}")

pannello_finale.to_csv(os.path.join(CARTELLA_RISULTATI, "classifica_consenso.csv"), index=False)

geni_pannello = pannello_finale["Gene_Symbol"].tolist()
pd.Series(geni_pannello).to_csv(
    os.path.join(CARTELLA_RISULTATI, "geni_pannello_descoperta.txt"),
    index=False, header=False
)

# Riepilogo
riepilogo = {
    "totale_geni_unici_deg": int(len(frequenza_deg)),
    "boruta_confermati": int(boruta_df["Boruta_Selezionato"].sum()),
    "geni_candidati": int(len(geni_candidati)),
    "dimensione_pannello_finale": int(len(pannello_finale)),
    "geni_pannello_finale": geni_pannello
}
with open(os.path.join(CARTELLA_RISULTATI, "riepilogo_consenso.json"), "w") as f:
    json.dump(riepilogo, f, indent=2)

print(f"\nGeni nel pannello:")
for i, gene in enumerate(geni_pannello, 1):
    punteggio = pannello_finale[pannello_finale["Gene_Symbol"] == gene]["Punteggio_Consenso"].values[0]
    print(f"  {i:2d}. {gene:20s} (Consenso: {punteggio:.4f})")

print(f"\nSTEP 5 COMPLETATO.")
print(f"Pannello di {len(geni_pannello)} geni salvato in geni_pannello_descoperta.txt")
