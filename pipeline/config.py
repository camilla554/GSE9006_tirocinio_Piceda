import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CARTELLA_DATI = os.path.join(BASE_DIR, "data")
CARTELLA_RISULTATI = os.path.join(BASE_DIR, "pipeline", "risultati")

os.makedirs(CARTELLA_RISULTATI, exist_ok=True)

FILE_TRAINING = os.path.join(CARTELLA_DATI, "clean_gene_expression.csv")
FILE_META_TRAINING = os.path.join(CARTELLA_DATI, "target_metadata_GPL96.csv")

FILE_VALIDAZIONE_RAW = os.path.join(
    os.path.dirname(BASE_DIR), "datasets", "GSE33440_series_matrix.txt.gz"
)
FILE_META_VALIDAZIONE = os.path.join(CARTELLA_RISULTATI, "metadata_GSE33440.csv")
FILE_EXPR_VALIDAZIONE = os.path.join(CARTELLA_RISULTATI, "espressione_GSE33440.csv")
FILE_VALIDAZIONE_PULITO = os.path.join(CARTELLA_RISULTATI, "espressione_validazione_pulita.csv")

NUM_FOLD_OUTER = 5
NUM_FOLD_INNER = 3

SOGLIA_LOG2FC = 1.0
SOGLIA_FDR = 0.05
MAX_DEG_PER_FOLD = 50
MIN_DEG_PER_FOLD = 5

BOOTSTRAP_ELASTIC_NET = 500
ALPHA_ELASTIC_NET = 0.5
BORUTA_MAX_ITER = 200

SOGLIA_FREQ_FOLD = 0.6
SOGLIA_EN = 0.6
PESO_FREQ = 0.5
PESO_EN = 0.4
PESO_FC = 0.1
MIN_PANNELLO = 5
MAX_PANNELLO = 20

GENI_T2D_RANA = [
    "ZNF428", "PDE6B", "SAC3D1", "TNKS2", "GJD2",
    "KCNAB3", "PWAR5", "BTBD2", "XAB2", "ETS1",
    "ZNF76", "RALGDS"
]
SEME_CASUALE = 42
N_PERMUTAZIONI = 500
BOOTSTRAP_AUC = 2000
PREVALENZA_SCREENING = 0.11
