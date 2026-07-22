"""
Esecuzione pipeline T1D biomarker discovery.
Usage: python esegui_tutto.py [--step N] [--salta-permutazione]
"""
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("0", "step0_scelta_dataset.py", "SCELTA DATASET (GSE9006 + GSE33440)"),
    ("1", "step1_verifica_preprocessing.py", "VERIFICA PREPROCESSING"),
    ("2", "step2_prepara_validazione.py", "PREPARAZIONE VALIDAZIONE (GSE33440)"),
    ("3", "step3_cv_annidata_deg.py", "NESTED CV + ANALISI DEG (con inner CV)"),
    ("4", "step4_selezione_feature.py", "SELEZIONE FEATURE (Boruta + Elastic Net)"),
    ("5", "step5_pannello_consenso.py", "PANNELLO DI CONSENSO"),
    ("6", "step6_correzione_batch.py", "CORREZIONE BATCH + PCA"),
    ("7", "step7_modello_finale.py", "MODELLO FINALE ELASTIC NET"),
    ("8", "step8_validazione_esterna.py", "VALIDAZIONE ESTERNA"),
    ("9", "step9_metriche_cliniche.py", "METRICHE CLINICHE"),
    ("10", "step10_test_permutazione.py", "TEST DI PERMUTAZIONE"),
    ("11", "step11_confronto_t2d_rq3.py", "INTERPRETAZIONE + RQ3"),
]


def interpreta_argomenti():
    args = sys.argv[1:]
    salta_perm = "--salta-permutazione" in args

    if "--step" in args:
        idx = args.index("--step")
        valore = args[idx + 1]
        if "-" in valore:
            inizio, fine = map(int, valore.split("-"))
            return inizio, fine, salta_perm
        else:
            step = int(valore)
            return step, step, salta_perm

    return 1, len(STEPS), salta_perm


def main():
    inizio, fine, salta_perm = interpreta_argomenti()

    if salta_perm:
        print("NOTA: --salta-permutazione attivo, salto test di permutazione.")

    for numero, script, descrizione in STEPS:
        s = int(numero)
        if s < inizio or s > fine:
            continue
        if salta_perm and s == 10:
            print(f"\n{'=' * 70}")
            print(f"STEP {numero}: {descrizione} - SALTATO")
            print(f"{'=' * 70}")
            continue

        print(f"\n{'=' * 70}")
        print(f"STEP {numero}: {descrizione}")
        print(f"{'=' * 70}")

        codice_uscita = os.system(f"python {script}")

        if codice_uscita != 0:
            print(f"\nERRORE: Step {numero} fallito (codice {codice_uscita})")
            sys.exit(1)

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETATA CON SUCCESSO!")
    print(f"{'=' * 70}")
    print(f"\nTutti i risultati sono in: {os.path.join(os.getcwd(), 'risultati')}")


if __name__ == "__main__":
    main()
