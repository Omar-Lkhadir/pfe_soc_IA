"""
Nettoyage + adaptation canonique de CSE-CIC-IDS2018 (4 jours), ajoutés comme
DIVERSITÉ SUPPLÉMENTAIRE à la source 'cicflowmeter' existante (même outil que
CICIDS2017, réseau/période différents) -> répond au constat de TEST 4 : le
Random Forest généralise mal entre deux datasets du même outil.

CICIDS2018 utilise des noms de colonnes ABRÉGÉS, différents de CICIDS2017
(ex. 'Dst Port' vs 'Destination Port') -> renommage avant réutilisation de
cicflowmeter_adapter.py (inchangé).

Certains fichiers CICIDS2018 contiennent des lignes corrompues/en-têtes
répétées en milieu de fichier (valeurs numériques ou "Label" litéral dans la
colonne Label) -> filtrées.

Source : https://www.unb.ca/cic/datasets/ids-2018.html (AWS S3 public)
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from commun.adapters.cicflowmeter_adapter import CICFlowMeterAdapter
from commun.mapping_attaques import mapper

MD4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DIR = os.path.join(MD4_DIR, '..', 'TEST', 'test4', 'data', 'raw')
OUT_DIR = os.path.join(MD4_DIR, 'commun', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

RENAME_2018_VERS_2017 = {
    'Dst Port': 'Destination Port',
    'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
    'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets',
    'Pkt Len Max': 'Max Packet Length',
    'Pkt Len Min': 'Min Packet Length',
    'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': 'SYN Flag Count',
    'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count',
    'ACK Flag Cnt': 'ACK Flag Count',
    'URG Flag Cnt': 'URG Flag Count',
}
COLS_2018 = list(RENAME_2018_VERS_2017.keys()) + [
    'Protocol', 'Flow Duration', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Label',
]

LABELS_VALIDES = {
    'Benign', 'FTP-BruteForce', 'SSH-Bruteforce',
    'DoS attacks-GoldenEye', 'DoS attacks-Slowloris',
    'Bot', 'Infilteration',
}

FICHIERS = [
    ('Wednesday-14-02-2018.csv', {'FTP-BruteForce': 50_000, 'SSH-Bruteforce': 50_000}, 40_000),
    ('Thursday-15-02-2018.csv', {'DoS attacks-GoldenEye': 40_000, 'DoS attacks-Slowloris': 10_990}, 40_000),
    ('Friday-02-03-2018.csv', {'Bot': 50_000}, 40_000),
    ('Thursday-01-03-2018.csv', {'Infilteration': 50_000}, 40_000),
]

print("=" * 70)
print("NETTOYAGE + ADAPTATION — CSE-CIC-IDS2018 (source cicflowmeter, diversité)")
print("=" * 70)

adapter = CICFlowMeterAdapter()
morceaux = []

for fichier, cibles_attaque, cible_normal in FICHIERS:
    chemin = os.path.join(RAW_DIR, fichier)
    print(f"\n📂 Lecture de {fichier}...")
    df_raw = pd.read_csv(chemin, usecols=lambda c: c.strip() in COLS_2018, low_memory=False)
    df_raw = df_raw.rename(columns=RENAME_2018_VERS_2017)

    avant = len(df_raw)
    df_raw = df_raw[df_raw['Label'].isin(LABELS_VALIDES)].reset_index(drop=True)
    print(f"   ✓ {avant:,} -> {len(df_raw):,} lignes (labels corrompus/en-têtes répétées écartés)")
    print(f"   Labels : {df_raw['Label'].value_counts().to_dict()}")

    df_canon = adapter.extract(df_raw)
    del df_raw
    df_canon['dataset'] = 'cicids2018'

    cat_5 = pd.Series(
        np.where(df_canon['Label_binaire'] == 1, df_canon['Attack_brut'].apply(mapper), 'Benign'),
        index=df_canon.index,
    )

    for label_brut, cible_n in cibles_attaque.items():
        mask = df_canon['Attack_brut'] == label_brut
        n_dispo = int(mask.sum())
        n_pris = min(cible_n, n_dispo)
        morceaux.append(df_canon[mask].sample(n=n_pris, random_state=42))
        print(f"   ✓ {label_brut:<28s} ({cat_5[mask].iloc[0] if n_dispo else '?'}) : "
              f"{n_dispo:,} dispo -> {n_pris:,} retenus")

    mask_normal = df_canon['Label_binaire'] == 0
    n_normal_dispo = int(mask_normal.sum())
    n_normal_pris = min(cible_normal, n_normal_dispo)
    morceaux.append(df_canon[mask_normal].sample(n=n_normal_pris, random_state=42))
    print(f"   ✓ Benign : {n_normal_dispo:,} dispo -> {n_normal_pris:,} retenus")

    del df_canon

df_final = pd.concat(morceaux, ignore_index=True)
print(f"\n🔗 Total CICIDS2018 retenu : {len(df_final):,} lignes")
print(f"   Normal={int((df_final['Label_binaire']==0).sum()):,} | "
      f"Attaques={int((df_final['Label_binaire']==1).sum()):,}")

df_final.to_pickle(os.path.join(OUT_DIR, 'cicflowmeter_2018_canonical.pkl'))
print(f"\n💾 Sauvegardé -> commun/data/cicflowmeter_2018_canonical.pkl")
print("=" * 70)
