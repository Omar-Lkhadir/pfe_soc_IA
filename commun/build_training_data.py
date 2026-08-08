"""
Construit le jeu de données canonique combiné (NetFlow + CICFlowMeter) à
partir des adaptateurs de commun/adapters/. Étape 4 du plan de refonte
multi-source : produit un DataFrame canonique PAR SOURCE (checkpoint sur
disque), pour ne jamais avoir à relire le CSV de 14 Go si une étape
ultérieure (split, entraînement) échoue.

Sortie : commun/data/netflow_canonical.pkl, commun/data/cicflowmeter_canonical.pkl
(DataFrames pandas, colonnes = canonical_schema.ALL_OUTPUT_COLS)
"""

import os
import sys
import time
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from commun import canonical_schema as cs
from commun.adapters.netflow_adapter import NetFlowAdapter, RAW_COLS as NETFLOW_RAW_COLS
from commun.adapters.cicflowmeter_adapter import CICFlowMeterAdapter, COLS_VOULUES as CIC_COLS

MD4_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep + '..'
MD4_DIR = os.path.abspath(MD4_DIR)
OUT_DIR = os.path.join(MD4_DIR, 'commun', 'data')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# SOURCE 1 : NETFLOW (BigFlow-NIDS-V2.csv) — même échantillonnage stratifié
# que isolation_forest/scripts/nettoyage.py, pour rester comparable au v1.
# ============================================================
print("=" * 70)
print("CONSTRUCTION DU JEU CANONIQUE — NetFlow (BigFlow-NIDS-V2)")
print("=" * 70)

FICHIER_NETFLOW = os.path.join(MD4_DIR, 'data', 'raw', 'BigFlow-NIDS-V2.csv')
CHUNK_SIZE = 300_000
TOTAL_VISE = 3_000_000
RATIO_NORMAL = 0.68
CIBLE_NORMAL = int(TOTAL_VISE * RATIO_NORMAL)
CIBLE_ATTAQUES = TOTAL_VISE - CIBLE_NORMAL
TOTAL_BENIGN_CONNU = 36_596_560
TOTAL_ATTAQUES_CONNU = 30_338_461
PROB_BENIGN = min(1.0, CIBLE_NORMAL / TOTAL_BENIGN_CONNU)
PROB_ATTAQUE = min(1.0, CIBLE_ATTAQUES / TOTAL_ATTAQUES_CONNU)

np.random.seed(42)
t0 = time.time()
adapter_nf = NetFlowAdapter()
chunks_out = []
total_lu, total_garde = 0, 0

for i, chunk in enumerate(pd.read_csv(FICHIER_NETFLOW, chunksize=CHUNK_SIZE,
                                        usecols=lambda c: c.strip() in NETFLOW_RAW_COLS,
                                        low_memory=False)):
    total_lu += len(chunk)
    chunk.columns = [c.strip() for c in chunk.columns]
    chunk['Label'] = pd.to_numeric(chunk['Label'], errors='coerce').fillna(0)
    benign_mask = chunk['Label'] == 0
    probs = np.where(benign_mask, PROB_BENIGN, PROB_ATTAQUE)
    mask = np.random.rand(len(chunk)) < probs
    chunk = chunk[mask]
    if len(chunk) > 0:
        chunks_out.append(adapter_nf.extract(chunk))
    total_garde += len(chunk)
    if i % 20 == 0:
        print(f"   lues: {total_lu:>12,} | gardées: {total_garde:>10,} | {time.time()-t0:.0f}s")

df_netflow = pd.concat(chunks_out, ignore_index=True)
df_netflow = df_netflow.drop_duplicates(subset=cs.TIER1_FEATURES + cs.TIER2_FEATURES)
del chunks_out
print(f"\n✓ NetFlow : {len(df_netflow):,} lignes canoniques "
      f"(Normal={int((df_netflow.Label_binaire==0).sum()):,}, "
      f"Attaques={int((df_netflow.Label_binaire==1).sum()):,}) en {time.time()-t0:.0f}s")

df_netflow.to_pickle(os.path.join(OUT_DIR, 'netflow_canonical.pkl'))
print(f"✓ Sauvegardé -> commun/data/netflow_canonical.pkl")

# ============================================================
# SOURCE 2 : CICFLOWMETER (8 fichiers CICIDS2017, TEST/test1/data/raw)
# ============================================================
print("\n" + "=" * 70)
print("CONSTRUCTION DU JEU CANONIQUE — CICFlowMeter (CICIDS2017)")
print("=" * 70)

RAW_DIR_CIC = os.path.join(MD4_DIR, '..', 'TEST', 'test1', 'data', 'raw')
fichiers_cic = sorted(glob.glob(os.path.join(RAW_DIR_CIC, '*.csv')))
print(f"\n{len(fichiers_cic)} fichiers trouvés : {[os.path.basename(f) for f in fichiers_cic]}")

adapter_cic = CICFlowMeterAdapter()
dfs_cic = []
t0 = time.time()
for fpath in fichiers_cic:
    jour = os.path.basename(fpath).split('.')[0]
    df_raw = pd.read_csv(fpath, encoding='latin1', low_memory=False,
                          usecols=lambda c: c.strip() in CIC_COLS)
    df_out = adapter_cic.extract(df_raw)
    df_out['jour'] = jour
    dfs_cic.append(df_out)
    print(f"   [{jour}] {len(df_out):,} lignes | "
          f"Normal={int((df_out.Label_binaire==0).sum()):,} "
          f"Attaques={int((df_out.Label_binaire==1).sum()):,}")

df_cic = pd.concat(dfs_cic, ignore_index=True)
del dfs_cic
print(f"\n✓ CICFlowMeter : {len(df_cic):,} lignes canoniques en {time.time()-t0:.0f}s")

df_cic.to_pickle(os.path.join(OUT_DIR, 'cicflowmeter_canonical.pkl'))
print(f"✓ Sauvegardé -> commun/data/cicflowmeter_canonical.pkl")

print("\n" + "=" * 70)
print("✅ CONSTRUCTION TERMINÉE")
print("=" * 70)
