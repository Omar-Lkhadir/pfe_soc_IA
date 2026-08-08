"""
Étape 7 du plan : combine les deux sources canoniques et fait un split
train/test unique, stratifié par (source x label) pour que les deux formats
et les deux classes soient représentés dans les deux splits. Le split est
ensuite réutilisé tel quel par les deux scripts d'entraînement (Isolation
Forest filtre sur le Normal, Random Forest filtre sur les Attaques) — c'est
le MÊME train/test pour les deux modèles, comme dans le pipeline v1.

Les sous-ensembles de test PAR SOURCE (nécessaires étape 10 pour prouver la
généralisation séparément) se dérivent simplement en filtrant
combined_test.pkl sur la colonne `source` — pas besoin d'un split dédié.
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from commun import canonical_schema as cs

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

print("=" * 70)
print("COMBINAISON DES SOURCES + SPLIT STRATIFIE (source x label)")
print("=" * 70)

df_nf = pd.read_pickle(os.path.join(DATA_DIR, 'netflow_canonical.pkl'))
df_cic = pd.read_pickle(os.path.join(DATA_DIR, 'cicflowmeter_canonical.pkl'))
if 'jour' in df_cic.columns:
    df_cic = df_cic.drop(columns=['jour'])

df = pd.concat([df_nf, df_cic], ignore_index=True)
del df_nf, df_cic
print(f"\nCombiné : {len(df):,} lignes")
print(df.groupby(['source', 'Label_binaire']).size().unstack(fill_value=0))

strat_key = df['source'] + '_' + df['Label_binaire'].astype(str)
df_train, df_test = train_test_split(
    df, test_size=0.20, stratify=strat_key, random_state=42
)
del df

print(f"\nTrain : {len(df_train):,} lignes")
print(df_train.groupby(['source', 'Label_binaire']).size().unstack(fill_value=0))
print(f"\nTest  : {len(df_test):,} lignes")
print(df_test.groupby(['source', 'Label_binaire']).size().unstack(fill_value=0))

df_train.to_pickle(os.path.join(DATA_DIR, 'combined_train.pkl'))
df_test.to_pickle(os.path.join(DATA_DIR, 'combined_test.pkl'))
print(f"\n✓ Sauvegardé -> commun/data/combined_train.pkl, combined_test.pkl")
print("=" * 70)
