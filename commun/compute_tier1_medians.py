"""
Calcule la médiane de chaque feature du palier 1, par source, à partir des
données canoniques d'entraînement. Utilisé par commun/inference.py pour
l'imputation dégradée (un événement reconnu mais avec un champ du palier 1
manquant est complété par la médiane de SA PROPRE source — jamais une
constante fabriquée entre sources, contrairement à l'imputation qui avait
causé l'effondrement de performance du modèle v1/v2 sur des formats croisés).

Sortie : hist_gradient_boosting/models/medianes_palier1/{source}.pkl
"""

import os
import sys
import joblib
import pandas as pd

MD4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, MD4_DIR)
from commun import canonical_schema as cs

DATA_DIR = os.path.join(MD4_DIR, 'commun', 'data')
MODELS_DIR = os.path.join(MD4_DIR, 'hist_gradient_boosting', 'models')

FICHIERS_SOURCE = {
    'netflow': 'netflow_canonical.pkl',
    'cicflowmeter': 'cicflowmeter_canonical.pkl',
    'zeek': 'zeek_canonical.pkl',
}

print("=" * 70)
print("CALCUL DES MEDIANES PALIER 1 PAR SOURCE (pour imputation dégradée)")
print("=" * 70)

for source, fichier in FICHIERS_SOURCE.items():
    chemin = os.path.join(DATA_DIR, fichier)
    if not os.path.exists(chemin):
        print(f"   ⚠️  {fichier} introuvable -> {source} ignorée")
        continue
    df = pd.read_pickle(chemin)[cs.TIER1_FEATURES]
    medianes = df.median().to_dict()
    out_dir = os.path.join(MODELS_DIR, 'medianes_palier1')
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(medianes, os.path.join(out_dir, f'{source}.pkl'))
    print(f"   ✓ {source:<14s} -> {out_dir}\\{source}.pkl")
    del df

print("\n" + "=" * 70)
print("✅ TERMINÉ")
print("=" * 70)
