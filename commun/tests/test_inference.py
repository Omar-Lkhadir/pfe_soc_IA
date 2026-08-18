"""
Validation de commun/inference.py (NIDSPredictor) contre des données réelles
déjà connues, pour vérifier que le routage/l'imputation/le fallback se
comportent comme prévu et donnent des résultats coherents avec les scripts
de test autonomes (TEST/test2, TEST/test5).
"""

import os
import sys
import pandas as pd

MD4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, MD4_DIR)
from commun.inference import NIDSPredictor, detect_source

print("=" * 70)
print("VALIDATION — commun/inference.py (NIDSPredictor)")
print("=" * 70)

predictor = NIDSPredictor()
print(f"\n✓ Modèles chargés. Sources IF disponibles : {list(predictor.if_models.keys())}")

# ------------------------------------------------------------
# 1. Détection de source par event.module explicite
# ------------------------------------------------------------
print(f"\n🔎 [1] Détection de source (event.module explicite)...")
for module in ['netflow', 'zeek', 'cicflowmeter', 'inconnu']:
    src = detect_source({'event.module': module, 'x': 1})
    print(f"   event.module='{module}' -> {src}")

# ------------------------------------------------------------
# 2. Détection de source par signature de champs (sans event.module)
# ------------------------------------------------------------
print(f"\n🔎 [2] Détection de source (signature de champs)...")
print("   NetFlow-style :", detect_source({'L4_DST_PORT': 80, 'IN_BYTES': 100, 'PROTOCOL': 6}))
print("   Zeek-style    :", detect_source({'id.orig_p': 80, 'id.resp_p': 443, 'proto': 'tcp'}))
print("   Inconnu       :", detect_source({'foo': 1, 'bar': 2}))

# ------------------------------------------------------------
# 3. predict_batch sur un échantillon réel NetFlow (TEST 5)
# ------------------------------------------------------------
print(f"\n📂 [3] Batch réel — extrait NetFlow (BigFlow-NIDS-V2, TEST 5)...")
raw = pd.read_csv(
    os.path.join(MD4_DIR, '..', 'TEST', 'test5', 'data', 'raw', 'BigFlow-NIDS-V2_last4M.csv'),
    nrows=5000,
)
res = predictor.predict_batch(raw, source='netflow')
print(res['ml_status'].value_counts())
print(f"   is_attack : {res['is_attack'].value_counts(dropna=False).to_dict()}")
verite = (raw['Label'] == 1).values
predite = res['is_attack'].infer_objects(copy=False).fillna(False).values
accord = (verite == predite).mean() * 100
print(f"   Accord avec la vérité terrain (Label) sur cet échantillon : {accord:.1f}%")

# ------------------------------------------------------------
# 4. Format non reconnu : événement sans aucune signature connue
# ------------------------------------------------------------
print(f"\n🔎 [4] Événement de format inconnu (aucune signature ne correspond)...")
res_inconnu = predictor.predict({'champ_bizarre': 1, 'autre_champ': 'x'})
print(f"   {res_inconnu}")
assert res_inconnu['ml_status'] == 'format_non_reconnu', "devrait être format_non_reconnu"
print("   ✓ Correctement marqué format_non_reconnu, pas de scoring forcé.")

# ------------------------------------------------------------
# 5. Imputation dégradée : événement NetFlow avec un champ palier 1 manquant
# ------------------------------------------------------------
print(f"\n🔎 [5] Événement NetFlow avec un champ palier 1 manquant (IN_BYTES absent)...")
event_partiel = raw.iloc[0].to_dict()
del event_partiel['IN_BYTES']
res_partiel = predictor.predict(event_partiel, source='netflow')
print(f"   ml_status = {res_partiel['ml_status']} (attendu: scoring_degrade)")
assert res_partiel['ml_status'] == 'scoring_degrade'
print("   ✓ Imputation dégradée déclenchée comme prévu.")

print(f"\n{'='*70}\n✅ TOUS LES CONTROLES SONT PASSES\n{'='*70}")
