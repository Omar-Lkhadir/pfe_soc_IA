"""
Test zero-shot des modèles v2 (jamais entraînés sur du Zeek) sur le jeu
canonique Zeek réel (commun/data/zeek_canonical.pkl). Aucun entraînement ici.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from commun import canonical_schema as cs
from commun.mapping_attaques import mapper

MD4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
df = pd.read_pickle(os.path.join(MD4_DIR, 'commun', 'data', 'zeek_canonical.pkl'))

print("=" * 70)
print("TEST ZERO-SHOT — Modèles v2 sur Zeek/IoT-23 (jamais vu à l'entraînement)")
print("=" * 70)

y_bin = df['Label_binaire'].values

# --- Isolation Forest v2 ---
if_dir = os.path.join(MD4_DIR, 'isolation_forest', 'models')
model_if = joblib.load(os.path.join(if_dir, 'isolation_forest_model_v2_canonical.pkl'))
scaler_if = joblib.load(os.path.join(if_dir, 'scaler_v2_canonical.pkl'))
seuils_src = joblib.load(os.path.join(if_dir, 'seuil_optimal_v2_par_source.pkl'))
# Zeek jamais vu -> pas de seuil dédié ; on utilise le seuil cicflowmeter
# (le plus proche par nature : biflow riche, comme un fallback documenté)
seuil = seuils_src.get('cicflowmeter')

X_if = df[cs.TIER1_FEATURES].values.astype(np.float32)
scores = model_if.decision_function(scaler_if.transform(X_if))
pred = (scores < seuil).astype(int)

acc = accuracy_score(y_bin, pred) * 100
prec = precision_score(y_bin, pred, zero_division=0) * 100
rec = recall_score(y_bin, pred, zero_division=0) * 100
f1 = f1_score(y_bin, pred, zero_division=0) * 100
auc = roc_auc_score(y_bin, scores * -1) * 100
tn, fp, fn, tp = confusion_matrix(y_bin, pred, labels=[0, 1]).ravel()
print(f"\n🌲 Isolation Forest v2 (seuil cicflowmeter par défaut, Zeek jamais entraîné)")
print(f"   Acc={acc:.1f}%  Prec={prec:.1f}%  Rec={rec:.1f}%  F1={f1:.1f}%  AUC={auc:.1f}%")
print(f"   TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}  FPR={fp/(fp+tn)*100:.1f}%")

# --- Random Forest v2 (attaques uniquement) ---
rf_dir = os.path.join(MD4_DIR, 'random_forest', 'models')
model_rf = joblib.load(os.path.join(rf_dir, 'random_forest_model_v2_canonical.pkl'))
le = joblib.load(os.path.join(rf_dir, 'label_encoder_attacks_v2_canonical.pkl'))

mask = y_bin == 1
df_att = df[mask]
y_true_cat = df_att['Attack_brut'].apply(mapper).values
X_att = df_att[cs.TIER1_FEATURES + cs.TIER2_FEATURES].values.astype(np.float32)  # NaN tolere (HGB)
y_true_enc = le.transform(y_true_cat)
y_pred_enc = model_rf.predict(X_att)

acc_rf = accuracy_score(y_true_enc, y_pred_enc) * 100
f1_rf = f1_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0) * 100
print(f"\n🌳 Random Forest v2 ({len(X_att):,} attaques Zeek, jamais entraîné)")
print(f"   Accuracy={acc_rf:.1f}%  F1 pondéré={f1_rf:.1f}%")
print(f"   Catégories vraies présentes : {sorted(set(y_true_cat))}")
cm = confusion_matrix(y_true_enc, y_pred_enc, labels=range(len(le.classes_)))
for i, row in enumerate(cm):
    if row.sum() > 0:
        print(f"   {le.classes_[i]:<22s} : {row.tolist()}  (classes={list(le.classes_)})")

with open(os.path.join(MD4_DIR, 'commun', 'rapport_test_zeek_zeroshot.txt'), 'w', encoding='utf-8') as f:
    f.write("TEST ZERO-SHOT — Modeles v2 (NetFlow+CICFlowMeter) sur Zeek/IoT-23 reel\n")
    f.write("="*70 + "\n\n")
    f.write(f"Isolation Forest v2 (seuil cicflowmeter, pas de seuil zeek dedie) :\n")
    f.write(f"  Acc={acc:.1f}% Prec={prec:.1f}% Rec={rec:.1f}% F1={f1:.1f}% AUC={auc:.1f}% FPR={fp/(fp+tn)*100:.1f}%\n\n")
    f.write(f"Random Forest v2 ({len(X_att):,} attaques) :\n")
    f.write(f"  Accuracy={acc_rf:.1f}% F1 pondere={f1_rf:.1f}%\n")

print(f"\n✓ commun/rapport_test_zeek_zeroshot.txt")
