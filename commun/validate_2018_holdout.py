"""
Validation SANS FUITE de l'apport de CICIDS2018 à l'entraînement : reproduit
exactement la sélection d'entraînement de build_cicids2018_data.py (mêmes
sample(random_state=42)), puis construit un échantillon FRAIS à partir des
lignes RESTANTES (jamais utilisées à l'entraînement, random_state différent).

Répond à la question laissée ouverte par le recouvrement TEST4/train détecté
après coup (94% de recouvrement -> TEST4 n'est plus valide comme mesure de
l'apport de CICIDS2018).
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

MD4_DIR = os.path.abspath(os.path.dirname(__file__) + r'\..')
sys.path.insert(0, MD4_DIR)
from commun import canonical_schema as cs
from commun.adapters.cicflowmeter_adapter import CICFlowMeterAdapter
from commun.mapping_attaques import mapper

RAW_DIR = os.path.join(MD4_DIR, '..', 'TEST', 'test4', 'data', 'raw')
RENAME_2018_VERS_2017 = {
    'Dst Port': 'Destination Port', 'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
    'TotLen Bwd Pkts': 'Total Length of Bwd Packets', 'Tot Fwd Pkts': 'Total Fwd Packets',
    'Tot Bwd Pkts': 'Total Backward Packets', 'Pkt Len Max': 'Max Packet Length',
    'Pkt Len Min': 'Min Packet Length', 'FIN Flag Cnt': 'FIN Flag Count',
    'SYN Flag Cnt': 'SYN Flag Count', 'RST Flag Cnt': 'RST Flag Count',
    'PSH Flag Cnt': 'PSH Flag Count', 'ACK Flag Cnt': 'ACK Flag Count', 'URG Flag Cnt': 'URG Flag Count',
}
COLS_2018 = list(RENAME_2018_VERS_2017.keys()) + [
    'Protocol', 'Flow Duration', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Label',
]
LABELS_VALIDES = {'Benign', 'FTP-BruteForce', 'SSH-Bruteforce', 'DoS attacks-GoldenEye',
                   'DoS attacks-Slowloris', 'Bot', 'Infilteration'}
FICHIERS = [
    ('Wednesday-14-02-2018.csv', {'FTP-BruteForce': 50_000, 'SSH-Bruteforce': 50_000}, 40_000),
    ('Thursday-15-02-2018.csv', {'DoS attacks-GoldenEye': 40_000, 'DoS attacks-Slowloris': 10_990}, 40_000),
    ('Friday-02-03-2018.csv', {'Bot': 50_000}, 40_000),
    ('Thursday-01-03-2018.csv', {'Infilteration': 50_000}, 40_000),
]
CIBLE_FRAIS_PAR_CATEGORIE = 5_000

print("=" * 70)
print("VALIDATION SANS FUITE — apport CICIDS2018 (échantillon frais, disjoint du train)")
print("=" * 70)

adapter = CICFlowMeterAdapter()
frais = []

for fichier, cibles_attaque, cible_normal in FICHIERS:
    print(f"\n📂 {fichier}...")
    df_raw = pd.read_csv(os.path.join(RAW_DIR, fichier), usecols=lambda c: c.strip() in COLS_2018, low_memory=False)
    df_raw = df_raw.rename(columns=RENAME_2018_VERS_2017)
    df_raw = df_raw[df_raw['Label'].isin(LABELS_VALIDES)].reset_index(drop=True)
    df_canon = adapter.extract(df_raw)
    del df_raw
    cat_5 = pd.Series(
        np.where(df_canon['Label_binaire'] == 1, df_canon['Attack_brut'].apply(mapper), 'Benign'),
        index=df_canon.index,
    )
    df_canon['categorie_5'] = cat_5

    for label_brut, cible_n in cibles_attaque.items():
        mask = df_canon['Attack_brut'] == label_brut
        idx_train = set(df_canon[mask].sample(n=min(cible_n, int(mask.sum())), random_state=42).index)
        idx_reste = df_canon[mask].index.difference(idx_train)
        n_frais = min(CIBLE_FRAIS_PAR_CATEGORIE, len(idx_reste))
        if n_frais == 0:
            print(f"   ⚠️  {label_brut} : aucune ligne restante disjointe du train -> ignoré")
            continue
        echantillon = df_canon.loc[idx_reste].sample(n=n_frais, random_state=123)
        frais.append(echantillon)
        print(f"   ✓ {label_brut:<28s} : {len(idx_reste):,} lignes disjointes dispo -> {n_frais:,} retenues (frais)")

    mask_normal = df_canon['Label_binaire'] == 0
    idx_train_n = set(df_canon[mask_normal].sample(n=min(cible_normal, int(mask_normal.sum())), random_state=42).index)
    idx_reste_n = df_canon[mask_normal].index.difference(idx_train_n)
    n_frais_n = min(CIBLE_FRAIS_PAR_CATEGORIE, len(idx_reste_n))
    frais.append(df_canon.loc[idx_reste_n].sample(n=n_frais_n, random_state=123))
    print(f"   ✓ Benign : {len(idx_reste_n):,} lignes disjointes dispo -> {n_frais_n:,} retenues (frais)")
    del df_canon

df_frais = pd.concat(frais, ignore_index=True)
print(f"\n🔗 Échantillon frais total (garanti disjoint du train) : {len(df_frais):,} lignes")
print(f"   Normal={int((df_frais['Label_binaire']==0).sum()):,} | "
      f"Attaques={int((df_frais['Label_binaire']==1).sum()):,}")
print(df_frais.groupby('categorie_5').size())

# ============================================================
# EVALUATION — modeles deja entraines
# ============================================================
IF_DIR = os.path.join(MD4_DIR, 'hist_gradient_boosting', 'models')
RF_DIR = os.path.join(MD4_DIR, 'random_forest', 'models')

# Palier 1 supervisé (remplace l'ancien Isolation Forest cicflowmeter,
# jamais re-vérifié contre cet échantillon frais jusqu'ici -- contrairement
# à l'ancien IF, entraîné sur TOUT combined_train.pkl y compris le normal
# 2018 exclu spécifiquement pour l'ancien IF).
model_if = joblib.load(os.path.join(IF_DIR, 'model.pkl'))
seuil_if = joblib.load(os.path.join(IF_DIR, 'seuil_optimal.pkl'))['seuil_optimal']

X_if = df_frais[cs.ISOLATION_FOREST_FEATURES].values.astype(np.float32)
scores = model_if.predict_proba(X_if)[:, 1]
y_pred_if = (scores >= seuil_if).astype(int)
y_true = df_frais['Label_binaire'].values
tn, fp, fn, tp = confusion_matrix(y_true, y_pred_if, labels=[0, 1]).ravel()
fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else float('nan')
print(f"\n🌲 Palier 1 supervisé (cicflowmeter, modèle partagé) sur échantillon frais :")
print(f"   Acc={accuracy_score(y_true,y_pred_if)*100:.1f}% Prec={precision_score(y_true,y_pred_if,zero_division=0)*100:.1f}% "
      f"Rec={recall_score(y_true,y_pred_if,zero_division=0)*100:.1f}% F1={f1_score(y_true,y_pred_if,zero_division=0)*100:.1f}% "
      f"AUC={roc_auc_score(y_true,scores)*100:.1f}% FPR={fpr:.2f}%")
print(f"   TN={tn:,} FP={fp:,} FN={fn:,} TP={tp:,}")

model_rf = joblib.load(os.path.join(RF_DIR, 'model.pkl'))
le_rf = joblib.load(os.path.join(RF_DIR, 'label_encoder.pkl'))
features_rf = joblib.load(os.path.join(RF_DIR, 'features.pkl'))['features']

mask_att = df_frais['Label_binaire'] == 1
X_rf = df_frais.loc[mask_att, features_rf].values.astype(np.float32)
y_true_cat = le_rf.transform(df_frais.loc[mask_att, 'categorie_5'].values)
y_pred_cat = model_rf.predict(X_rf)

print(f"\n🌳 Random Forest sur échantillon frais ({int(mask_att.sum()):,} attaques) :")
print(f"   Accuracy={accuracy_score(y_true_cat,y_pred_cat)*100:.1f}% "
      f"F1 pondéré={f1_score(y_true_cat,y_pred_cat,average='weighted',zero_division=0)*100:.1f}%")
print(classification_report(y_true_cat, y_pred_cat, labels=range(len(le_rf.classes_)),
                             target_names=le_rf.classes_, zero_division=0))

print("=" * 70)
print("✅ TERMINÉ (résultat garanti sans fuite train/test)")
print("=" * 70)
