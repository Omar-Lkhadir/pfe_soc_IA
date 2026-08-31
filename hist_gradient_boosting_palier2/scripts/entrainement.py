"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODELE SOC - PALIER 2 — SOCLE CANONIQUE MULTI-SOURCE                       ║
║  PFE IT6 - SOC Intelligent basé sur ELK + IA                                  ║
║                                                                                ║
║  Catégorise en 5 classes les événements que le palier 1                     ║
║  (hist_gradient_boosting/) a détectés comme attaque. Entraîné sur NetFlow   ║
║  (BigFlow-NIDS-V2) + CICFlowMeter (CICIDS2017 + 2018) + Zeek/IoT-23         ║
║  combinés                                                                    ║
║  (commun/combine_and_split.py), un seul modèle partagé entre les 3          ║
║  sources -- n'entraîne que sur des attaques (Label_binaire==1), jamais      ║
║  affecté par le déséquilibre de volume "normal" entre sources qui a         ║
║  nécessité une pondération par source côté palier 1.                        ║
║  Utilise PALIER 1 + PALIER 2 du schéma canonique (22 features,              ║
║  PALIER2_FEATURES de commun/canonical_schema.py). Le palier 2               ║
║  (taille de paquet, IAT) contient des NaN pour les sources qui ne           ║
║  l'exposent pas -> HistGradientBoostingClassifier (gère nativement les      ║
║  NaN, contrairement à un RandomForestClassifier classique) : jamais         ║
║  besoin de fabriquer une constante d'imputation. Pas de                     ║
║  StandardScaler nécessaire (arbres à base d'histogrammes, invariants aux    ║
║  échelles).                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import joblib
import os
import sys
import time
import gc
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) or '.'
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MD4_DIR = os.path.join(SCRIPT_DIR, '..', '..')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, MD4_DIR)

from commun import canonical_schema as cs
from commun.mapping_attaques import TYPES_ATTAQUES, mapper

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
np.random.seed(42)

print("=" * 70)
print("MODELE SOC - RANDOM FOREST (SOCLE CANONIQUE MULTI-SOURCE)")
print("=" * 70)

DATA_DIR = os.path.join(MD4_DIR, 'commun', 'data')
start_total = time.time()

# ============================================================
# ÉTAPE 1 : CHARGEMENT (attaques uniquement)
# ============================================================
print(f"\n📂 [ÉTAPE 1] Chargement du jeu combiné (attaques uniquement)...")

df_train = pd.read_pickle(os.path.join(DATA_DIR, 'combined_train.pkl'))
df_test = pd.read_pickle(os.path.join(DATA_DIR, 'combined_test.pkl'))
df_train = df_train[df_train.Label_binaire == 1].reset_index(drop=True)
df_test = df_test[df_test.Label_binaire == 1].reset_index(drop=True)

feature_names = list(cs.PALIER2_FEATURES)
X_train = df_train[feature_names].values.astype(np.float32)
X_test = df_test[feature_names].values.astype(np.float32)
source_train = df_train['source'].values
source_test = df_test['source'].values
y_train_types = df_train['Attack_brut'].values
y_test_types = df_test['Attack_brut'].values
del df_train, df_test; gc.collect()

print(f"   Train : {len(X_train):,} attaques | Test : {len(X_test):,} attaques")
print(f"   Features ({len(feature_names)}, palier 1+2, NaN toléré) : {feature_names}")
print(f"   NaN dans X_train : {np.isnan(X_train).sum():,} ({100*np.isnan(X_train).mean():.2f}%)")

for nom, arr in [('X_train', X_train), ('X_test', X_test)]:
    n_inf = np.isinf(arr).sum()
    if n_inf > 0:
        print(f"   ⚠️  {nom}: {n_inf:,} infinis → corrigés (NaN, laissé au modèle)")
        arr[np.isinf(arr)] = np.nan

# ============================================================
# ÉTAPE 1bis : CONTRÔLE ANTI-FUITE (chevauchement train/test)
# ============================================================
print(f"\n🔎 [ÉTAPE 1bis] Contrôle anti-fuite (chevauchement train/test)...")


def _hash_rows(arr):
    a = np.nan_to_num(arr, nan=-999999.0)
    contig = np.ascontiguousarray(a)
    contig.flags.writeable = False
    return set(contig.view([('', contig.dtype)] * contig.shape[1]).reshape(-1))


chevauchement = len(_hash_rows(X_train) & _hash_rows(X_test))
print(f"   {'✓ Aucun chevauchement' if chevauchement == 0 else f'⚠️  {chevauchement:,} ligne(s) en commun'}")

# ============================================================
# ÉTAPE 2 : MAPPING DES TYPES (module partagé commun/mapping_attaques.py)
# ============================================================
print(f"\n🏷️  [ÉTAPE 2] Mapping des types...")

y_train = np.array([mapper(t) for t in y_train_types])
y_test = np.array([mapper(t) for t in y_test_types])

print(f"   Distribution (train) :")
for t in TYPES_ATTAQUES:
    n_nf = int(((y_train == t) & (source_train == 'netflow')).sum())
    n_cic = int(((y_train == t) & (source_train == 'cicflowmeter')).sum())
    print(f"   {t:<22s} netflow={n_nf:>9,}  cicflowmeter={n_cic:>9,}  total={n_nf+n_cic:>9,}")

# ============================================================
# ÉTAPE 3 : ENCODAGE (pas de normalisation : HistGradientBoosting est
# invariant à l'échelle des features, StandardScaler inutile)
# ============================================================
print(f"\n🔤 [ÉTAPE 3] Encodage des catégories...")

le = LabelEncoder()
le.fit(TYPES_ATTAQUES)
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

for i, c in enumerate(le.classes_):
    print(f"   {c:<22s} : {(y_train_enc==i).sum():>10,}")

# ============================================================
# ÉTAPE 4 : RECHERCHE D'HYPERPARAMÈTRES
# ============================================================
print(f"\n🔍 [ÉTAPE 4] Recherche d'hyperparamètres (RandomizedSearchCV, cv=2)...")

n_recherche = min(300_000, len(X_train))
if n_recherche < len(X_train):
    X_search, _, y_search, _ = train_test_split(
        X_train, y_train_enc, train_size=n_recherche, stratify=y_train_enc, random_state=42
    )
else:
    X_search, y_search = X_train, y_train_enc

grille = {
    'max_iter': [150, 250],
    'max_depth': [None, 10, 15],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_leaf_nodes': [31, 63],
}

recherche = RandomizedSearchCV(
    HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
    param_distributions=grille, n_iter=6, cv=2, scoring='f1_weighted',
    random_state=42, n_jobs=-1, verbose=2,
)

t0 = time.time()
recherche.fit(X_search, y_search)
print(f"   ✓ Recherche terminée en {time.time()-t0:.0f}s "
      f"(sur {n_recherche:,} lignes, F1 CV={recherche.best_score_:.4f})")
print(f"   ✓ Meilleurs paramètres : {recherche.best_params_}")

del X_search, y_search; gc.collect()

# ============================================================
# ÉTAPE 5 : ENTRAÎNEMENT FINAL SUR TOUT LE TRAIN
# ============================================================
print(f"\n🚀 [ÉTAPE 5] Entraînement final sur {len(X_train):,} attaques...")

model = HistGradientBoostingClassifier(
    **recherche.best_params_, class_weight='balanced', random_state=42,
)
t0 = time.time()
model.fit(X_train, y_train_enc)
print(f"   ✓ Entraîné en {time.time()-t0:.0f}s")

# ============================================================
# ÉTAPE 6 : ÉVALUATION — POOLÉE PUIS PAR SOURCE
# ============================================================
print(f"\n📊 [ÉTAPE 6] Évaluation sur le test...")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)
confiance = y_proba.max(axis=1)


def eval_bloc(nom, y_true_enc, y_pred_enc):
    acc = accuracy_score(y_true_enc, y_pred_enc) * 100
    prec = precision_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0) * 100
    rec = recall_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0) * 100
    f1 = f1_score(y_true_enc, y_pred_enc, average='weighted', zero_division=0) * 100
    print(f"   [{nom:<20}] n={len(y_true_enc):>8,}  Acc={acc:5.1f}%  Prec={prec:5.1f}%  "
          f"Rec={rec:5.1f}%  F1={f1:5.1f}%")
    return dict(nom=nom, n=len(y_true_enc), acc=acc, prec=prec, rec=rec, f1=f1)


resultats = [eval_bloc('POOLE', y_test_enc, y_pred)]
for src in np.unique(source_test):
    m = source_test == src
    resultats.append(eval_bloc(src.upper(), y_test_enc[m], y_pred[m]))

r_poole = resultats[0]
acc, prec, rec, f1 = r_poole['acc'], r_poole['prec'], r_poole['rec'], r_poole['f1']
cm = confusion_matrix(y_test_enc, y_pred, labels=range(len(le.classes_)))

print(f"\n   Rapport de classification (poolé) :\n")
rapport_cls = classification_report(y_test_enc, y_pred, labels=range(len(le.classes_)),
                                     target_names=le.classes_, zero_division=0)
print(rapport_cls)

print(f"   [ANALYSE DE CONFIANCE]")
for seuil_test in [0.5, 0.6, 0.7, 0.8, 0.9]:
    mask_c = confiance >= seuil_test
    if mask_c.sum() > 0:
        acc_c = accuracy_score(y_test_enc[mask_c], y_pred[mask_c]) * 100
        pct_couvert = mask_c.sum() / len(y_test_enc) * 100
        print(f"      Seuil >= {seuil_test:.1f} : {pct_couvert:>5.1f}% des cas couverts, accuracy = {acc_c:.1f}%")

SEUIL_CONFIANCE = 0.7
mask_confiant = confiance >= SEUIL_CONFIANCE
acc_confiant = accuracy_score(y_test_enc[mask_confiant], y_pred[mask_confiant]) * 100 if mask_confiant.sum() > 0 else float('nan')
print(f"\n   ✓ Seuil retenu = {SEUIL_CONFIANCE} : {mask_confiant.sum():,}/{len(y_test_enc):,} "
      f"({mask_confiant.sum()/len(y_test_enc)*100:.1f}%) automatiques, accuracy = {acc_confiant:.1f}%")

# ============================================================
# ÉTAPE 7 : SAUVEGARDE
# ============================================================
print(f"\n💾 [ÉTAPE 7] Sauvegarde dans {MODELS_DIR}...")

joblib.dump(model, os.path.join(MODELS_DIR, 'model.pkl'))
joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
joblib.dump({'seuil_confiance': SEUIL_CONFIANCE}, os.path.join(MODELS_DIR, 'seuil_confiance.pkl'))
joblib.dump({'features': feature_names}, os.path.join(MODELS_DIR, 'features.pkl'))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f'Random Forest (canonique) | Accuracy={acc:.1f}% | F1={f1:.1f}%', fontsize=14, fontweight='bold')
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=le.classes_, yticklabels=le.classes_,
            ax=axes[0], cbar=False)
axes[0].set_xlabel('Prédit'); axes[0].set_ylabel('Réel'); axes[0].set_title('Matrice de Confusion (test poolé)')
plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right')

x = np.arange(len(le.classes_)); w = 0.25
for i, c in enumerate(le.classes_):
    mask = y_test_enc == i
    if mask.sum() > 0:
        p = precision_score(y_test_enc == i, y_pred == i, zero_division=0) * 100
        r = recall_score(y_test_enc == i, y_pred == i, zero_division=0) * 100
        fsc = f1_score(y_test_enc == i, y_pred == i, zero_division=0) * 100
        axes[1].bar(i - w, p, w, color='#3498db')
        axes[1].bar(i, r, w, color='#2ecc71')
        axes[1].bar(i + w, fsc, w, color='#f39c12')
axes[1].bar(0, 0, w, label='Precision', color='#3498db')
axes[1].bar(0, 0, w, label='Recall', color='#2ecc71')
axes[1].bar(0, 0, w, label='F1', color='#f39c12')
axes[1].set_xticks(x); axes[1].set_xticklabels(le.classes_, rotation=30, ha='right')
axes[1].set_ylabel('%'); axes[1].set_title('Métriques par Classe (test poolé)')
axes[1].legend(fontsize=8); axes[1].set_ylim(0, 115)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'evaluation.png'), dpi=150, bbox_inches='tight')
plt.close()

fig2, ax2 = plt.subplots(figsize=(10, 6))
noms = [r['nom'] for r in resultats]
xr = np.arange(len(noms)); wr = 0.2
for i, (metrique, couleur) in enumerate(zip(['acc', 'prec', 'rec', 'f1'],
                                              ['#2ecc71', '#3498db', '#f39c12', '#9b59b6'])):
    vals = [r[metrique] for r in resultats]
    ax2.bar(xr + (i - 1.5) * wr, vals, wr, label=metrique.upper(), color=couleur)
ax2.set_xticks(xr); ax2.set_xticklabels(noms)
ax2.set_ylabel('%'); ax2.set_ylim(0, 115)
ax2.set_title('Random Forest — métriques poolé vs par source')
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'evaluation_par_source.png'), dpi=150, bbox_inches='tight')
plt.close()

y_test_labels = le.inverse_transform(y_test_enc)
y_pred_labels = le.inverse_transform(y_pred)
counts_verite = pd.Series(y_test_labels).value_counts().reindex(TYPES_ATTAQUES, fill_value=0)
counts_predit = pd.Series(y_pred_labels).value_counts().reindex(TYPES_ATTAQUES, fill_value=0)
fig3, ax3 = plt.subplots(figsize=(12, 6))
xb = np.arange(5); wb = 0.35
ax3.bar(xb - wb/2, counts_verite.values, wb, label='Vérité terrain', color='#3498db', edgecolor='white')
ax3.bar(xb + wb/2, counts_predit.values, wb, label='Prédit par le modèle', color='#e74c3c', edgecolor='white')
for i, (v, p) in enumerate(zip(counts_verite.values, counts_predit.values)):
    ax3.text(i - wb/2, v, f'{v:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax3.text(i + wb/2, p, f'{p:,}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax3.set_xticks(xb); ax3.set_xticklabels(TYPES_ATTAQUES, rotation=20, ha='right')
ax3.set_ylabel("Nombre d'attaques"); ax3.set_title('Vérité Terrain vs Prédictions — comptage par catégorie', fontweight='bold')
ax3.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'verite_vs_prediction.png'), dpi=150, bbox_inches='tight')
plt.close()

with open(os.path.join(RESULTS_DIR, 'rapport.txt'), 'w', encoding='utf-8') as f:
    f.write("RAPPORT RANDOM FOREST (SOCLE CANONIQUE MULTI-SOURCE, HistGradientBoosting)\n" + "=" * 60 + "\n\n")
    f.write(f"features ({len(feature_names)}, palier 1+2) : {feature_names}\n")
    f.write(f"Meilleurs hyperparamètres : {recherche.best_params_}\n\n")
    f.write("Résultats par bloc (poolé + par source) :\n")
    for r in resultats:
        f.write(f"  [{r['nom']:<20}] n={r['n']:,}  Acc={r['acc']:.1f}%  Prec={r['prec']:.1f}%  "
                f"Rec={r['rec']:.1f}%  F1={r['f1']:.1f}%\n")
    f.write(f"\nSeuil de confiance = {SEUIL_CONFIANCE} : {mask_confiant.sum():,}/{len(y_test_enc):,} "
            f"automatiques, accuracy = {acc_confiant:.1f}%\n\n")
    f.write("Rapport de classification détaillé (poolé) :\n")
    f.write(rapport_cls)

print(f"\n   ✓ model.pkl, label_encoder.pkl, seuil_confiance.pkl, features.pkl")
print(f"   ✓ evaluation.png, evaluation_par_source.png, "
      f"verite_vs_prediction.png, rapport.txt")

print(f"\n{'='*70}")
print(f"✅ TERMINÉ EN {time.time()-start_total:.0f}s | Accuracy(poolé)={acc:.1f}% | F1(poolé)={f1:.1f}%")
print("=" * 70)
