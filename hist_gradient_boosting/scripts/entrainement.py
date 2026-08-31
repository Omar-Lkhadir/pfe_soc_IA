"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  MODELE SOC - PALIER 1 — HistGradientBoostingClassifier                     ║
║  PFE IT6 - SOC Intelligent basé sur ELK + IA                                  ║
║                                                                                ║
║  Un ancien Isolation Forest non-supervisé (3 modèles, un par source)        ║
║  plafonnait structurellement à 60-85% d'AUC car il n'utilisait jamais       ║
║  Label_binaire : il devinait "normal" par densité statistique, pas par      ║
║  apprentissage de ce qui distingue réellement une attaque. Retiré du        ║
║  projet (historique dans git) une fois ce script confirmé nettement         ║
║  meilleur (96-99,9% sur les mêmes 17 features Tier1).                       ║
║                                                                                ║
║  Ce script entraîne le palier 1 comme un problème supervisé : même          ║
║  HistGradientBoostingClassifier que le palier 2                             ║
║  (hist_gradient_boosting_palier2/), MÊMES 17 features Tier1                 ║
║  (PALIER1_FEATURES, zéro nouvelle                                           ║
║  feature), cible = Label_binaire au lieu de la catégorie d'attaque. UN      ║
║  SEUL modèle partagé entre les 3 sources (comme le palier 2, jamais         ║
║  besoin d'un modèle par source puisqu'il apprend "qu'est-ce qui             ║
║  distingue une attaque" plutôt que "à quoi ressemble LE normal de CETTE     ║
║  source"). Pondéré par source pendant l'entraînement (poids inversement     ║
║  proportionnel au volume brut) pour éviter que cicflowmeter/netflow,       ║
║  bien plus volumineux, n'écrasent la frontière apprise pour zeek.           ║
║                                                                                ║
║  Compromis assumé : perd la capacité de détection d'anomalie "zero-day"     ║
║  (un pattern qui ne ressemble à aucune attaque connue sera classé Normal    ║
║  au lieu d'être flaggé suspect). Décision utilisateur explicite : accepter  ║
║  ce compromis pour de meilleures performances mesurées sur trafic connu.    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
import joblib
import os
import sys
import time
import gc
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 120
np.random.seed(42)

print("=" * 70)
print("MODELE SOC - PALIER 1 SUPERVISÉ (HistGradientBoostingClassifier)")
print("=" * 70)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) or '.'
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MD4_DIR = os.path.join(SCRIPT_DIR, '..', '..')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
sys.path.insert(0, MD4_DIR)

from commun import canonical_schema as cs

DATA_DIR = os.path.join(MD4_DIR, 'commun', 'data')
FEATURES = list(cs.PALIER1_FEATURES)
start_total = time.time()

# ============================================================
# ÉTAPE 1 : CHARGEMENT (toutes les lignes, normal + attaques, 3 sources)
# ============================================================
print(f"\n📂 [ÉTAPE 1] Chargement du jeu combiné (NetFlow + CICFlowMeter + Zeek)...")

df_train = pd.read_pickle(os.path.join(DATA_DIR, 'combined_train.pkl'))
df_test = pd.read_pickle(os.path.join(DATA_DIR, 'combined_test.pkl'))

print(f"   Train : {len(df_train):,} lignes | Test : {len(df_test):,} lignes")
print(df_train.groupby(['source', 'Label_binaire']).size().unstack(fill_value=0))

X_all_train = df_train[FEATURES].values.astype(np.float32)
y_all_train = df_train['Label_binaire'].values.astype(np.int8)
source_all_train = df_train['source'].values
X_test = df_test[FEATURES].values.astype(np.float32)
y_test = df_test['Label_binaire'].values.astype(np.int8)
source_test = df_test['source'].values
X_all_train[np.isinf(X_all_train)] = 0
X_test[np.isinf(X_test)] = 0
del df_train, df_test; gc.collect()

# ============================================================
# ÉTAPE 2 : SPLIT TRAIN_SUB / VALIDATION (stratifié source+label, comme
# commun/combine_and_split.py, pour calibrer le seuil de décision)
# ============================================================
print(f"\n✂️  [ÉTAPE 2] Split train_sub/validation (85/15, stratifié source+label)...")

strat_key = pd.Series(source_all_train).astype(str) + '_' + pd.Series(y_all_train).astype(str)
X_train_sub, X_val, y_train_sub, y_val, src_train_sub, src_val = train_test_split(
    X_all_train, y_all_train, source_all_train, test_size=0.15, stratify=strat_key, random_state=42
)
del X_all_train, y_all_train, source_all_train; gc.collect()

print(f"   Train_sub : {len(X_train_sub):,} | Validation : {len(X_val):,}")

# ============================================================
# ÉTAPE 2bis : PONDÉRATION PAR SOURCE (en plus de class_weight='balanced',
# qui équilibre normal/attaque mais PAS le volume entre sources -- zeek
# (3,6k normal) est noyé par cicflowmeter (1,9M normal)/netflow (459k) en
# comptage brut. poids_source = n_total / (n_sources * n_cette_source),
# même formule que class_weight='balanced' mais appliquée à la source au
# lieu du label -> zeek pèse proportionnellement autant que les autres
# sources dans la fonction de perte, sans dupliquer ni ajouter de donnée.
# ============================================================
print(f"\n⚖️  [ÉTAPE 2bis] Pondération par source (corrige le déséquilibre de volume)...")

sources_u, comptes_u = np.unique(src_train_sub, return_counts=True)
poids_source = {s: len(src_train_sub) / (len(sources_u) * c) for s, c in zip(sources_u, comptes_u)}
for s, c in zip(sources_u, comptes_u):
    print(f"   {s:<14s} n={c:>10,}  poids={poids_source[s]:.3f}")
sample_weight_sub = np.array([poids_source[s] for s in src_train_sub], dtype=np.float64)

# ============================================================
# ÉTAPE 3 : RECHERCHE D'HYPERPARAMÈTRES
# ============================================================
print(f"\n🔍 [ÉTAPE 3] Recherche d'hyperparamètres (RandomizedSearchCV, cv=2)...")

n_recherche = min(300_000, len(X_train_sub))
if n_recherche < len(X_train_sub):
    idx_recherche, _ = train_test_split(
        np.arange(len(X_train_sub)), train_size=n_recherche, stratify=y_train_sub, random_state=42
    )
    X_search, y_search, w_search = X_train_sub[idx_recherche], y_train_sub[idx_recherche], sample_weight_sub[idx_recherche]
else:
    X_search, y_search, w_search = X_train_sub, y_train_sub, sample_weight_sub

grille = {
    'max_iter': [150, 250],
    'max_depth': [None, 10, 15],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_leaf_nodes': [31, 63],
}

recherche = RandomizedSearchCV(
    HistGradientBoostingClassifier(class_weight='balanced', random_state=42),
    param_distributions=grille, n_iter=6, cv=2, scoring='roc_auc',
    random_state=42, n_jobs=-1, verbose=2,
)

t0 = time.time()
recherche.fit(X_search, y_search, sample_weight=w_search)
print(f"   ✓ Recherche terminée en {time.time()-t0:.0f}s "
      f"(sur {n_recherche:,} lignes, AUC CV={recherche.best_score_:.4f})")
print(f"   ✓ Meilleurs paramètres : {recherche.best_params_}")

del X_search, y_search, w_search; gc.collect()

# ============================================================
# ÉTAPE 4 : ENTRAÎNEMENT FINAL SUR TRAIN_SUB
# ============================================================
print(f"\n🚀 [ÉTAPE 4] Entraînement final sur {len(X_train_sub):,} lignes...")

model = HistGradientBoostingClassifier(
    **recherche.best_params_, class_weight='balanced', random_state=42,
)
t0 = time.time()
model.fit(X_train_sub, y_train_sub, sample_weight=sample_weight_sub)
print(f"   ✓ Entraîné en {time.time()-t0:.0f}s")

# ============================================================
# ÉTAPE 5 : SEUIL OPTIMAL SUR VALIDATION (F1-max, comme IF)
# ============================================================
print(f"\n🎯 [ÉTAPE 5] Calibration du seuil de décision sur validation...")

proba_val = model.predict_proba(X_val)[:, 1]
seuils = np.arange(0.01, 1.00, 0.01)
best_seuil, best_f1v = 0.5, 0.0
for s in seuils:
    f1c = f1_score(y_val, (proba_val >= s).astype(int), zero_division=0)
    if f1c > best_f1v:
        best_f1v, best_seuil = f1c, s
print(f"   ✓ Seuil optimal (F1 validation={best_f1v*100:.1f}%) : {best_seuil:.2f}")

# ============================================================
# ÉTAPE 6 : ÉVALUATION FINALE — POOLÉE PUIS PAR SOURCE (test jamais touché)
# ============================================================
print(f"\n📊 [ÉTAPE 6] Évaluation sur le test...")

proba_test = model.predict_proba(X_test)[:, 1]
y_pred = (proba_test >= best_seuil).astype(int)


def eval_bloc(nom, y_true, y_pred_):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred_) * 100
    prec = precision_score(y_true, y_pred_, zero_division=0) * 100
    rec = recall_score(y_true, y_pred_, zero_division=0) * 100
    f1 = f1_score(y_true, y_pred_, zero_division=0) * 100
    fpr = fp / (fp + tn) * 100 if (fp + tn) > 0 else float('nan')
    print(f"   [{nom:<20}] n={len(y_true):>9,}  Acc={acc:5.1f}%  Prec={prec:5.1f}%  "
          f"Rec={rec:5.1f}%  F1={f1:5.1f}%  FPR={fpr:5.1f}%")
    return dict(nom=nom, n=len(y_true), acc=acc, prec=prec, rec=rec, f1=f1, fpr=fpr,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


resultats = []
auc_poole = roc_auc_score(y_test, proba_test) * 100 if len(np.unique(y_test)) > 1 else float('nan')
r = eval_bloc('POOLE', y_test, y_pred)
r['auc'] = auc_poole
resultats.append(r)
print(f"   [{'POOLE':<20}] AUC={auc_poole:.1f}%")

for src in np.unique(source_test):
    m = source_test == src
    r = eval_bloc(src.upper(), y_test[m], y_pred[m])
    r['auc'] = roc_auc_score(y_test[m], proba_test[m]) * 100 if len(np.unique(y_test[m])) > 1 else float('nan')
    resultats.append(r)

r_poole = resultats[0]
tn, fp, fn, tp = r_poole['tn'], r_poole['fp'], r_poole['fn'], r_poole['tp']
acc, prec, rec, f1, fpr, auc = r_poole['acc'], r_poole['prec'], r_poole['rec'], r_poole['f1'], r_poole['fpr'], auc_poole

# ============================================================
# ÉTAPE 7 : SAUVEGARDE (nouveaux fichiers, l'ancien IF n'est PAS touché)
# ============================================================
print(f"\n💾 [ÉTAPE 7] Sauvegarde dans {MODELS_DIR}...")

joblib.dump(model, os.path.join(MODELS_DIR, 'model.pkl'))
# Seuil GLOBAL unique (pas par source) : une recalibration par source a été
# testée et abandonnée (pas de bénéfice réel sur données réellement
# nouvelles -- le pool de validation par source reflète la distribution
# d'entraînement, pas les captures externes) ; seul le seuil global est
# utilisé en production.
joblib.dump({'seuil_optimal': float(best_seuil)}, os.path.join(MODELS_DIR, 'seuil_optimal.pkl'))
joblib.dump({'features': FEATURES}, os.path.join(MODELS_DIR, 'features.pkl'))

# ============================================================
# GRAPHIQUE + RAPPORT (comparaison face à l'ancien IF, rapport.txt existant)
# ============================================================
print(f"\n📈 Graphiques...")

ancien_if_reference = {
    'NETFLOW': dict(acc=82.5, prec=89.0, rec=63.1, f1=73.8, auc=83.1, fpr=5.0),
    'CICFLOWMETER': dict(acc=77.9, prec=54.7, rec=67.4, f1=60.4, auc=79.6, fpr=18.6),
    'ZEEK': dict(acc=99.5, prec=99.5, rec=100.0, f1=99.7, auc=92.6, fpr=17.1),
}

sources_ok = [r['nom'] for r in resultats if r['nom'] != 'POOLE']
fig, axes = plt.subplots(1, len(sources_ok), figsize=(6 * len(sources_ok), 6))
if len(sources_ok) == 1:
    axes = [axes]
fig.suptitle("Palier 1 — Isolation Forest (non-supervisé) vs nouveau modèle supervisé",
             fontsize=14, fontweight='bold')

for ax, src in zip(axes, sources_ok):
    r_new = next(r for r in resultats if r['nom'] == src)
    r_old = ancien_if_reference.get(src)
    labels_m = ['Prec', 'Rec', 'F1', 'AUC']
    x = np.arange(len(labels_m))
    w = 0.35
    if r_old:
        vp = [r_old['prec'], r_old['rec'], r_old['f1'], r_old['auc']]
        ax.bar(x - w/2, vp, w, label='IF (ancien)', color='#95a5a6')
        for i, a in enumerate(vp):
            ax.text(i - w/2, a + 1, f'{a:.0f}', ha='center', fontsize=9)
    vd = [r_new['prec'], r_new['rec'], r_new['f1'], r_new['auc']]
    ax.bar(x + w/2, vd, w, label='Supervisé (nouveau)', color='#2ecc71')
    for i, b in enumerate(vd):
        ax.text(i + w/2, b + 1, f'{b:.0f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(labels_m)
    ax.set_ylim(0, 115); ax.set_ylabel('%')
    ax.set_title(src)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'comparaison_if_vs_supervise.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ comparaison_if_vs_supervise.png")

with open(os.path.join(RESULTS_DIR, 'rapport_supervise.txt'), 'w', encoding='utf-8') as f:
    f.write("PALIER 1 SUPERVISÉ — remplace Isolation Forest (HistGradientBoostingClassifier)\n" + "=" * 60 + "\n\n")
    f.write("UN SEUL modèle partagé entre les 3 sources (comme Random Forest), contrairement\n")
    f.write("à l'ancien Isolation Forest qui nécessitait un modèle dédié par source.\n")
    f.write(f"Features ({len(FEATURES)}, Tier1 uniquement, identiques à l'ancien IF) : {FEATURES}\n")
    f.write(f"Meilleurs hyperparamètres : {recherche.best_params_}\n")
    f.write(f"Seuil de décision optimal (F1 validation) : {best_seuil:.2f}\n\n")
    f.write("⚠️  COMPROMIS ASSUMÉ : ce modèle est supervisé, il ne détecte que des patterns\n")
    f.write("ressemblant aux attaques vues à l'entraînement (5 catégories connues). Il perd\n")
    f.write("la capacité 'zero-day' de l'ancien Isolation Forest (flaguer un pattern inconnu\n")
    f.write("comme suspect sans jamais l'avoir vu). Décision utilisateur explicite : accepter\n")
    f.write("ce compromis pour de meilleures performances mesurées sur trafic connu.\n\n")
    f.write("Résultats par bloc (poolé + par source) :\n")
    for r in resultats:
        f.write(f"  [{r['nom']:<20}] n={r['n']:,}  Acc={r['acc']:.1f}%  Prec={r['prec']:.1f}%  "
                f"Rec={r['rec']:.1f}%  F1={r['f1']:.1f}%  AUC={r['auc']:.1f}%  FPR={r['fpr']:.1f}%\n")
        f.write(f"      TN={r['tn']:,} FP={r['fp']:,} FN={r['fn']:,} TP={r['tp']:,}\n")
    f.write("\nComparaison vs ancien Isolation Forest (hist_gradient_boosting/results/rapport.txt) :\n")
    f.write(f"{'Source':<14}{'F1 IF (ancien)':>16}{'F1 supervisé':>14}{'  Δ':>8}"
            f"{'AUC IF (ancien)':>17}{'AUC supervisé':>15}{'  Δ':>8}\n")
    for src in sources_ok:
        r_new = next(r for r in resultats if r['nom'] == src)
        r_old = ancien_if_reference.get(src)
        if r_old:
            f.write(f"{src:<14}{r_old['f1']:>15.1f}%{r_new['f1']:>13.1f}%{r_new['f1']-r_old['f1']:>+7.1f}"
                    f"{r_old['auc']:>16.1f}%{r_new['auc']:>14.1f}%{r_new['auc']-r_old['auc']:>+7.1f}\n")

print(f"   ✓ rapport_supervise.txt")
print(f"\n{'='*70}")
print(f"✅ TERMINÉ EN {time.time()-start_total:.0f}s | Accuracy(poolé)={acc:.1f}% | "
      f"F1(poolé)={f1:.1f}% | AUC(poolé)={auc:.1f}% | FPR(poolé)={fpr:.1f}%")
print("=" * 70)
