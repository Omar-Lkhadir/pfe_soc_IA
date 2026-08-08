"""
Étapes 5 et 6 du plan de refonte multi-source :

5. Contrôle de dérive par feature : pour chaque feature canonique, compare
   sa distribution entre source=netflow et source=cicflowmeter, À LABEL
   CONSTANT (normal-vs-normal, attaque-vs-attaque). Une feature qui sépare
   nettement par source plutôt que par label est un candidat au
   shortcut-learning (le modèle apprendrait "c'est telle source" au lieu de
   "c'est une attaque").
6. Table de comptage source × catégorie (5 classes Random Forest) : vérifie
   qu'aucune catégorie n'est quasi exclusive à une seule source (sinon RF
   apprendrait à reconnaître une source plutôt qu'une catégorie d'attaque).

Ce script ne bloque rien automatiquement — il produit un rapport à lire
avant de lancer l'entraînement (étapes 7+).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from commun import canonical_schema as cs
from commun.mapping_attaques import mapper

DATA_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep + 'data'
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rapport_derive_et_equilibre.txt')

print("=" * 70)
print("CONTROLE DE DERIVE PAR FEATURE + EQUILIBRE SOURCE x CATEGORIE")
print("=" * 70)

df_nf = pd.read_pickle(os.path.join(DATA_DIR, 'netflow_canonical.pkl'))
df_cic = pd.read_pickle(os.path.join(DATA_DIR, 'cicflowmeter_canonical.pkl'))
print(f"\nNetFlow : {len(df_nf):,} lignes | CICFlowMeter : {len(df_cic):,} lignes")

lignes_rapport = []
lignes_rapport.append("RAPPORT — CONTROLE DE DERIVE PAR FEATURE + EQUILIBRE SOURCE x CATEGORIE")
lignes_rapport.append("=" * 78)
lignes_rapport.append(f"\nNetFlow : {len(df_nf):,} lignes | CICFlowMeter : {len(df_cic):,} lignes\n")

# ============================================================
# ETAPE 5 : CONTROLE DE DERIVE (KS-test par feature, à label constant)
# ============================================================
print("\n[ETAPE 5] Controle de derive par feature (KS-test, a label constant)...")
lignes_rapport.append("-" * 78)
lignes_rapport.append("ETAPE 5 : CONTROLE DE DERIVE PAR FEATURE (test de Kolmogorov-Smirnov)")
lignes_rapport.append("-" * 78)
lignes_rapport.append("D >= 0.3 : separation nette par SOURCE plutot que par label -> a surveiller\n")

features_a_tester = cs.TIER1_FEATURES + cs.TIER2_FEATURES
suspects = []

for label_val, nom_label in [(0, 'Normal'), (1, 'Attaque')]:
    lignes_rapport.append(f"\n--- {nom_label} (Label_binaire={label_val}) ---")
    sub_nf = df_nf[df_nf.Label_binaire == label_val]
    sub_cic = df_cic[df_cic.Label_binaire == label_val]
    if len(sub_nf) < 50 or len(sub_cic) < 50:
        lignes_rapport.append("  (echantillon trop petit d'un cote, ignore)")
        continue
    for feat in features_a_tester:
        a = sub_nf[feat].dropna().astype(float).values
        b = sub_cic[feat].dropna().astype(float).values
        if len(a) < 50 or len(b) < 50:
            lignes_rapport.append(f"  {feat:<28s} : donnees insuffisantes (palier 2 absent ?)")
            continue
        # sous-echantillonnage pour un KS-test rapide sur de gros volumes
        rng = np.random.RandomState(42)
        if len(a) > 20000:
            a = rng.choice(a, 20000, replace=False)
        if len(b) > 20000:
            b = rng.choice(b, 20000, replace=False)
        d_stat, p_val = stats.ks_2samp(a, b)
        marqueur = " <-- SUSPECT" if d_stat >= 0.3 else ""
        lignes_rapport.append(f"  {feat:<28s} : D={d_stat:.3f}  p={p_val:.1e}{marqueur}")
        if d_stat >= 0.3:
            suspects.append((nom_label, feat, d_stat))

print(f"   {len(suspects)} paire(s) (label, feature) suspecte(s) (D>=0.3)")
for nom_label, feat, d in suspects:
    print(f"      {nom_label} / {feat} : D={d:.3f}")

lignes_rapport.append(f"\n=> {len(suspects)} paire(s) suspecte(s) au total.")

# ============================================================
# ETAPE 6 : TABLE SOURCE x CATEGORIE (5 classes RF)
# ============================================================
print("\n[ETAPE 6] Table de comptage source x categorie...")
lignes_rapport.append("\n" + "-" * 78)
lignes_rapport.append("ETAPE 6 : TABLE SOURCE x CATEGORIE (Random Forest, 5 classes)")
lignes_rapport.append("-" * 78)

att_nf = df_nf[df_nf.Label_binaire == 1].copy()
att_cic = df_cic[df_cic.Label_binaire == 1].copy()
att_nf['categorie'] = att_nf['Attack_brut'].apply(mapper)
att_cic['categorie'] = att_cic['Attack_brut'].apply(mapper)

from commun.mapping_attaques import TYPES_ATTAQUES
table = pd.DataFrame(index=TYPES_ATTAQUES, columns=['netflow', 'cicflowmeter', 'total', '%_min_source'])
categories_exclusives = []
for cat in TYPES_ATTAQUES:
    n_nf = int((att_nf['categorie'] == cat).sum())
    n_cic = int((att_cic['categorie'] == cat).sum())
    total = n_nf + n_cic
    pct_min = 100 * min(n_nf, n_cic) / total if total > 0 else 0
    table.loc[cat] = [n_nf, n_cic, total, round(pct_min, 1)]
    if total > 0 and pct_min < 5:
        categories_exclusives.append(cat)

print(table.to_string())
lignes_rapport.append("\n" + table.to_string())
if categories_exclusives:
    msg = f"\n<!> Categories quasi exclusives a une source (<5% de l'autre) : {categories_exclusives}"
else:
    msg = "\nAucune categorie quasi exclusive a une source (>=5% de chaque cote)."
print(msg)
lignes_rapport.append(msg)

with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lignes_rapport))
print(f"\nRapport ecrit -> {RESULTS_FILE}")
