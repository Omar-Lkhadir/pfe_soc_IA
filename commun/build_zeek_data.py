"""
Nettoyage + visualisation + adaptation canonique du jeu Zeek réel (IoT-23,
Stratosphere Lab / CTU) téléchargé dans data/raw_zeek/. Fait passer
ZeekAdapter du statut "testé sur données synthétiques" à "validé sur
données réelles" (même procédure que cicflowmeter_adapter.py en son temps).

Source : CTU-IoT-Malware-Capture-3-1 (botnet Muhstik sur Raspberry Pi),
format conn.log.labeled — https://www.stratosphereips.org/datasets-iot23
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from commun import canonical_schema as cs
from commun.adapters.zeek_adapter import ZeekAdapter
from commun.mapping_attaques import mapper

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

MD4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_FILE = os.path.join(MD4_DIR, 'data', 'raw_zeek', 'CTU-IoT-Malware-Capture-3-1_conn.log.labeled')
OUT_DIR = os.path.join(MD4_DIR, 'commun', 'data')
VIZ_DIR = os.path.join(MD4_DIR, 'commun', 'visualisations_zeek')
os.makedirs(VIZ_DIR, exist_ok=True)

print("=" * 70)
print("NETTOYAGE + VISUALISATION — Zeek conn.log.labeled (IoT-23, données réelles)")
print("=" * 70)

# ============================================================
# ÉTAPE 1 : LECTURE (format Zeek TSV, lignes # = commentaires/en-tête)
# ============================================================
print(f"\n📂 [ÉTAPE 1] Lecture de {os.path.basename(RAW_FILE)}...")

# Quirk connue du format IoT-23 .labeled (Stratosphere Lab) : tout le fichier
# est tab-séparé SAUF les 3 derniers champs (tunnel_parents, label,
# detailed-label), séparés par des espaces -> on les lit comme une seule
# colonne "tail" puis on la re-découpe.
colonnes_tab = ['ts', 'uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p',
                 'proto', 'service', 'duration', 'orig_bytes', 'resp_bytes',
                 'conn_state', 'local_orig', 'local_resp', 'missed_bytes',
                 'history', 'orig_pkts', 'orig_ip_bytes', 'resp_pkts',
                 'resp_ip_bytes', 'tail']

df_raw = pd.read_csv(RAW_FILE, sep='\t', comment='#', names=colonnes_tab,
                      na_values=['-', '(empty)'], low_memory=False)
tail_split = df_raw['tail'].astype(str).str.split(r'\s+', n=2, regex=True, expand=True)
df_raw['tunnel_parents'] = tail_split[0]
df_raw['label'] = tail_split[1]
df_raw['detailed-label'] = tail_split[2]
df_raw = df_raw.drop(columns=['tail'])
print(f"   ✓ {len(df_raw):,} flux bruts, {len(df_raw.columns)} colonnes : {df_raw.columns.tolist()}")

# ============================================================
# ÉTAPE 2 : NETTOYAGE
# ============================================================
print(f"\n🧹 [ÉTAPE 2] Nettoyage...")

avant = len(df_raw)
df_raw = df_raw.dropna(subset=['id.orig_p', 'id.resp_p', 'proto']).reset_index(drop=True)
print(f"   ✓ {avant:,} -> {len(df_raw):,} lignes (lignes sans port/protocole écartées)")

# label/detailed-label -> Label_binaire (0=Benign) / Attack_brut (type mappé 5 catégories)
df_raw['Label_binaire'] = (df_raw['label'].astype(str).str.strip() != 'Benign').astype(np.int8)
detailed = df_raw['detailed-label'].fillna('Benign').astype(str).str.strip()
df_raw['Attack_brut'] = detailed

nb_normal = int((df_raw['Label_binaire'] == 0).sum())
nb_attaques = int((df_raw['Label_binaire'] == 1).sum())
print(f"   Normal={nb_normal:,} ({nb_normal/len(df_raw)*100:.1f}%) | "
      f"Attaques={nb_attaques:,} ({nb_attaques/len(df_raw)*100:.1f}%)")
print(f"   Types d'attaques bruts : {detailed[df_raw['Label_binaire']==1].value_counts().to_dict()}")

# ============================================================
# ÉTAPE 3 : ADAPTATION AU SCHÉMA CANONIQUE (ZeekAdapter, validé ici sur données réelles)
# ============================================================
print(f"\n🔧 [ÉTAPE 3] Adaptation au schéma canonique (ZeekAdapter)...")

adapter = ZeekAdapter()
df_canon = adapter.extract(df_raw)  # to_canonical() + validate()
print(f"   ✓ {len(df_canon):,} lignes canoniques, {len(cs.ALL_OUTPUT_COLS)} colonnes")
print(f"   ✓ Palier 2 (IAT/taille paquet) : {df_canon[cs.TIER2_FEATURES].isna().all().all() and 'NaN partout (attendu, absent de conn.log standard)' or 'partiellement rempli'}")

categorie_5 = df_canon['Attack_brut'].apply(lambda a: mapper(a) if a != 'Benign' else 'Benign')
print(f"\n   Répartition 5 catégories (attaques) :")
print(f"   {categorie_5[df_canon['Label_binaire']==1].value_counts().to_string()}")

df_canon.to_pickle(os.path.join(OUT_DIR, 'zeek_canonical.pkl'))
print(f"\n💾 Sauvegardé -> commun/data/zeek_canonical.pkl")

# ============================================================
# GRAPHIQUES
# ============================================================
print(f"\n📈 [GRAPHIQUES]...")

fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
fig1.suptitle('Distribution du Trafic — Zeek/IoT-23 (CTU-IoT-Malware-Capture-3-1, données réelles)',
               fontsize=16, fontweight='bold', y=1.02)
axes1[0].pie([nb_normal, nb_attaques], labels=['Normal', 'Attaque'],
             colors=['#2ecc71', '#e74c3c'], autopct='%1.1f%%',
             explode=(0, 0.06), startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'})
axes1[0].set_title(f'{len(df_raw):,} Flux', fontweight='bold')
axes1[1].barh(['Normal', 'Attaque'], [nb_normal, nb_attaques],
              color=['#2ecc71', '#e74c3c'], edgecolor='white', height=0.6)
for i, v in enumerate([nb_normal, nb_attaques]):
    axes1[1].text(v + len(df_raw)*0.01, i, f'{v:,} ({v/len(df_raw)*100:.1f}%)',
                  fontsize=12, fontweight='bold', va='center')
axes1[1].set_title('Comptage', fontweight='bold')
axes1[1].set_xlim(0, max(nb_normal, nb_attaques) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '01_distribution_globale.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ 01_distribution_globale.png")

fig2, ax2 = plt.subplots(figsize=(10, 6))
couleurs = ['#e74c3c', '#f39c12', '#3498db', '#9b59b6']
vc = detailed[df_raw['Label_binaire']==1].value_counts()
ax2.barh(vc.index[::-1], vc.values[::-1], color=couleurs[:len(vc)][::-1], edgecolor='white')
for i, v in enumerate(vc.values[::-1]):
    ax2.text(v + max(vc.values)*0.01, i, f'{v:,}', va='center', fontweight='bold')
ax2.set_xlabel("Nombre de flux")
ax2.set_title("Types d'attaques bruts (IoT-23, botnet Muhstik)", fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '02_types_attaques_bruts.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ 02_types_attaques_bruts.png")

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 12))
fig3.suptitle('Distribution des Features Canoniques (palier 1)', fontsize=16, fontweight='bold', y=1.01)
axes3[0, 0].hist(df_canon['OUT_BYTES'].clip(0, 5000), bins=60, color='#e74c3c', alpha=0.7, edgecolor='white')
axes3[0, 0].set_xlabel('Bytes Out'); axes3[0, 0].set_title('Volume Sortant', fontweight='bold')
axes3[0, 1].hist(df_canon['IN_BYTES'].clip(0, 5000), bins=60, color='#3498db', alpha=0.7, edgecolor='white')
axes3[0, 1].set_xlabel('Bytes In'); axes3[0, 1].set_title('Volume Entrant', fontweight='bold')
axes3[1, 0].hist(df_canon['duree_sec'].clip(0, 30), bins=60, color='#2ecc71', alpha=0.7, edgecolor='white')
axes3[1, 0].set_xlabel('Durée (s)'); axes3[1, 0].set_title('Durée des Flux', fontweight='bold')
axes3[1, 1].hist(df_canon['ratio_bytes'].clip(0, 20), bins=60, color='#f39c12', alpha=0.7, edgecolor='white')
axes3[1, 1].set_xlabel('Ratio OUT/IN bytes'); axes3[1, 1].set_title('Ratio Bytes', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(VIZ_DIR, '03_distribution_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✓ 03_distribution_features.png")

print(f"\n{'='*70}")
print("✅ TERMINÉ")
print("=" * 70)
