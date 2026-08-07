"""
Schéma canonique tool-agnostic partagé par Isolation Forest et Random Forest.

Contexte : les modèles historiques (v1) étaient entraînés sur les colonnes
NetFlow brutes de BigFlow-NIDS-V2.csv. Testés sur un format différent
(CICFlowMeter/CICIDS2017), ils s'effondraient car plusieurs de leurs features
n'existent tout simplement pas dans cet autre format (imputées à une
constante -> zéro information réelle). Ce module définit un schéma commun
que N'IMPORTE QUEL adaptateur de format (NetFlow, CICFlowMeter, Zeek,
Suricata, ...) doit remplir, en deux paliers :

- PALIER 1 (TIER1) : garanti par tout format de flux "biflow" standard,
  jamais de valeur manquante. Utilisé par les DEUX modèles.
- PALIER 2 (TIER2) : disponible sur NetFlow et CICFlowMeter, mais pas
  garanti partout (ex. Zeek conn.log basique n'a pas les stats d'IAT).
  Utilisé UNIQUEMENT par Random Forest (HistGradientBoostingClassifier,
  qui tolère nativement les NaN) -> jamais de constante fabriquée pour
  compenser une source qui ne fournit pas ce palier.

Voir commun/adapters/README.md pour la liste des formats couverts et
comment en ajouter un nouveau.
"""

import numpy as np
import pandas as pd

# ============================================================
# PALIER 1 — cœur, garanti par tout adaptateur
# ============================================================
TIER1_RAW = [
    'L4_DST_PORT_BUCKET',   # 0=well_known(<1024) 1=registered(1024-49151) 2=ephemeral(>=49152)
    'PROTOCOL',             # 0=TCP 1=UDP 2=OTHER (bucketisé, cf. bucket_protocol)
    'IN_BYTES',
    'OUT_BYTES',
    'IN_PKTS',
    'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS',
]

TCP_FLAG_COLS = [
    'HAS_SYN', 'HAS_ACK', 'HAS_FIN', 'HAS_RST', 'HAS_PSH', 'HAS_URG',
]

DERIVED_COLS = [
    'ratio_bytes', 'total_bytes', 'total_pkts', 'duree_sec',
]

TIER1_FEATURES = TIER1_RAW + TCP_FLAG_COLS + DERIVED_COLS  # 17 features

# ============================================================
# PALIER 2 — étendu, optionnel (NaN si l'adaptateur ne peut pas le remplir)
# ============================================================
TIER2_FEATURES = [
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT',
    'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
    'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV',
]  # 6 features

# Features utilisées par chaque modèle
ISOLATION_FOREST_FEATURES = TIER1_FEATURES                    # 17, jamais de NaN
RANDOM_FOREST_FEATURES = TIER1_FEATURES + TIER2_FEATURES      # 23, NaN toléré (palier 2)

# Colonnes non-feature toujours présentes en sortie d'un adaptateur
META_COLS = ['Label_binaire', 'Attack_brut', 'source']

ALL_OUTPUT_COLS = TIER1_FEATURES + TIER2_FEATURES + META_COLS

# Bits du bitmask NetFlow TCP_FLAGS (convention Cisco/nProbe standard)
FLAG_BITS = {'HAS_FIN': 0x01, 'HAS_SYN': 0x02, 'HAS_RST': 0x04,
             'HAS_PSH': 0x08, 'HAS_ACK': 0x10, 'HAS_URG': 0x20}


def bucket_port(port_series: pd.Series) -> pd.Series:
    """well_known(<1024)=0, registered(1024-49151)=1, ephemeral(>=49152)=2."""
    p = pd.to_numeric(port_series, errors='coerce').fillna(0)
    return np.select([p < 1024, p < 49152], [0, 1], default=2).astype(np.int8)


def bucket_protocol(proto_series: pd.Series) -> pd.Series:
    """TCP=0, UDP=1, OTHER=2. Accepte des codes IANA numériques (6, 17, ...)
    ou des chaînes ('TCP', 'UDP', ...). Bucketisé par précaution : le code de
    protocole brut peut sinon devenir un identifiant de source déguisé
    (distribution de protocoles différente entre deux datasets sans lien
    avec la vraie nature de l'attaque)."""
    s = proto_series
    if s.dtype == object:
        s_norm = s.astype(str).str.upper()
        return np.select([s_norm == 'TCP', s_norm == 'UDP'], [0, 1], default=2).astype(np.int8)
    p = pd.to_numeric(s, errors='coerce').fillna(-1)
    return np.select([p == 6, p == 17], [0, 1], default=2).astype(np.int8)


def flags_from_bitmask(tcp_flags: pd.Series) -> dict:
    """Dérive les 6 booléens palier 1 à partir d'un bitmask NetFlow (TCP_FLAGS)."""
    flags = pd.to_numeric(tcp_flags, errors='coerce').fillna(0).astype(np.int64)
    return {name: ((flags & bit) > 0).astype(np.int8) for name, bit in FLAG_BITS.items()}


def flags_from_counts(fin, syn, rst, psh, ack, urg) -> dict:
    """Dérive les 6 booléens palier 1 à partir de compteurs par flag
    (convention CICFlowMeter : 'FIN Flag Count', 'SYN Flag Count', ...)."""

    def present(count_series):
        return (pd.to_numeric(count_series, errors='coerce').fillna(0) > 0).astype(np.int8)

    return {
        'HAS_SYN': present(syn), 'HAS_ACK': present(ack), 'HAS_FIN': present(fin),
        'HAS_RST': present(rst), 'HAS_PSH': present(psh), 'HAS_URG': present(urg),
    }


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute ratio_bytes/total_bytes/total_pkts/duree_sec. `df` doit déjà
    contenir IN_BYTES/OUT_BYTES/IN_PKTS/OUT_PKTS/FLOW_DURATION_MILLISECONDS."""
    df = df.copy()
    df['ratio_bytes'] = np.where(df['IN_BYTES'] > 0, df['OUT_BYTES'] / df['IN_BYTES'], 0)
    df['ratio_bytes'] = df['ratio_bytes'].astype(np.float32).clip(0, 1000)
    df['total_bytes'] = (df['IN_BYTES'] + df['OUT_BYTES']).astype(np.float32)
    df['total_pkts'] = (df['IN_PKTS'] + df['OUT_PKTS']).astype(np.float32)
    df['duree_sec'] = (df['FLOW_DURATION_MILLISECONDS'] / 1000.0).astype(np.float32)
    return df


def finalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Force les types finaux : palier 1 en float32 sans NaN, palier 2 en
    float32 avec NaN toléré."""
    df = df.copy()
    for c in TIER1_FEATURES:
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    for c in TIER2_FEATURES:
        if c in df.columns:
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).astype(np.float32)
        else:
            df[c] = np.float32(np.nan)
    return df[ALL_OUTPUT_COLS]
