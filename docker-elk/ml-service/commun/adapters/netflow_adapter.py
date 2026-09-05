"""
Adaptateur NetFlow-v9 / IPFIX (format BigFlow-NIDS-V2.csv, export nProbe).
Toute sonde NetFlow v9/IPFIX standard exporte des champs équivalents
(cf. registre IANA IPFIX) — cet adaptateur généralise donc au-delà du seul
fichier BigFlow-NIDS-V2.

Colonnes brutes attendues (sous-ensemble des ~31 colonnes de
isolation_forest/scripts/nettoyage.py) : L4_DST_PORT, PROTOCOL, IN_BYTES,
OUT_BYTES, IN_PKTS, OUT_PKTS, TCP_FLAGS, FLOW_DURATION_MILLISECONDS,
LONGEST_FLOW_PKT, SHORTEST_FLOW_PKT, SRC_TO_DST_IAT_AVG,
SRC_TO_DST_IAT_STDDEV, DST_TO_SRC_IAT_AVG, DST_TO_SRC_IAT_STDDEV, Label,
Attack.

Statut : ENTRAÎNÉ + VALIDÉ sur données réelles (BigFlow-NIDS-V2.csv).
"""

import numpy as np
import pandas as pd
from .base import BaseFlowAdapter
from .. import canonical_schema as cs

RAW_COLS = [
    'L4_DST_PORT', 'PROTOCOL', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS', 'LONGEST_FLOW_PKT',
    'SHORTEST_FLOW_PKT', 'SRC_TO_DST_IAT_AVG', 'SRC_TO_DST_IAT_STDDEV',
    'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV', 'Label', 'Attack',
]

# Sous-ensemble de RAW_COLS qui n'alimente QUE le palier 2 (jamais le palier
# 1) -- vu sur le dataset public NF-BoT-IoT (schema NetFlow v1 UQ, 14
# colonnes) : ces 6 colonnes sont absentes alors que tout le palier 1 est
# present. Sans cette distinction, inference.py comptait ces colonnes dans
# le taux d'incompletude global (6/14 ~ 43% > SEUIL_IMPUTATION) et rejetait
# TOUT le fichier (format_non_reconnu) alors que la detection d'attaque
# (palier 1) etait parfaitement calculable.
RAW_COLS_TIER2_ONLY = {
    'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT', 'SRC_TO_DST_IAT_AVG',
    'SRC_TO_DST_IAT_STDDEV', 'DST_TO_SRC_IAT_AVG', 'DST_TO_SRC_IAT_STDDEV',
}


class NetFlowAdapter(BaseFlowAdapter):
    SOURCE_NAME = 'netflow'

    def to_canonical(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df_raw = df_raw.copy()
        df_raw.columns = [c.strip() for c in df_raw.columns]
        for c in ['L4_DST_PORT', 'PROTOCOL', 'IN_BYTES', 'OUT_BYTES', 'IN_PKTS',
                  'OUT_PKTS', 'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS',
                  'LONGEST_FLOW_PKT', 'SHORTEST_FLOW_PKT']:
            df_raw[c] = pd.to_numeric(df_raw[c], errors='coerce')

        out = pd.DataFrame(index=df_raw.index)
        out['L4_DST_PORT_BUCKET'] = cs.bucket_port(df_raw['L4_DST_PORT'])
        out['PROTOCOL'] = cs.bucket_protocol(df_raw['PROTOCOL'])
        out['IN_BYTES'] = df_raw['IN_BYTES']
        out['OUT_BYTES'] = df_raw['OUT_BYTES']
        out['IN_PKTS'] = df_raw['IN_PKTS']
        out['OUT_PKTS'] = df_raw['OUT_PKTS']
        out['FLOW_DURATION_MILLISECONDS'] = df_raw['FLOW_DURATION_MILLISECONDS']

        for name, series in cs.flags_from_bitmask(df_raw['TCP_FLAGS']).items():
            out[name] = series

        out = cs.add_derived_features(out)

        # Palier 2 : disponible nativement en NetFlow
        out['LONGEST_FLOW_PKT'] = df_raw['LONGEST_FLOW_PKT']
        out['SHORTEST_FLOW_PKT'] = df_raw['SHORTEST_FLOW_PKT']
        out['SRC_TO_DST_IAT_AVG'] = pd.to_numeric(df_raw['SRC_TO_DST_IAT_AVG'], errors='coerce')
        out['SRC_TO_DST_IAT_STDDEV'] = pd.to_numeric(df_raw['SRC_TO_DST_IAT_STDDEV'], errors='coerce')
        out['DST_TO_SRC_IAT_AVG'] = pd.to_numeric(df_raw['DST_TO_SRC_IAT_AVG'], errors='coerce')
        out['DST_TO_SRC_IAT_STDDEV'] = pd.to_numeric(df_raw['DST_TO_SRC_IAT_STDDEV'], errors='coerce')

        out['Label_binaire'] = pd.to_numeric(df_raw['Label'], errors='coerce').fillna(0).astype(np.int8)
        out['Attack_brut'] = df_raw['Attack'].fillna('Benign').astype(str).str.strip()
        out['source'] = self.SOURCE_NAME

        return cs.finalize_dtypes(out)
