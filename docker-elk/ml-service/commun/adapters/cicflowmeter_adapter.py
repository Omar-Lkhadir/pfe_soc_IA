"""
Adaptateur CICFlowMeter (format brut CICIDS2017 : Thursday-*.pcap_ISCX.csv,
Monday-WorkingHours.pcap_ISCX.csv, ...).

Unités vérifiées sur données réelles (voir étape 1 du plan) : `Flow Duration`
et les champs `Fwd/Bwd IAT Mean/Std` sont tous en MICROSECONDES (confirmé par
comparaison des ordres de grandeur, max IAT proche du max Flow Duration) ->
conversion /1000 vers ms pour rester cohérent avec la convention NetFlow.
Correction par rapport à TEST/test1/scripts/01_nettoyage_test.py (session
précédente) qui convertissait la durée mais PAS les 4 champs IAT — bug corrigé
ici.

`Flow Duration` contient occasionnellement des valeurs négatives (artefact
connu de CICFlowMeter) -> clip à 0.

Statut : ENTRAÎNÉ + VALIDÉ sur données réelles (8 fichiers CICIDS2017).
"""

import numpy as np
import pandas as pd
from .base import BaseFlowAdapter
from .. import canonical_schema as cs

COLS_VOULUES = {
    'Destination Port', 'Source Port', 'Protocol', 'Flow Duration',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Total Fwd Packets', 'Total Backward Packets',
    'Max Packet Length', 'Min Packet Length',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Bwd IAT Mean', 'Bwd IAT Std',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count',
    'Label',
}

# Cf. netflow_adapter.RAW_COLS_TIER2_ONLY : sous-ensemble n'alimentant que le
# palier 2, a exclure du taux d'incompletude qui bloque le scoring.
RAW_COLS_TIER2_ONLY = {
    'Max Packet Length', 'Min Packet Length',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Bwd IAT Mean', 'Bwd IAT Std',
}


class CICFlowMeterAdapter(BaseFlowAdapter):
    SOURCE_NAME = 'cicflowmeter'

    def to_canonical(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df_raw = df_raw.copy()
        df_raw.columns = [c.strip() for c in df_raw.columns]

        label_str = df_raw['Label'].astype(str).str.strip()
        valid = label_str.replace({'nan': ''}) != ''
        df_raw = df_raw[valid].reset_index(drop=True)
        label_str = label_str[valid].reset_index(drop=True)

        out = pd.DataFrame(index=df_raw.index)
        out['L4_DST_PORT_BUCKET'] = cs.bucket_port(df_raw['Destination Port'])
        out['PROTOCOL'] = cs.bucket_protocol(pd.to_numeric(df_raw['Protocol'], errors='coerce'))
        out['IN_BYTES'] = pd.to_numeric(df_raw['Total Length of Fwd Packets'], errors='coerce')
        out['OUT_BYTES'] = pd.to_numeric(df_raw['Total Length of Bwd Packets'], errors='coerce')
        out['IN_PKTS'] = pd.to_numeric(df_raw['Total Fwd Packets'], errors='coerce')
        out['OUT_PKTS'] = pd.to_numeric(df_raw['Total Backward Packets'], errors='coerce')

        duree_us = pd.to_numeric(df_raw['Flow Duration'], errors='coerce').clip(lower=0)
        out['FLOW_DURATION_MILLISECONDS'] = duree_us / 1000.0

        for name, series in cs.flags_from_counts(
            fin=df_raw['FIN Flag Count'], syn=df_raw['SYN Flag Count'],
            rst=df_raw['RST Flag Count'], psh=df_raw['PSH Flag Count'],
            ack=df_raw['ACK Flag Count'], urg=df_raw['URG Flag Count'],
        ).items():
            out[name] = series

        out = cs.add_derived_features(out)

        # Palier 2
        out['LONGEST_FLOW_PKT'] = pd.to_numeric(df_raw['Max Packet Length'], errors='coerce')
        out['SHORTEST_FLOW_PKT'] = pd.to_numeric(df_raw['Min Packet Length'], errors='coerce')
        out['SRC_TO_DST_IAT_AVG'] = pd.to_numeric(df_raw['Fwd IAT Mean'], errors='coerce') / 1000.0
        out['SRC_TO_DST_IAT_STDDEV'] = pd.to_numeric(df_raw['Fwd IAT Std'], errors='coerce') / 1000.0
        out['DST_TO_SRC_IAT_AVG'] = pd.to_numeric(df_raw['Bwd IAT Mean'], errors='coerce') / 1000.0
        out['DST_TO_SRC_IAT_STDDEV'] = pd.to_numeric(df_raw['Bwd IAT Std'], errors='coerce') / 1000.0

        out['Label_binaire'] = (label_str.str.upper() != 'BENIGN').astype(np.int8)
        out['Attack_brut'] = np.where(out['Label_binaire'] == 1, label_str, 'Benign')
        out['source'] = self.SOURCE_NAME

        return cs.finalize_dtypes(out)
