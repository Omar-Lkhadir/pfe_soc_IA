"""
Adaptateur Suricata EVE JSON, événements `event_type=flow`.

L'appelant est responsable de charger le eve.json (une ligne = un événement
JSON), filtrer sur event_type=='flow', puis aplatir en DataFrame (par ex.
`pd.json_normalize(events, sep='.')`) avant d'appeler to_canonical(). Cet
adaptateur ne fait que la conversion de colonnes.

Structure JSON de référence (doc officielle, docs.suricata.io/en/latest/
output/eve/eve-json-format.html) :
    proto: "TCP"                      (chaîne, pas un code IANA numérique)
    src_port / dest_port
    flow.pkts_toserver / flow.pkts_toclient
    flow.bytes_toserver / flow.bytes_toclient
    flow.age                          (durée en secondes, directement fournie)
    tcp.syn / tcp.ack / tcp.rst / tcp.psh   (booléens, présents dans l'exemple
                                              officiel)
    tcp.fin / tcp.urg                 (booléens documentés par Suricata mais
                                        absents de l'exemple officiel -> lus
                                        avec valeur par défaut False si la
                                        colonne n'existe pas dans le flux
                                        aplati)

Palier 2 (IAT, taille min/max paquet) : absent des événements flow standard
-> NaN.

Statut : NON ENTRAÎNÉ, pas de données Suricata réelles disponibles cette
session. Validé uniquement par test unitaire sur des événements synthétiques
construits d'après la documentation (voir tests/test_suricata_adapter.py).
"""

import numpy as np
import pandas as pd
from .base import BaseFlowAdapter
from .. import canonical_schema as cs


class SuricataAdapter(BaseFlowAdapter):
    SOURCE_NAME = 'suricata'

    def to_canonical(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df_raw = df_raw.copy()

        out = pd.DataFrame(index=df_raw.index)
        out['L4_DST_PORT_BUCKET'] = cs.bucket_port(df_raw['dest_port'])
        out['PROTOCOL'] = cs.bucket_protocol(df_raw['proto'])
        out['IN_BYTES'] = pd.to_numeric(df_raw['flow.bytes_toserver'], errors='coerce')
        out['OUT_BYTES'] = pd.to_numeric(df_raw['flow.bytes_toclient'], errors='coerce')
        out['IN_PKTS'] = pd.to_numeric(df_raw['flow.pkts_toserver'], errors='coerce')
        out['OUT_PKTS'] = pd.to_numeric(df_raw['flow.pkts_toclient'], errors='coerce')
        out['FLOW_DURATION_MILLISECONDS'] = pd.to_numeric(df_raw['flow.age'], errors='coerce') * 1000.0

        def flag_col(name):
            if name in df_raw.columns:
                return df_raw[name].fillna(False).astype(bool).astype(np.int8)
            return pd.Series(0, index=df_raw.index, dtype=np.int8)

        out['HAS_SYN'] = flag_col('tcp.syn')
        out['HAS_ACK'] = flag_col('tcp.ack')
        out['HAS_FIN'] = flag_col('tcp.fin')
        out['HAS_RST'] = flag_col('tcp.rst')
        out['HAS_PSH'] = flag_col('tcp.psh')
        out['HAS_URG'] = flag_col('tcp.urg')

        out = cs.add_derived_features(out)

        out['LONGEST_FLOW_PKT'] = np.nan
        out['SHORTEST_FLOW_PKT'] = np.nan
        out['SRC_TO_DST_IAT_AVG'] = np.nan
        out['SRC_TO_DST_IAT_STDDEV'] = np.nan
        out['DST_TO_SRC_IAT_AVG'] = np.nan
        out['DST_TO_SRC_IAT_STDDEV'] = np.nan

        label = df_raw.get('Label_binaire')
        out['Label_binaire'] = (pd.to_numeric(label, errors='coerce').fillna(0).astype(np.int8)
                                 if label is not None else np.int8(0))
        attack = df_raw.get('Attack_brut')
        out['Attack_brut'] = attack.astype(str) if attack is not None else 'Benign'
        out['source'] = self.SOURCE_NAME

        return cs.finalize_dtypes(out)
