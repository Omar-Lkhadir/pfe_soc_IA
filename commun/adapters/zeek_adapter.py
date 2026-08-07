"""
Adaptateur Zeek `conn.log` (JSON ou TSV déjà chargé en DataFrame par
l'appelant — ce module ne lit pas de fichier, il convertit des colonnes).

Colonnes source (doc officielle, docs.zeek.org/en/current/logs/conn.html) :
id.orig_p, id.resp_p, proto ('tcp'/'udp'/'icmp', PAS un code IANA numérique),
duration (secondes), orig_ip_bytes/resp_ip_bytes (octets IP totaux — PAS
orig_bytes/resp_bytes qui sont le payload applicatif seulement, pour rester
cohérent avec la convention NetFlow/CICFlowMeter qui comptent les octets IP
totaux), orig_pkts/resp_pkts, history.

`history` encode la présence de flags par direction (majuscule=origine,
minuscule=répondeur) : S=SYN(sans ACK), H=SYN+ACK, A=ACK pur, D=payload,
F=FIN, R=RST (table complète : docs.zeek.org, script
base/protocols/conn/main.zeek). Pas de code dédié pour URG -> HAS_URG toujours
False ici (limitation documentée, pas une approximation trompeuse : Zeek ne
l'expose simplement pas dans ce champ).

Palier 2 (IAT, taille min/max paquet) : absent de conn.log standard -> NaN.

Statut : NON ENTRAÎNÉ, pas de données Zeek réelles disponibles cette session.
Validé uniquement par test unitaire sur des lignes synthétiques construites
d'après la documentation (voir tests/test_zeek_adapter.py). À faire passer en
"entraîné" le jour où de vraies données conn.log sont disponibles, en suivant
la même procédure que cicflowmeter_adapter.py.
"""

import numpy as np
import pandas as pd
from .base import BaseFlowAdapter
from .. import canonical_schema as cs


def _history_flags(history: pd.Series) -> dict:
    h = history.fillna('').astype(str).str.lower()
    return {
        'HAS_SYN': (h.str.contains('s') | h.str.contains('h')).astype(np.int8),
        'HAS_ACK': (h.str.contains('a') | h.str.contains('h')).astype(np.int8),
        'HAS_FIN': h.str.contains('f').astype(np.int8),
        'HAS_RST': h.str.contains('r').astype(np.int8),
        'HAS_PSH': h.str.contains('d').astype(np.int8),  # proxy : paquet avec payload
        'HAS_URG': pd.Series(0, index=history.index, dtype=np.int8),  # non observable
    }


class ZeekAdapter(BaseFlowAdapter):
    SOURCE_NAME = 'zeek'

    def to_canonical(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df_raw = df_raw.copy()

        out = pd.DataFrame(index=df_raw.index)
        out['L4_DST_PORT_BUCKET'] = cs.bucket_port(df_raw['id.resp_p'])
        out['PROTOCOL'] = cs.bucket_protocol(df_raw['proto'])
        out['IN_BYTES'] = pd.to_numeric(df_raw['orig_ip_bytes'], errors='coerce')
        out['OUT_BYTES'] = pd.to_numeric(df_raw['resp_ip_bytes'], errors='coerce')
        out['IN_PKTS'] = pd.to_numeric(df_raw['orig_pkts'], errors='coerce')
        out['OUT_PKTS'] = pd.to_numeric(df_raw['resp_pkts'], errors='coerce')
        out['FLOW_DURATION_MILLISECONDS'] = pd.to_numeric(df_raw['duration'], errors='coerce') * 1000.0

        for name, series in _history_flags(df_raw.get('history', pd.Series(index=df_raw.index, dtype=object))).items():
            out[name] = series

        out = cs.add_derived_features(out)

        # Palier 2 : non disponible dans conn.log standard
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
