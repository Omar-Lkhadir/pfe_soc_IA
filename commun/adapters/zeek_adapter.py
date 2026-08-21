"""
Adaptateur Zeek — DEUX formats d'entrée possibles pour la MÊME source zeek :

1. `conn.log` NATIF (JSON ou TSV déjà chargé en DataFrame par l'appelant --
   ce module ne lit pas de fichier lui-même). Colonnes (doc officielle,
   docs.zeek.org/en/current/logs/conn.html) : id.orig_p, id.resp_p, proto
   ('tcp'/'udp'/'icmp', PAS un code IANA numérique), duration (secondes),
   orig_ip_bytes/resp_ip_bytes (octets IP totaux -- PAS orig_bytes/
   resp_bytes qui sont le payload applicatif seulement, pour rester
   cohérent avec la convention NetFlow/CICFlowMeter), orig_pkts/resp_pkts,
   history. Utilisé pour l'entraînement/les tests (fichiers .labeled
   IoT-23 téléchargés directement).

2. Module FILEBEAT Zeek (production ELK réelle) : Filebeat NE PRÉSERVE PAS
   les noms de champs Zeek natifs -- il normalise tout en ECS avant même
   que cet adaptateur ne voie l'événement (vérifié sur le code source réel,
   x-pack/filebeat/module/zeek/connection/config/connection.yml,
   github.com/elastic/beats) :
     id.resp_p -> destination.port      proto -> network.transport
     orig_ip_bytes -> source.bytes      resp_ip_bytes -> destination.bytes
     orig_pkts -> source.packets        resp_pkts -> destination.packets
     duration -> event.duration (NANOSECONDES, pas secondes -- pipeline.yml
                 fait temp.duration * 1_000_000_000)
     history -> zeek.connection.history (renommage id->zeek.connection
                 global, 'history' n'est pas davantage renommé)
     conn_state -> zeek.connection.state
   AUCUN champ natif (id.resp_p, orig_ip_bytes, ...) n'existe donc dans un
   vrai événement Filebeat -- avant ce correctif, to_canonical() levait un
   KeyError sur CHAQUE événement zeek réel via ELK, silencieusement avalé
   par commun/inference.py (`except Exception: return None`) et classé
   format_non_reconnu malgré une source correctement détectée. Confirmé,
   pas hypothétique -- voir TEST/test7.

`history` encode la présence de flags par direction (majuscule=origine,
minuscule=répondeur) : S=SYN(sans ACK), H=SYN+ACK, A=ACK pur, D=payload,
F=FIN, R=RST (table complète : docs.zeek.org, script
base/protocols/conn/main.zeek). Pas de code dédié pour URG -> HAS_URG toujours
False ici (limitation documentée, pas une approximation trompeuse : Zeek ne
l'expose simplement pas dans ce champ).

Palier 2 (IAT, taille min/max paquet) : absent des deux formats -> NaN.

Statut : ENTRAÎNÉ + VALIDÉ sur données réelles (IoT-23 / Stratosphere Lab,
CTU-IoT-Malware-Capture-3-1, commun/build_zeek_data.py) via le format natif.
Note importante découverte sur données réelles (pas dans la doc) : le
format .labeled IoT-23 est tab-séparé SAUF les 3 derniers champs
(tunnel_parents, label, detailed-label), séparés par des espaces -> à gérer
explicitement au parsing (voir build_zeek_data.py), sans quoi pandas
fusionne ces 3 colonnes. Le format Filebeat/ECS est validé par
correspondance avec le pipeline source réel (ci-dessus) + test unitaire
synthétique (TEST/test7) sur des flux IoT-23 réels ré-encodés en forme
ECS -- jamais confirmé contre un VRAI cluster ELK/Filebeat en direct.
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
        via_filebeat = 'destination.port' in df_raw.columns and 'id.resp_p' not in df_raw.columns

        out = pd.DataFrame(index=df_raw.index)
        if via_filebeat:
            out['L4_DST_PORT_BUCKET'] = cs.bucket_port(df_raw['destination.port'])
            out['PROTOCOL'] = cs.bucket_protocol(df_raw['network.transport'])
            out['IN_BYTES'] = pd.to_numeric(df_raw['source.bytes'], errors='coerce')
            out['OUT_BYTES'] = pd.to_numeric(df_raw['destination.bytes'], errors='coerce')
            out['IN_PKTS'] = pd.to_numeric(df_raw['source.packets'], errors='coerce')
            out['OUT_PKTS'] = pd.to_numeric(df_raw['destination.packets'], errors='coerce')
            # event.duration est en NANOSECONDES (pipeline Filebeat) -> ms
            out['FLOW_DURATION_MILLISECONDS'] = pd.to_numeric(df_raw['event.duration'], errors='coerce') / 1e6
            history_col = df_raw.get('zeek.connection.history', pd.Series(index=df_raw.index, dtype=object))
        else:
            out['L4_DST_PORT_BUCKET'] = cs.bucket_port(df_raw['id.resp_p'])
            out['PROTOCOL'] = cs.bucket_protocol(df_raw['proto'])
            out['IN_BYTES'] = pd.to_numeric(df_raw['orig_ip_bytes'], errors='coerce')
            out['OUT_BYTES'] = pd.to_numeric(df_raw['resp_ip_bytes'], errors='coerce')
            out['IN_PKTS'] = pd.to_numeric(df_raw['orig_pkts'], errors='coerce')
            out['OUT_PKTS'] = pd.to_numeric(df_raw['resp_pkts'], errors='coerce')
            out['FLOW_DURATION_MILLISECONDS'] = pd.to_numeric(df_raw['duration'], errors='coerce') * 1000.0
            history_col = df_raw.get('history', pd.Series(index=df_raw.index, dtype=object))

        for name, series in _history_flags(history_col).items():
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
