"""
Tests unitaires des 4 adaptateurs sur des lignes synthétiques (construites
à la main d'après la documentation officielle de chaque format -- pas de
vraies données pour Suricata, jamais entraîné/validé faute de données
réelles disponibles ; Zeek, lui, EST entraîné/validé sur données réelles
IoT-23, cf. commun/build_zeek_data.py -- ces lignes synthétiques servent
seulement à couvrir vite les cas limites, pas à le valider pour la
première fois). Vérifie : le contrat de base.py (palier 1 complet sans
NaN, palier 2 présent), et quelques valeurs dérivées clés (flags TCP,
bucketisation port/protocole, conversions d'unités).

Exécution : python commun/tests/test_adapters.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from commun import canonical_schema as cs
from commun.adapters.netflow_adapter import NetFlowAdapter
from commun.adapters.cicflowmeter_adapter import CICFlowMeterAdapter
from commun.adapters.zeek_adapter import ZeekAdapter
from commun.adapters.suricata_adapter import SuricataAdapter

erreurs = []


def check(nom, cond, detail=''):
    status = 'OK' if cond else 'ECHEC'
    print(f'  [{status}] {nom} {detail}')
    if not cond:
        erreurs.append(f'{nom} {detail}')


def check_contrat(nom_source, df):
    check(f'{nom_source}: palier 1 sans NaN',
          not df[cs.TIER1_FEATURES].isna().any().any())
    check(f'{nom_source}: palier 2 present (colonnes)',
          all(c in df.columns for c in cs.TIER2_FEATURES))
    check(f'{nom_source}: colonnes meta presentes',
          all(c in df.columns for c in cs.META_COLS))


# ============================================================
print('=== NetFlowAdapter ===')
df_netflow = pd.DataFrame({
    'L4_DST_PORT': [22, 443, 80],
    'PROTOCOL': [6, 6, 17],
    'IN_BYTES': [1000, 500, 2000],
    'OUT_BYTES': [200, 1500, 0],
    'IN_PKTS': [10, 5, 20],
    'OUT_PKTS': [8, 12, 0],
    'TCP_FLAGS': [0x02 | 0x10, 0x01 | 0x10, 0x00],  # SYN+ACK, FIN+ACK, rien (UDP)
    'FLOW_DURATION_MILLISECONDS': [150, 5000, 10],
    'LONGEST_FLOW_PKT': [1460, 1460, 512],
    'SHORTEST_FLOW_PKT': [40, 40, 512],
    'SRC_TO_DST_IAT_AVG': [12.0, 300.0, 0.0],
    'SRC_TO_DST_IAT_STDDEV': [3.0, 50.0, 0.0],
    'DST_TO_SRC_IAT_AVG': [8.0, 250.0, 0.0],
    'DST_TO_SRC_IAT_STDDEV': [2.0, 40.0, 0.0],
    'Label': [1, 0, 0],
    'Attack': ['SSH-Bruteforce', 'Benign', 'Benign'],
})
out_nf = NetFlowAdapter().extract(df_netflow)
check_contrat('netflow', out_nf)
check('netflow: port 22 -> well_known(0)', out_nf.loc[0, 'L4_DST_PORT_BUCKET'] == 0)
check('netflow: PROTOCOL UDP -> bucket 1', out_nf.loc[2, 'PROTOCOL'] == 1)
check('netflow: HAS_SYN ligne0 (bit 0x02 pose)', out_nf.loc[0, 'HAS_SYN'] == 1)
check('netflow: HAS_FIN ligne1 (bit 0x01 pose)', out_nf.loc[1, 'HAS_FIN'] == 1)
check('netflow: HAS_SYN ligne2 = 0 (UDP, aucun flag)', out_nf.loc[2, 'HAS_SYN'] == 0)
check('netflow: palier 2 rempli (pas de NaN, source native)',
      not out_nf[cs.TIER2_FEATURES].isna().any().any())
check('netflow: Label_binaire ligne0 == 1', out_nf.loc[0, 'Label_binaire'] == 1)

# ============================================================
print('\n=== CICFlowMeterAdapter ===')
df_cic = pd.DataFrame({
    'Destination Port': [22, 80],
    'Source Port': [51000, 51002],
    'Protocol': [6, 6],
    'Flow Duration': [150000, -1],  # microsecondes ; 2e ligne = durée négative (artefact connu)
    'Total Length of Fwd Packets': [1000, 500],
    'Total Length of Bwd Packets': [200, 100],
    'Total Fwd Packets': [10, 5],
    'Total Backward Packets': [8, 4],
    'Max Packet Length': [1460, 512],
    'Min Packet Length': [40, 40],
    'Fwd IAT Mean': [12000.0, 1000.0],  # microsecondes
    'Fwd IAT Std': [3000.0, 200.0],
    'Bwd IAT Mean': [8000.0, 500.0],
    'Bwd IAT Std': [2000.0, 100.0],
    'FIN Flag Count': [0, 0],
    'SYN Flag Count': [1, 1],
    'RST Flag Count': [0, 0],
    'PSH Flag Count': [1, 0],
    'ACK Flag Count': [1, 1],
    'URG Flag Count': [0, 0],
    'Label': ['FTP-Patator', 'BENIGN'],
})
out_cic = CICFlowMeterAdapter().extract(df_cic)
check_contrat('cicflowmeter', out_cic)
check('cicflowmeter: duree convertie us->ms (150000us -> 150ms)',
      abs(out_cic.loc[0, 'FLOW_DURATION_MILLISECONDS'] - 150.0) < 1e-6)
check('cicflowmeter: duree negative clampee a 0',
      out_cic.loc[1, 'FLOW_DURATION_MILLISECONDS'] == 0.0)
check('cicflowmeter: IAT convertie us->ms (12000us -> 12ms)',
      abs(out_cic.loc[0, 'SRC_TO_DST_IAT_AVG'] - 12.0) < 1e-6)
check('cicflowmeter: HAS_SYN=1 (count>0)', out_cic.loc[0, 'HAS_SYN'] == 1)
check('cicflowmeter: HAS_FIN=0 (count=0)', out_cic.loc[0, 'HAS_FIN'] == 0)
check('cicflowmeter: Label_binaire attaque', out_cic.loc[0, 'Label_binaire'] == 1)
check('cicflowmeter: Label_binaire benign', out_cic.loc[1, 'Label_binaire'] == 0)

# ============================================================
print('\n=== ZeekAdapter — forme conn.log native (entraîné + validé sur IoT-23 réel) ===')
df_zeek = pd.DataFrame({
    'id.resp_p': [22, 80],
    'proto': ['tcp', 'udp'],
    'orig_ip_bytes': [1200, 600],
    'resp_ip_bytes': [300, 0],
    'orig_pkts': [10, 3],
    'resp_pkts': [8, 0],
    'duration': [0.15, 0.01],  # secondes
    'history': ['ShADadF', 'D'],  # SYN,ACK par origine (S,A,F maj) + data resp (minuscule d)
})
out_zeek = ZeekAdapter().extract(df_zeek)
check_contrat('zeek', out_zeek)
check('zeek: duree convertie s->ms (0.15s -> 150ms)',
      abs(out_zeek.loc[0, 'FLOW_DURATION_MILLISECONDS'] - 150.0) < 1e-6)
check('zeek: HAS_SYN via history "S"', out_zeek.loc[0, 'HAS_SYN'] == 1)
check('zeek: HAS_FIN via history "F"', out_zeek.loc[0, 'HAS_FIN'] == 1)
check('zeek: HAS_URG toujours 0 (non observable)', out_zeek['HAS_URG'].eq(0).all())
check('zeek: palier 2 tout en NaN (non disponible)',
      out_zeek[cs.TIER2_FEATURES].isna().all().all())

# ============================================================
print('\n=== ZeekAdapter — forme Filebeat/ECS (voir TEST/test7) ===')
# Mêmes flux que le bloc conn.log natif ci-dessus, ré-exprimés dans la forme
# ECS réelle produite par le module Filebeat Zeek (vérifié sur
# x-pack/filebeat/module/zeek/connection/config/connection.yml,
# github.com/elastic/beats) -- doit donner un résultat IDENTIQUE au bloc
# natif, preuve que les deux chemins convergent vers le même schéma
# canonique. Garde-fou de non-régression pour le bug corrigé en TEST 7
# (avant correctif : KeyError sur 'id.resp_p', absent de cette forme).
df_zeek_ecs = pd.DataFrame({
    'destination.port': [22, 80],
    'network.transport': ['tcp', 'udp'],
    'source.bytes': [1200, 600],
    'destination.bytes': [300, 0],
    'source.packets': [10, 3],
    'destination.packets': [8, 0],
    'event.duration': [0.15 * 1e9, 0.01 * 1e9],  # nanosecondes
    'zeek.connection.history': ['ShADadF', 'D'],
})
out_zeek_ecs = ZeekAdapter().extract(df_zeek_ecs)
check_contrat('zeek (ECS)', out_zeek_ecs)
check('zeek (ECS): résultat identique à la forme native',
      out_zeek_ecs[cs.TIER1_FEATURES].equals(out_zeek[cs.TIER1_FEATURES]))

# ============================================================
print('\n=== SuricataAdapter (stub, non entraine) ===')
df_suri = pd.DataFrame({
    'dest_port': [35361, 53],
    'proto': ['TCP', 'UDP'],
    'flow.bytes_toserver': [3536402, 100],
    'flow.bytes_toclient': [94102, 200],
    'flow.pkts_toserver': [3869, 1],
    'flow.pkts_toclient': [1523, 1],
    'flow.age': [40, 0],  # secondes
    'tcp.syn': [True, False],
    'tcp.ack': [True, False],
    'tcp.rst': [True, False],
    'tcp.psh': [True, False],
    # tcp.fin / tcp.urg volontairement absentes -> doivent retomber a False
})
out_suri = SuricataAdapter().extract(df_suri)
check_contrat('suricata', out_suri)
check('suricata: duree convertie s->ms (40s -> 40000ms)',
      abs(out_suri.loc[0, 'FLOW_DURATION_MILLISECONDS'] - 40000.0) < 1e-6)
check('suricata: HAS_SYN=1 (tcp.syn=True)', out_suri.loc[0, 'HAS_SYN'] == 1)
check('suricata: HAS_FIN=0 (colonne absente -> defaut False)', out_suri.loc[0, 'HAS_FIN'] == 0)
check('suricata: PROTOCOL UDP -> bucket 1', out_suri.loc[1, 'PROTOCOL'] == 1)
check('suricata: palier 2 tout en NaN (non disponible)',
      out_suri[cs.TIER2_FEATURES].isna().all().all())

# ============================================================
print(f"\n{'='*60}")
if erreurs:
    print(f"ECHEC : {len(erreurs)} assertion(s) en erreur :")
    for e in erreurs:
        print(f'  - {e}')
    sys.exit(1)
else:
    print("TOUS LES TESTS PASSENT")
