"""
Simule un flux Zeek "en direct" en rejouant un vrai fichier conn.log.labeled
(capture reelle IoT-23 / CTU-IoT-Malware-Capture-3-1) : ecrit l'entete une
fois, puis ajoute quelques lignes de donnees toutes les quelques secondes,
en boucle. Filebeat (filebeat-nids) tail ce fichier et l'envoie a la
pipeline Logstash "nids" comme s'il s'agissait d'un vrai capteur Zeek.

Absence de capture reseau live dans cet environnement -- ceci est le moyen
honnete de demontrer la chaine temps reel complete (Filebeat -> Logstash ->
ml-service -> Elasticsearch -> Kibana) avec des donnees d'attaque reelles,
en attendant qu'une vraie sonde Zeek soit branchee en production.
"""

import os
import sys
import time

SOURCE = "/data/CTU-IoT-Malware-Capture-3-1-zeek-conn-log.labeled"
TARGET = "/var/log/zeek-live/conn.log"
LIGNES_PAR_LOT = int(os.environ.get("LIGNES_PAR_LOT", "3"))
INTERVALLE_SEC = float(os.environ.get("INTERVALLE_SEC", "4"))


def charger_source():
    header_lines = []
    data_lines = []
    with open(SOURCE, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line)
            elif line.strip():
                data_lines.append(line)
    return header_lines, data_lines


def main():
    print(f"[zeek-replay] chargement de {SOURCE}...", flush=True)
    header_lines, data_lines = charger_source()
    print(f"[zeek-replay] {len(data_lines)} lignes de donnees chargees.", flush=True)

    os.makedirs(os.path.dirname(TARGET), exist_ok=True)
    with open(TARGET, "w", encoding="utf-8") as f:
        f.writelines(header_lines)
        f.flush()
    print(f"[zeek-replay] entete Zeek ecrite dans {TARGET}.", flush=True)

    idx = 0
    n = len(data_lines)
    while True:
        with open(TARGET, "a", encoding="utf-8") as f:
            for _ in range(LIGNES_PAR_LOT):
                f.write(data_lines[idx % n])
                idx += 1
            f.flush()
        print(f"[zeek-replay] {LIGNES_PAR_LOT} evenement(s) ajoute(s) (position {idx % n}/{n})", flush=True)
        time.sleep(INTERVALLE_SEC)


if __name__ == "__main__":
    main()
