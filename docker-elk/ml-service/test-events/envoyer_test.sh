#!/usr/bin/env bash
# Envoie les événements de test (evenements_test.jsonl) ligne par ligne vers
# la pipeline Logstash "nids" (input tcp, port 50001), qui appelle ml-service
# puis indexe le résultat dans Elasticsearch (index nids-ml-*).
#
# Usage : ./envoyer_test.sh [host] [port]

set -euo pipefail

HOST="${1:-localhost}"
PORT="${2:-50001}"
FICHIER="$(dirname "$0")/evenements_test.jsonl"

echo "Envoi de $(wc -l < "$FICHIER") événement(s) vers ${HOST}:${PORT}..."
nc -q1 "$HOST" "$PORT" < "$FICHIER"
echo "Envoyé. Vérifier dans Kibana (index nids-ml-*) ou avec :"
echo "  curl -s 'http://${HOST}:9200/nids-ml-*/_search?pretty&size=20'"
