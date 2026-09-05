#!/usr/bin/env bash
# Recree la regle d'alerte NIDS + ses connecteurs dans Kibana.
#
# Pourquoi un script separe (pas dans nids-dashboards.ndjson) : les regles
# d'alerte Kibana (type "alert") embarquent une cle d'API chiffree geree par
# le plugin Alerting/Task Manager -- un import via l'API generique des saved
# objects la recree desactivee et cassee. Les API dediees /api/actions et
# /api/alerting recreent une regle saine directement.
#
# Ce que ca fait :
#   - Connecteur "Log Kibana - Alertes NIDS" (.server-log) : trace l'alerte
#     dans les logs du conteneur Kibana.
#   - Connecteur "Index - Alertes NIDS" (.index) : ecrit un document par
#     declenchement dans l'index nids-alerts (IP/hostname attaquant, cible,
#     categorie, score) -- consomme par le panneau "Alertes actives" du
#     dashboard temps reel (nids-dashboard, cf. nids-dashboards.ndjson).
#   - Regle "NIDS - Attaque detectee" (.es-query, verifie nids-ml-* toutes
#     les 1 min, condition is_attack:true) declenchant les deux connecteurs.
#
# Prerequis : Kibana up, dashboards deja importes (nids-dashboards.ndjson),
# donc l'index-pattern nids-ml existe. A relancer seulement apres une perte
# de donnees Kibana (les IDs de connecteurs generes different a chaque run,
# donc ne PAS relancer si la regle existe deja -- verifier d'abord avec
# `curl .../api/alerting/rules/_find`).

set -euo pipefail

KIBANA_URL="${KIBANA_URL:-http://localhost:5601}"

log_connector_id=$(curl -sf -X POST "$KIBANA_URL/api/actions/connector" \
  -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d '{
    "name": "Log Kibana - Alertes NIDS",
    "connector_type_id": ".server-log"
  }' | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
echo "Connecteur server-log cree : $log_connector_id"

index_connector_id=$(curl -sf -X POST "$KIBANA_URL/api/actions/connector" \
  -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d '{
    "name": "Index - Alertes NIDS",
    "connector_type_id": ".index",
    "config": { "index": "nids-alerts" }
  }' | python3 -c "import json,sys;print(json.load(sys.stdin)['id'])")
echo "Connecteur index cree : $index_connector_id"

curl -sf -X POST "$KIBANA_URL/api/alerting/rule" \
  -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -d "$(python3 -c "
import json
print(json.dumps({
    'name': 'NIDS - Attaque detectee',
    'rule_type_id': '.es-query',
    'consumer': 'alerts',
    'schedule': {'interval': '1m'},
    'tags': [],
    'params': {
        'searchType': 'esQuery',
        'esQuery': json.dumps({'query': {'term': {'is_attack': True}}}),
        'index': ['nids-ml-*'],
        'timeField': '@timestamp',
        'size': 100,
        'thresholdComparator': '>',
        'threshold': [0],
        'timeWindowSize': 1,
        'timeWindowUnit': 'm',
        'excludeHitsFromPreviousRun': True,
        'aggType': 'count',
        'groupBy': 'all',
    },
    'actions': [
        {
            'group': 'query matched',
            'id': '$log_connector_id',
            'params': {
                'message': 'ALERTE NIDS: {{context.hits.length}} attaque(s) reseau detectee(s) par le modele IA dans les 1 dernieres minutes. Voir index nids-ml-*.',
            },
        },
        {
            'group': 'query matched',
            'id': '$index_connector_id',
            'params': {
                'documents': [{
                    '@timestamp': '{{context.date}}',
                    'message': '{{context.message}}',
                    'ip_source': '{{context.hits.0._source.ip_source}}',
                    'attacker_hostname': '{{context.hits.0._source.attacker_hostname}}',
                    'ip_destination': '{{context.hits.0._source.ip_destination}}',
                    'attack_category': '{{context.hits.0._source.attack_category}}',
                    'anomaly_score': '{{context.hits.0._source.anomaly_score}}',
                    'nb_attaques': '{{context.value}}',
                    'rule_name': 'NIDS - Attaque detectee',
                    'severity': 'high',
                }],
            },
        },
    ],
}))
")" | python3 -c "import json,sys;d=json.load(sys.stdin);print('Regle creee :', d.get('id', d))"

echo "Termine. Verifier : curl -s $KIBANA_URL/api/alerting/rules/_find -H 'kbn-xsrf: true'"
