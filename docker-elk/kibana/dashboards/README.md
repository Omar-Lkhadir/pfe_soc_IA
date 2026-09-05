# Dashboards NIDS (SOC IA)

Export des saved objects Kibana pour les deux tableaux de bord du pipeline NIDS :

- `nids-dashboard` — **Détection d'intrusion (temps réel)**, alimenté par l'index
  `nids-ml-*` (flux live Filebeat → Logstash → ml-service). C'est le tableau de
  bord principal/standard. La table d'événements affiche l'IP source
  (`id.orig_h`) et son nom d'hôte résolu (`attacker_hostname`, via le
  dictionnaire `logstash/pipeline/attacker_hosts.yml`).
- `nids-import-dashboard` — **Analyse des données importées**, alimenté par
  l'index `nids-import` (upload CSV ponctuel via `ml-service` `/predict_csv`).
  Accessible depuis le dashboard temps réel via le lien en haut de page ; un
  lien retour est disponible dans l'autre sens.

## Alerte en cas d'attaque detectee

Une règle Kibana Alerting (`NIDS - Attaque detectee`, type `.es-query`)
surveille l'index `nids-ml-*` toutes les minutes (`is_attack: true`). À
chaque déclenchement, elle :

1. écrit une ligne dans les logs Kibana (connecteur `.server-log`) ;
2. indexe un document dans `nids-alerts` (IP et nom d'hôte de l'attaquant,
   cible, catégorie, score) via un connecteur `.index`.

Le panneau rouge **"🚨 Alertes actives"** en haut du dashboard temps réel
(`nids-dashboard`) affiche ces documents — c'est la partie visible de
l'alerte pour l'opérateur SOC, pas seulement un log serveur.

Cette règle + ses connecteurs ne sont **pas** dans le ndjson ci-dessus (une
règle d'alerte embarque une clé d'API chiffrée que l'import générique de
saved objects ne recrée pas correctement) : recréez-les avec
`./setup_alerting.sh` (voir le script pour le détail, à exécuter après avoir
importé les dashboards).

## Réimporter après une réinitialisation d'Elasticsearch/Kibana

```bash
curl -X POST "http://localhost:5601/api/saved_objects/_import?overwrite=true" \
  -H "kbn-xsrf: true" \
  --form file=@kibana/dashboards/nids-dashboards.ndjson
```

Puis définir le dashboard temps réel comme page d'accueil de Kibana :

```bash
curl -X POST "http://localhost:5601/internal/kibana/settings" \
  -H "kbn-xsrf: true" -H "Content-Type: application/json" \
  -H "x-elastic-internal-origin: kibana" \
  -d '{"changes":{"defaultRoute":"/app/dashboards#/view/nids-dashboard"}}'
```

(Ce réglage `defaultRoute` est une "advanced setting" globale, pas un saved
object — elle n'est donc pas incluse dans l'export ndjson et doit être
réappliquée séparément.)
