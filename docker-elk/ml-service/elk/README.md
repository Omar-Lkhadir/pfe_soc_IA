# Intégration ELK — PFE IT6

## Ordre de mise en place

1. **Service ML** (ce dépôt, `integration/`) :
   ```
   pip install -r requirements.txt
   uvicorn commun.service:app --host 0.0.0.0 --port 8000
   ```
   Vérifier : `curl http://localhost:8000/health` doit répondre
   `{"status":"ok","sources_disponibles":["netflow","cicflowmeter","zeek"]}`.

2. **Template Elasticsearch** (à faire AVANT le premier event indexé, sinon
   les 5 champs `ml_*` seront mappés dynamiquement par ES au lieu du typage
   explicite ci-dessous) :
   ```
   curl -X PUT "http://localhost:9200/_index_template/nids-ml" \
     -H "Content-Type: application/json" -d @es-index-template.json
   ```

3. **Filebeat** : activer les modules nécessaires (`filebeat modules enable
   zeek`, config `netflow` selon la source réelle), pointer vers Logstash
   port 5044 (déjà le défaut du module Filebeat correspondant, rien à
   changer côté Filebeat lui-même).

4. **Logstash** : `logstash -f logstash-nids.conf`.

## ⚠️ Points non vérifiés en conditions réelles

Aucune instance Logstash/Elasticsearch/Filebeat n'était disponible pour
tester ce pipeline de bout en bout dans cet environnement — contrairement
au service ML (`commun/service.py`), testé avec de vraies requêtes HTTP
(voir le rapport d'intégration). À vérifier en premier lors du déploiement :

- **Comportement du filtre `http` de Logstash si le service ML est
  injoignable** : la doc officielle ne précise pas explicitement si
  l'event continue vers la sortie Elasticsearch ou est perdu. Le
  `logstash-nids.conf` fourni suppose qu'il continue (comportement standard
  des filtres Logstash) et ajoute `ml_status: service_indisponible` dans ce
  cas — à confirmer avec un test réel (couper le service ML et vérifier
  qu'un event brut arrive bien dans Elasticsearch).
- **Nom exact de `event.module`** tel qu'envoyé par Filebeat selon la
  version installée (utilisé par `commun/inference.py::detect_source()`
  comme méthode de détection prioritaire).
- **Compatibilité Filebeat/Zeek** : vérifiée contre le code source réel de
  Filebeat et une fixture officielle Elastic (voir `TEST/test7`), mais pas
  contre une vraie instance Filebeat en fonctionnement.
