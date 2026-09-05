# SOC IA — Pipeline NIDS temps réel sur stack ELK

Projet de Fin d'Études (PFE IT6) — intégration d'un modèle de Machine Learning de détection d'intrusion réseau (NIDS) au sein d'une stack ELK (Elasticsearch, Logstash, Kibana), avec capture réseau temps réel via Zeek.

## Description du projet

Ce projet transforme du trafic réseau brut en verdicts de détection d'intrusion exploitables par un analyste SOC, via la chaîne suivante :

```
Zeek (capture) → Filebeat → Logstash → ml-service (modèle ML) → Elasticsearch → Kibana
```

**Composants principaux :**

| Composant | Rôle |
|---|---|
| **Zeek** (`zeek-live` / `zeek-replay`) | Capture le trafic réseau et produit un journal de connexions (`conn.log`). `zeek-live` capture réellement sur l'interface de la machine surveillée ; `zeek-replay` rejoue un jeu de données labellisé (IoT-23) pour démonstration en l'absence de sonde réelle. **Ne jamais lancer les deux en même temps** (ils écrivent dans le même fichier). |
| **Filebeat** (`filebeat-nids`) | Lit `conn.log` en continu et l'envoie à Logstash. |
| **Logstash** | Décode le format Zeek, filtre le trafic IPv6 (bruit), détermine la source/destination réelle d'une attaque, appelle le modèle ML, indexe le résultat. |
| **ml-service** | API FastAPI exposant deux modèles scikit-learn : un `HistGradientBoosting` (palier 1 — détection binaire attaque/normal) et un `RandomForest` (palier 2 — catégorisation en 5 classes d'attaque). Expose aussi un import CSV ponctuel (`/predict_csv`). |
| **Elasticsearch** | Stocke les évènements enrichis (`nids-ml-*` pour le flux temps réel, `nids-import` pour les imports ponctuels). Sécurité (`xpack.security`) activée. |
| **Kibana** | Deux tableaux de bord (temps réel + analyse d'imports) et une règle d'alerte (`nids-alerts`) déclenchée sur détection d'attaque. |

## Prérequis

- Ubuntu (ou toute distribution Linux avec Docker)
- [Docker](https://docs.docker.com/engine/install/ubuntu/) et [Docker Compose plugin](https://docs.docker.com/compose/install/linux/) (`docker compose version` doit fonctionner)
- Au moins 4 Go de RAM disponibles pour Docker (Elasticsearch + Logstash + ml-service + Kibana)
- `git`

Installation rapide de Docker sur Ubuntu, si nécessaire :

```bash
sudo apt update
sudo apt install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo usermod -aG docker $USER   # puis se déconnecter/reconnecter, ou `newgrp docker`
```

## Installation et lancement

### 1. Récupérer le projet

```bash
git clone https://github.com/Omar-Lkhadir/pfe_soc_IA.git
cd pfe_soc_IA/docker-elk
```

### 2. Configurer les mots de passe

```bash
cp .env.example .env
```

Éditer `.env` et remplacer chaque `changeme` par un mot de passe fort (au minimum `ELASTIC_PASSWORD`, `LOGSTASH_INTERNAL_PASSWORD`, `KIBANA_SYSTEM_PASSWORD`). **Ce fichier `.env` ne doit jamais être commité** (il est déjà dans `.gitignore`).

### 3. Démarrer la stack de base (Elasticsearch, Logstash, Kibana)

```bash
docker compose up -d
```

Attendre qu'Elasticsearch et Kibana soient prêts (1 à 2 minutes) :

```bash
curl -s http://localhost:9200 -o /dev/null -w "Elasticsearch: %{http_code}\n"
curl -s http://localhost:5601/api/status -o /dev/null -w "Kibana: %{http_code}\n"
```

### 4. Provisionner les comptes internes (obligatoire, une seule fois)

```bash
docker compose up setup
```

Ce service crée les utilisateurs internes (`logstash_internal`, `kibana_system`...) et les rôles (dont l'accès en écriture sur `nids-*`) à partir des mots de passe définis dans `.env`. **À relancer après toute modification de la sécurité Elasticsearch** (ex. changement de mot de passe), mais pas plus — cela réinitialise aussi les rôles à leurs valeurs par défaut.

Puis redémarrer les services qui dépendent de l'authentification :

```bash
docker compose restart logstash kibana ml-service
```

### 5. Démarrer la capture réseau

**Option A — Démonstration avec un jeu de données réel (par défaut, aucune sonde nécessaire) :**

```bash
docker compose up -d zeek-replay filebeat-nids
```

**Option B — Capture réelle sur l'interface de la machine surveillée** (nécessite d'adapter `ZEEK_IFACE` dans `.env` à l'interface réseau réelle) :

```bash
docker compose stop zeek-replay   # ne jamais cumuler avec zeek-replay
docker compose --profile live up -d zeek-live filebeat-nids
```

### 6. Importer les tableaux de bord Kibana

```bash
curl -u elastic:VOTRE_MOT_DE_PASSE_ELASTIC -H "kbn-xsrf: true" \
  -F "file=@kibana/dashboards/nids-dashboards.ndjson;type=application/ndjson" \
  "http://localhost:5601/api/saved_objects/_import?overwrite=true"
```

### 7. Activer la règle d'alerte (une seule fois)

```bash
KIBANA_URL=http://localhost:5601 ./kibana/dashboards/setup_alerting.sh
```

⚠️ Ne pas relancer ce script si la règle existe déjà (il recréerait des connecteurs en double) — vérifier d'abord avec `curl -u elastic:MOT_DE_PASSE -H "kbn-xsrf: true" http://localhost:5601/api/alerting/rules/_find`.

### 8. Accéder aux interfaces

| Interface | URL |
|---|---|
| Kibana (dashboards SOC) | http://localhost:5601 |
| Dashboard temps réel | http://localhost:5601/app/dashboards#/view/nids-dashboard |
| Dashboard imports | http://localhost:5601/app/dashboards#/view/nids-import-dashboard |
| Import manuel de données (ml-service) | http://localhost:8000 |

Identifiant Kibana : `elastic` / le mot de passe défini dans `.env`.

## Arrêter le projet

```bash
docker compose down          # arrête tout, conserve les données (volumes)
docker compose down -v       # arrête tout et supprime aussi les données
```

## Dépannage rapide

| Symptôme | Cause probable | Solution |
|---|---|---|
| Erreur 401 / `security_exception` dans les logs Logstash | Le service `setup` n'a pas été (re)lancé après activation de la sécurité | `docker compose up setup` puis redémarrer `logstash`/`kibana`/`ml-service` |
| Adresses source/destination incorrectes dans le dashboard | L'IP de la machine surveillée a changé (DHCP) | Mettre à jour la liste `victime_ips` dans `logstash/pipeline/nids.conf` et le dictionnaire `logstash/pipeline/attacker_hosts.yml` |
| Table "Alertes actives" figée | Clé API de la règle d'alerte invalidée (ex. après activation de la sécurité) | Régénérer via `POST /api/alerting/rule/{id}/_update_api_key` (voir l'ID avec `_find` ci-dessus) |
| `conn.log` mélange deux formats de données incohérents | `zeek-live` et `zeek-replay` tournent en même temps | N'en garder qu'un seul actif (`docker compose stop zeek-replay` ou `zeek-live`) |

## Structure du dépôt

```
docker-elk/
├── docker-compose.yml            # Orchestration de tous les services
├── .env.example                  # Modèle de configuration (mots de passe)
├── elasticsearch/                # Config Elasticsearch
├── logstash/
│   ├── pipeline/nids.conf        # Pipeline NIDS (cœur du traitement)
│   ├── pipeline/attacker_hosts.yml
│   └── config/pipelines.yml
├── filebeat-nids/                # Filebeat dédié au flux NIDS
├── zeek-replay/                  # Rejoue un dataset Zeek labellisé (démo)
├── zeek-live/                    # Politique Zeek pour capture réelle
├── ml-service/                   # API FastAPI + modèles scikit-learn (.pkl)
├── kibana/
│   ├── config/kibana.yml
│   └── dashboards/               # Dashboards (ndjson) + script d'alerting
└── setup/                        # Provisionnement des comptes/rôles Elasticsearch
```

## Crédits

Basé sur [deviantony/docker-elk](https://github.com/deviantony/docker-elk), adapté et étendu dans le cadre de ce PFE avec le pipeline NIDS, le service de Machine Learning (`ml-service`) et les tableaux de bord SOC.
