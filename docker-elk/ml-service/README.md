# modele_IA — Intégration ELK

PFE IT6 — SOC Intelligent basé sur ELK + IA. Ce dépôt contient tout ce qui
est nécessaire pour déployer le pipeline de détection d'intrusion réseau
(entraîné dans le dépôt principal `Soc_IA`) dans une stack ELK réelle.

Pipeline à deux paliers, généraliste sur plusieurs formats de flux
(NetFlow, CICFlowMeter, Zeek, Suricata) via un pattern adaptateur et un
schéma canonique commun.

## Structure

```
commun/                             code (adaptateurs, schéma canonique, inférence)
├── canonical_schema.py             source de vérité : features Palier 1 / Palier 2
├── mapping_attaques.py             mapping des 5 catégories d'attaque
├── inference.py                    NIDSPredictor — point d'entrée unique
├── service.py                      wrapper FastAPI (appelé par Logstash)
├── adapters/                       un module par format de flux
└── tests/                          tests de non-régression (adaptateurs + inférence)

hist_gradient_boosting/models/      Palier 1 — normal/attaque (modèle partagé, pondéré par source)
random_forest/models/               Palier 2 — catégorisation en 5 classes

elk/
├── logstash-nids.conf              pipeline Filebeat -> service ML -> Elasticsearch
├── es-index-template.json          mapping des champs de sortie (ml_status, is_attack, ...)
└── README.md                       ordre de déploiement + points à vérifier en conditions réelles

requirements.txt                    versions exactes (scikit-learn 1.4.2 notamment,
                                     nécessaire pour charger les .pkl)
```

## Démarrage rapide

```
pip install -r requirements.txt
uvicorn commun.service:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

Puis voir `elk/README.md` pour la suite (template Elasticsearch, Filebeat, Logstash).

## Statut de validation

- **Service ML** (`commun/service.py`) : testé avec de vraies requêtes HTTP
  (`/health`, `/predict` sur événement normal et sur une attaque réelle
  étiquetée) — fonctionne.
- **Compatibilité Filebeat/Zeek** : vérifiée contre le code source réel de
  Filebeat (module zeek) et une fixture officielle Elastic — voir le
  dépôt principal `Soc_IA`, dossier `TEST/test7`.
- **Config Logstash/Elasticsearch** (`elk/`) : syntaxe vérifiée contre la
  documentation officielle, mais **jamais exécutée contre une vraie
  instance** (aucune disponible lors de la rédaction) — voir les points
  de vigilance listés dans `elk/README.md` avant mise en prod.
- **Suricata** : adaptateur présent (`commun/adapters/suricata_adapter.py`)
  mais jamais entraîné/validé sur données réelles — à ne pas activer en
  production tant que ça reste vrai.

Dépôt d'entraînement complet (données, scripts, 7 campagnes de test sur
données réelles) : voir le dépôt `Soc_IA`.
