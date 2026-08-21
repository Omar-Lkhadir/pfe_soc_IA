# Soc_IA

PFE IT6 — SOC Intelligent basé sur ELK + IA.

Pipeline de détection d'intrusion réseau à deux paliers, généraliste sur plusieurs formats de flux (NetFlow, CICFlowMeter, Zeek, Suricata) via un pattern adaptateur et un schéma canonique commun (`commun/canonical_schema.py`).

- **`hist_gradient_boosting/`** — Palier 1 : détection normal/attaque (`HistGradientBoostingClassifier` supervisé, un seul modèle partagé entre les 3 sources, pondéré par source). Reçoit chaque événement en premier.
- **`random_forest/`** — Palier 2 : catégorisation des attaques détectées par le palier 1 en 5 classes (`brute_force`, `port_scanning`, `suspicious_activity`, `intrusion_reseau`, `acces_non_autorise`), même famille d'algorithme (`HistGradientBoostingClassifier`).
- **`commun/`** — schéma canonique, adaptateurs par format de flux, module d'inférence (`inference.py`, point d'intégration ELK/Logstash).

Chaque dossier de modèle contient `scripts/` (entraînement), `models/` (artefacts) et `results/` (rapports et graphiques). Les données brutes et les modèles volumineux sont exclus du dépôt (voir `.gitignore`) et se régénèrent via les scripts d'entraînement.

Validation externe sur données réelles jamais entraînées : voir `TEST/test1` à `TEST/test6` (hors dépôt).
