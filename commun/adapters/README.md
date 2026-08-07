# Adaptateurs de format de flux

Les modèles (Isolation Forest, Random Forest) ne voient jamais un format brut
— seulement le **schéma canonique** défini dans `commun/canonical_schema.py`.
Chaque outil d'export de flux (NetFlow, CICFlowMeter, Zeek, Suricata, ...) a
son propre adaptateur, dont le seul rôle est de convertir ses colonnes vers ce
schéma commun. **Ajouter un nouveau format = écrire un nouvel adaptateur,
jamais retoucher le schéma ni les modèles.**

## Formats couverts

| Adaptateur | Statut | Palier 2 (IAT, taille paquet) |
|---|---|---|
| `netflow_adapter.py` | Entraîné + validé (BigFlow-NIDS-V2) | Disponible |
| `cicflowmeter_adapter.py` | Entraîné + validé (CICIDS2017) | Disponible |
| `zeek_adapter.py` | Testé unitairement, **pas entraîné** (pas de données réelles) | Absent (`NaN`) |
| `suricata_adapter.py` | Testé unitairement, **pas entraîné** (pas de données réelles) | Absent (`NaN`) |

"Pas entraîné" ne veut pas dire "ne marchera pas" : le mapping de colonnes est
basé sur la documentation officielle et vérifié par tests unitaires
(`commun/tests/test_adapters.py`) sur des lignes synthétiques. Ça veut dire
que la performance réelle des modèles sur ce format n'a jamais été mesurée
faute de données. Random Forest (`HistGradientBoostingClassifier`) tolère
nativement l'absence du palier 2 — Isolation Forest n'en a de toute façon pas
besoin (palier 1 uniquement).

## Le schéma canonique en deux paliers

- **Palier 1** (`canonical_schema.TIER1_FEATURES`, 17 colonnes) : ports/
  protocole bucketisés, octets/paquets par direction, durée, 6 flags TCP
  booléens, 4 agrégats dérivés. **Obligatoire, jamais de `NaN`.** Utilisé par
  les deux modèles.
- **Palier 2** (`canonical_schema.TIER2_FEATURES`, 6 colonnes) : taille de
  paquet min/max, moyennes/écarts-types des inter-arrivées (IAT) dans les
  deux sens. **Optionnel** — un adaptateur qui ne peut pas le remplir met
  `NaN`, jamais une constante fabriquée. Utilisé uniquement par Random Forest.

Pourquoi cette scission : le palier 1 est ce que n'importe quel outil de flux
biflow sérieux expose (vérifié sur NetFlow, CICFlowMeter, Zeek, Suricata —
voir le plan de conception pour le détail des sources). Le palier 2 est plus
riche mais spécifique aux outils qui calculent des statistiques par flux
(NetFlow, CICFlowMeter) ; Zeek/Suricata en sortie standard ne l'exposent pas.

## Ajouter un adaptateur pour un nouveau format

1. Identifier dans la doc du format les colonnes équivalentes au palier 1 :
   port destination, protocole (numérique OU chaîne — `bucket_protocol()`
   gère les deux), octets/paquets par direction, durée, et un moyen de
   dériver au moins SYN/ACK/FIN/RST (bitmask, compteurs, ou champ d'état type
   `history`/`conn_state`).
2. Vérifier les **unités** avant d'écrire le mapping (piège classique : durée
   et IAT en microsecondes vs millisecondes vs secondes selon l'outil — cf.
   le bug corrigé dans `cicflowmeter_adapter.py`, où les champs IAT n'avaient
   pas été convertis alors que la durée l'était).
3. Vérifier la **convention de comptage des octets** : payload applicatif
   seul, ou octets IP totaux (en-têtes inclus) ? NetFlow/CICFlowMeter comptent
   les octets IP totaux — pour Zeek par exemple, ça veut dire utiliser
   `orig_ip_bytes`/`resp_ip_bytes`, pas `orig_bytes`/`resp_bytes`.
4. Sous-classer `BaseFlowAdapter`, implémenter `to_canonical(df_raw)` :
   construire les 7 colonnes brutes du palier 1 + les 6 flags + appeler
   `canonical_schema.add_derived_features()`, puis remplir le palier 2 si
   disponible sinon `np.nan`, puis les colonnes méta (`Label_binaire`,
   `Attack_brut`, `source`). Finir par `canonical_schema.finalize_dtypes()`.
5. Écrire un test unitaire sur 2-3 lignes synthétiques (cf.
   `commun/tests/test_adapters.py`) : au minimum, vérifier que le contrat de
   `BaseFlowAdapter.validate()` passe et que la conversion d'unité de durée
   est correcte.
6. Si de vraies données pour ce format deviennent disponibles : régénérer les
   arrays canoniques (comme `commun/build_training_data.py` le fait pour
   NetFlow/CICFlowMeter), les ajouter au jeu d'entraînement combiné, et
   ré-entraîner — même procédure que celle utilisée pour ajouter CICFlowMeter
   à cette version des modèles.

## Cas particulier connu : formats unidirectionnels (ex. AWS VPC Flow Logs)

NetFlow, CICFlowMeter, Zeek et Suricata produisent tous un enregistrement
**biflow** (les deux sens d'une connexion dans une seule ligne). AWS VPC Flow
Logs (format par défaut) produit un enregistrement **par direction** —
`IN_BYTES`/`OUT_BYTES` exigeraient de corréler deux lignes (5-tuple inversé,
même fenêtre temporelle) avant de pouvoir remplir le schéma canonique. Ce
n'est pas qu'un renommage de colonnes comme les 4 adaptateurs ci-dessus : ça
demande une étape d'agrégation biflow en amont de l'adaptateur. Pas implémenté
faute de données pour le valider — à traiter comme un adaptateur à part
entière (`vpc_flow_logs_adapter.py` + une fonction de corrélation biflow) le
jour où c'est nécessaire.
