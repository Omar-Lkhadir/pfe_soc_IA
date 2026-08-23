"""
Schéma canonique pour le HIDS (détection d'intrusion basée sur logs hôte) --
projet séparé du NIDS (commun/canonical_schema.py). Signal fondamentalement
différent : le NIDS lit des FLUX réseau (ports/octets/paquets/durée par
connexion) ; le HIDS lit des ÉVÉNEMENTS D'AUTHENTIFICATION individuels
(une ligne = une tentative de connexion SSH, ou une invocation sudo) --
aucune de ces lignes seule ne suffit à distinguer une attaque d'un usage
normal. Le signal d'attaque n'apparaît qu'en AGRÉGEANT plusieurs
événements sur une fenêtre de temps (ex. 20 échecs SSH en 10 secondes
depuis la même IP = brute-force ; un seul échec = un utilisateur qui s'est
trompé de mot de passe).

Champs source : module Filebeat `system` (fileset auth), vérifiés contre
la doc officielle Elastic (elastic.co/guide/en/beats/filebeat/current/
exported-fields-system.html) le 2026-08-23 -- pas devinés. Filebeat fait
déjà le grok en interne : ces champs arrivent structurés dans Logstash,
pas de re-parsing de texte brut nécessaire côté HIDS.

  SSH   : source.ip, source.port, user.name,
          system.auth.ssh.event ('Accepted'/'Invalid'/'Failed'),
          system.auth.ssh.method ('password'/'publickey'),
          system.auth.ssh.geoip.country_iso_code
  SUDO  : user.name (qui lance sudo), system.auth.sudo.user (cible),
          system.auth.sudo.command, system.auth.sudo.error (vide si OK)

Statut : SCHÉMA UNIQUEMENT à ce stade -- pas encore entraîné, pas de
données étiquetées réelles branchées (cf. discussion projet : Cowrie/
Kaggle nécessite un compte, AIT-LDS téléchargeable mais pas focalisé
SSH pur, génération de trafic réel via hydra/medusa sur un serveur de
test envisagée). Les fonctions d'agrégation ci-dessous sont testables
dès maintenant sur des événements synthétiques ; l'entraînement attend
la décision sur la source de données.
"""

import numpy as np
import pandas as pd

# ============================================================
# PALIER 1 — features agrégées par fenêtre, source SSH (par source.ip)
# ============================================================
SSH_WINDOW_FEATURES = [
    'nb_tentatives',              # total d'événements SSH dans la fenêtre
    'nb_echecs',                  # ssh.event in {Failed, Invalid}
    'nb_reussites',                # ssh.event == Accepted
    'ratio_echec',                 # nb_echecs / nb_tentatives
    'nb_utilisateurs_distincts',   # user.name distincts essayés
    'nb_methodes_distinctes',      # ssh.method distincts (password/publickey)
    'reussite_apres_echecs',       # bool : >=1 Accepted précédé d'échecs, même IP
    'duree_fenetre_sec',           # timestamp(dernier) - timestamp(premier)
    'intervalle_moyen_sec',        # durée_fenetre / (nb_tentatives-1), 0 si 1 seule tentative
]

# ============================================================
# PALIER 1 — features agrégées par fenêtre, source SUDO (par user.name)
# ============================================================
SUDO_WINDOW_FEATURES = [
    'nb_commandes_sudo',           # total d'invocations sudo dans la fenêtre
    'nb_echecs_sudo',               # sudo.error non vide
    'ratio_echec_sudo',
    'nb_commandes_distinctes',      # sudo.command distincts
    'nb_cibles_distinctes',         # sudo.user (cible) distincts
    'ratio_cible_root',             # fraction des commandes visant root
    'duree_fenetre_sec',
]

CATEGORIES_HIDS = ['ssh_bruteforce', 'sudo_abuse']  # extensible


def agreger_fenetre_ssh(df_events: pd.DataFrame, fenetre: str = '5min') -> pd.DataFrame:
    """Transforme des événements SSH individuels (1 ligne = 1 tentative,
    colonnes attendues : timestamp, source_ip, user_name, ssh_event,
    ssh_method -- noms Python, dérivés des champs Filebeat @timestamp/
    source.ip/user.name/system.auth.ssh.event/system.auth.ssh.method)
    en features agrégées par (source_ip, fenêtre de temps).

    `fenetre` : chaîne pandas resample (ex. '5min', '1min') -- fenêtre
    fixe pour une v1 simple ; une fenêtre glissante (plus réactive mais
    plus coûteuse) est une amélioration possible, pas nécessaire pour
    valider l'approche.
    """
    df = df_events.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['est_echec'] = df['ssh_event'].isin(['Failed', 'Invalid'])
    df['est_reussite'] = df['ssh_event'] == 'Accepted'

    lignes = []
    for (source_ip, periode), groupe in df.set_index('timestamp').groupby(
        ['source_ip', pd.Grouper(freq=fenetre)]
    ):
        if len(groupe) == 0:
            continue
        nb_tentatives = len(groupe)
        nb_echecs = int(groupe['est_echec'].sum())
        nb_reussites = int(groupe['est_reussite'].sum())
        ts = groupe.index.sort_values()
        duree = (ts[-1] - ts[0]).total_seconds() if len(ts) > 1 else 0.0
        # reussite précédée d'au moins un échec de la même IP dans la fenêtre
        premiere_reussite_idx = groupe.index[groupe['est_reussite']].min() if nb_reussites > 0 else None
        reussite_apres_echecs = bool(
            premiere_reussite_idx is not None and
            (groupe.loc[groupe.index < premiere_reussite_idx, 'est_echec'].sum() > 0)
        )
        lignes.append({
            'source_ip': source_ip,
            'fenetre_debut': periode,
            'nb_tentatives': nb_tentatives,
            'nb_echecs': nb_echecs,
            'nb_reussites': nb_reussites,
            'ratio_echec': nb_echecs / nb_tentatives,
            'nb_utilisateurs_distincts': groupe['user_name'].nunique(),
            'nb_methodes_distinctes': groupe['ssh_method'].nunique(),
            'reussite_apres_echecs': reussite_apres_echecs,
            'duree_fenetre_sec': duree,
            'intervalle_moyen_sec': duree / (nb_tentatives - 1) if nb_tentatives > 1 else 0.0,
        })
    return pd.DataFrame(lignes)


def agreger_fenetre_sudo(df_events: pd.DataFrame, fenetre: str = '1h') -> pd.DataFrame:
    """Idem agreger_fenetre_ssh mais pour les invocations sudo (1 ligne =
    1 commande sudo). Colonnes attendues : timestamp, user_name,
    sudo_user (cible), sudo_command, sudo_error (chaîne vide/NaN si OK)."""
    df = df_events.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['est_echec'] = df['sudo_error'].notna() & (df['sudo_error'].astype(str).str.strip() != '')

    lignes = []
    for (user_name, periode), groupe in df.set_index('timestamp').groupby(
        ['user_name', pd.Grouper(freq=fenetre)]
    ):
        if len(groupe) == 0:
            continue
        nb_commandes = len(groupe)
        nb_echecs = int(groupe['est_echec'].sum())
        ts = groupe.index.sort_values()
        duree = (ts[-1] - ts[0]).total_seconds() if len(ts) > 1 else 0.0
        lignes.append({
            'user_name': user_name,
            'fenetre_debut': periode,
            'nb_commandes_sudo': nb_commandes,
            'nb_echecs_sudo': nb_echecs,
            'ratio_echec_sudo': nb_echecs / nb_commandes,
            'nb_commandes_distinctes': groupe['sudo_command'].nunique(),
            'nb_cibles_distinctes': groupe['sudo_user'].nunique(),
            'ratio_cible_root': (groupe['sudo_user'] == 'root').mean(),
            'duree_fenetre_sec': duree,
        })
    return pd.DataFrame(lignes)


def finalize_dtypes_hids(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """Force les features en float32 sans NaN -- même discipline que
    commun/canonical_schema.py::finalize_dtypes (palier 1 NIDS)."""
    df = df.copy()
    for c in features:
        df[c] = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    return df
