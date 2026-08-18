"""
Point d'intégration UNIQUE pour ELK/Logstash : une seule classe, NIDSPredictor,
qui charge tous les modèles (3x Isolation Forest par source + 1x Random Forest
partagé) et expose une seule méthode predict(event, source=None).

Routage, jamais de fusion : pour un événement donné, un seul modèle IF est
utilisé (celui de sa source), jamais un vote/moyenne entre plusieurs.

Détection de la source :
  - explicite (paramètre `source`, ou clé `event.module` de l'événement —
    c'est ce que Filebeat renseigne nativement selon son module) ;
  - à défaut, détection par signature de champs (utile en test / hors ELK).

Gestion des cas limites (jamais de perte silencieuse d'événement) :
  - palier 1 complet                    -> ml_status = "ok"
  - palier 1 partiellement incomplet
    (< SEUIL_IMPUTATION manquant)       -> imputation par la médiane de LA
                                            MÊME source -> ml_status =
                                            "scoring_degrade"
  - source inconnue OU palier 1 trop
    incomplet (>= SEUIL_IMPUTATION)     -> AUCUN scoring forcé -> ml_status =
                                            "format_non_reconnu" (l'événement
                                            n'est jamais perdu, juste non noté)
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

MD4_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if MD4_DIR not in sys.path:
    sys.path.insert(0, MD4_DIR)

from commun import canonical_schema as cs
from commun.adapters.netflow_adapter import NetFlowAdapter
from commun.adapters.cicflowmeter_adapter import CICFlowMeterAdapter
from commun.adapters.zeek_adapter import ZeekAdapter
from commun.adapters.suricata_adapter import SuricataAdapter

SEUIL_IMPUTATION = 0.30  # au-delà, on refuse de scorer plutôt que d'imputer trop de champs

ADAPTERS = {
    'netflow': NetFlowAdapter,
    'cicflowmeter': CICFlowMeterAdapter,
    'zeek': ZeekAdapter,
    'suricata': SuricataAdapter,
}

# Signatures de champs (fallback si `source`/`event.module` absent) : au moins
# un des jeux de colonnes ci-dessous doit être présent dans l'événement brut.
SIGNATURES = {
    'netflow': [{'L4_DST_PORT', 'IN_BYTES', 'PROTOCOL'}],
    'cicflowmeter': [{'Destination Port', 'Flow Duration'}, {'Dst Port', 'Flow Duration'}],
    'zeek': [{'id.orig_p', 'id.resp_p', 'proto'}],
    'suricata': [{'flow', 'proto'}, {'pkts_toserver', 'bytes_toserver'}],
}


def detect_source(raw_event: dict) -> str | None:
    """Détecte la source d'un événement brut. Priorité : event.module explicite,
    puis signature de champs. Retourne None si aucune source connue ne correspond."""
    module = raw_event.get('event.module') or raw_event.get('source')
    if module in ADAPTERS:
        return module
    champs = set(raw_event.keys())
    for source, signatures in SIGNATURES.items():
        for sig in signatures:
            if sig.issubset(champs):
                return source
    return None


class NIDSPredictor:
    """Un seul point d'entrée pour ELK : predict(event, source=None) -> dict.
    Instancier UNE FOIS (charge tous les modèles en mémoire), puis appeler
    predict() par événement (ou predict_batch() pour un DataFrame entier —
    beaucoup plus efficace pour du traitement en masse / nos propres tests)."""

    def __init__(self, md4_dir: str = MD4_DIR):
        self.if_models = {}
        if_dir = os.path.join(md4_dir, 'isolation_forest', 'models')
        for source in ['netflow', 'cicflowmeter', 'zeek']:
            dossier = os.path.join(if_dir, source)
            if not os.path.exists(os.path.join(dossier, 'model.pkl')):
                continue
            medianes_path = os.path.join(dossier, 'medianes_palier1.pkl')
            self.if_models[source] = dict(
                model=joblib.load(os.path.join(dossier, 'model.pkl')),
                scaler=joblib.load(os.path.join(dossier, 'scaler.pkl')),
                seuil=joblib.load(os.path.join(dossier, 'seuil_optimal.pkl'))['seuil_optimal'],
                medianes=joblib.load(medianes_path) if os.path.exists(medianes_path) else None,
            )

        rf_dir = os.path.join(md4_dir, 'random_forest', 'models')
        self.rf_model = joblib.load(os.path.join(rf_dir, 'model.pkl'))
        self.rf_label_encoder = joblib.load(os.path.join(rf_dir, 'label_encoder.pkl'))
        self.rf_seuil_confiance = joblib.load(os.path.join(rf_dir, 'seuil_confiance.pkl'))['seuil_confiance']
        self.rf_features = joblib.load(os.path.join(rf_dir, 'features.pkl'))['features']

        self._adapters = {name: cls() for name, cls in ADAPTERS.items()}
        # Colonnes brutes attendues par adaptateur, quand exposées (RAW_COLS /
        # COLS_VOULUES) -> permet de compléter par NaN une colonne ENTIÈREMENT
        # absente de l'événement (pas juste une valeur nulle sur une ligne),
        # que les adaptateurs eux-mêmes ne savent pas tolérer (KeyError sinon).
        self._raw_cols = {}
        for name, cls in ADAPTERS.items():
            mod = sys.modules[cls.__module__]
            cols = getattr(mod, 'RAW_COLS', None) or getattr(mod, 'COLS_VOULUES', None)
            if cols:
                self._raw_cols[name] = list(cols)

    def _adapt(self, source: str, raw_df: pd.DataFrame) -> pd.DataFrame | None:
        attendu = self._raw_cols.get(source)
        if attendu:
            manquantes = [c for c in attendu if c not in raw_df.columns]
            if manquantes:
                raw_df = raw_df.reindex(columns=list(raw_df.columns) + manquantes)
                # 'Label' est un champ de VÉRITÉ TERRAIN (présent dans les jeux
                # d'entraînement/test étiquetés), jamais fourni par un flux réel
                # en production. cicflowmeter_adapter.py écarte toute ligne dont
                # Label est vide (utile pour nettoyer des CSV d'entraînement
                # corrompus) -> le laisser à NaN supprimerait TOUTES les lignes
                # en inférence réelle. Placeholder neutre pour éviter ça, sans
                # toucher à l'adaptateur validé.
                if 'Label' in manquantes:
                    raw_df['Label'] = 'Unknown'
        try:
            return self._adapters[source].to_canonical(raw_df)
        except Exception:
            return None  # adaptation impossible -> traité comme format non reconnu

    def predict_batch(self, raw_df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Score un DataFrame entier, connu comme appartenant à UNE SEULE source.
        Retourne un DataFrame avec les colonnes ml_status, is_attack,
        anomaly_score, attack_category, confidence (aligné sur raw_df.index)."""
        n = len(raw_df)
        out = pd.DataFrame(index=raw_df.index)
        out['ml_status'] = 'format_non_reconnu'
        out['is_attack'] = pd.NA
        out['anomaly_score'] = np.nan
        out['attack_category'] = pd.NA
        out['confidence'] = np.nan

        if source not in self.if_models:
            return out  # source inconnue du système -> tout en format_non_reconnu

        # --- complétude des colonnes BRUTES (les adaptateurs remplissent déjà
        # le palier 1 canonique par 0 via finalize_dtypes -> la détection doit
        # se faire AVANT l'adaptation, sur les colonnes brutes attendues,
        # sinon aucune valeur manquante n'est plus jamais visible) ---
        colonnes_label = {'Label', 'Attack', 'detailed-label'}
        cols_features_attendues = [c for c in self._raw_cols.get(source, []) if c not in colonnes_label]
        if cols_features_attendues:
            presentes = [c for c in cols_features_attendues if c in raw_df.columns]
            sous_df = raw_df[presentes] if presentes else pd.DataFrame(index=raw_df.index)
            manquantes_par_col = len(cols_features_attendues) - len(presentes)
            taux_manquant = (sous_df.isna().sum(axis=1) + manquantes_par_col) / len(cols_features_attendues)
        else:
            taux_manquant = pd.Series(0.0, index=raw_df.index)
        trop_incomplet = taux_manquant >= SEUIL_IMPUTATION
        a_impute = (taux_manquant > 0) & (~trop_incomplet)

        canon = self._adapt(source, raw_df)
        if canon is None:
            return out  # adaptation impossible -> tout en format_non_reconnu
        modele = self.if_models[source]

        # Événements dégradés : le 0 déjà mis par l'adaptateur (finalize_dtypes)
        # est remplacé par la médiane de CETTE source (jamais une constante
        # fabriquée entre sources), un peu plus large que le strict nécessaire
        # (tout le palier 1 de la ligne, pas juste les champs affectés — la
        # ligne reste marquée "scoring_degrade", donc explicitement moins fiable).
        if modele['medianes'] and a_impute.any():
            for col, med in modele['medianes'].items():
                canon.loc[a_impute, col] = med

        idx_scorable = ~trop_incomplet
        if idx_scorable.any():
            X = canon.loc[idx_scorable, cs.TIER1_FEATURES].values.astype(np.float32)
            X_scaled = modele['scaler'].transform(X).astype(np.float32)
            scores = modele['model'].decision_function(X_scaled)
            y_pred = (scores < modele['seuil']).astype(bool)

            out.loc[idx_scorable, 'ml_status'] = np.where(a_impute.loc[idx_scorable], 'scoring_degrade', 'ok')
            out.loc[idx_scorable, 'is_attack'] = y_pred
            out.loc[idx_scorable, 'anomaly_score'] = scores

        # --- Random Forest : uniquement sur les événements scorés ET détectés attaque ---
        mask_rf = idx_scorable & (out['is_attack'] == True)  # noqa: E712
        if mask_rf.any():
            X_rf = canon.loc[mask_rf, self.rf_features].values.astype(np.float32)
            pred_enc = self.rf_model.predict(X_rf)
            proba = self.rf_model.predict_proba(X_rf).max(axis=1)
            out.loc[mask_rf, 'attack_category'] = self.rf_label_encoder.inverse_transform(pred_enc)
            out.loc[mask_rf, 'confidence'] = proba

        return out

    def predict(self, raw_event: dict, source: str | None = None) -> dict:
        """Une seule ligne. Pratique pour un event Logstash unitaire ; pour du
        volume, préférer predict_batch()."""
        source = source or detect_source(raw_event)
        if source is None:
            return dict(ml_status='format_non_reconnu', is_attack=None, anomaly_score=None,
                        attack_category=None, confidence=None, source=None)
        raw_df = pd.DataFrame([raw_event])
        res = self.predict_batch(raw_df, source).iloc[0].to_dict()
        res['source'] = source
        return res
