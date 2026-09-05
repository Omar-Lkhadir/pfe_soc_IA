"""
Point d'intégration UNIQUE pour ELK/Logstash : une seule classe, NIDSPredictor,
qui charge tous les modèles (1x palier 1 supervisé partagé entre les 3
sources + 1x Random Forest partagé) et expose une seule méthode
predict(event, source=None).

Palier 1 : depuis hist_gradient_boosting/scripts/entrainement.py, un
HistGradientBoostingClassifier supervisé (même famille que Random Forest)
a remplacé l'ancien Isolation Forest non-supervisé (3 modèles, un par
source, conservés intacts dans hist_gradient_boosting/models/{source}/ pour
référence/rollback mais plus utilisés en production). Compromis assumé :
ce nouveau palier 1 ne détecte que des patterns ressemblant aux attaques
vues à l'entraînement, il perd la capacité "zero-day" de l'ancien IF —
décision utilisateur explicite, motivée par un gain de F1 massif (60-85%
-> 96-99% selon la source) sur trafic connu.

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
  - source inconnue/non validée OU
    palier 1 trop incomplet
    (>= SEUIL_IMPUTATION)               -> AUCUN scoring forcé -> ml_status =
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
    'zeek': [{'id.orig_p', 'id.resp_p', 'proto'},  # conn.log natif
             {'destination.port', 'network.transport', 'zeek.connection.history'}],  # via Filebeat/ECS
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
        if_dir = os.path.join(md4_dir, 'hist_gradient_boosting', 'models')
        self.palier1 = dict(
            model=joblib.load(os.path.join(if_dir, 'model.pkl')),
            seuil=joblib.load(os.path.join(if_dir, 'seuil_optimal.pkl'))['seuil_optimal'],
        )
        # Médianes par source (imputation en mode dégradé) : statistique
        # descriptive indépendante du modèle de scoring -> toujours calculée
        # par source, même si le palier 1 est maintenant un modèle partagé.
        # Seules les sources ayant des médianes pré-calculées (validées sur
        # données réelles) sont scorables ; suricata (adaptateur jamais
        # entraîné/validé faute de données réelles) reste exclu ici.
        self.medianes_par_source = {}
        for source in ['netflow', 'cicflowmeter', 'zeek']:
            medianes_path = os.path.join(if_dir, 'medianes_palier1', f'{source}.pkl')
            if os.path.exists(medianes_path):
                self.medianes_par_source[source] = joblib.load(medianes_path)

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
        # Sous-ensemble de _raw_cols[source] qui n'alimente QUE le palier 2
        # (RAW_COLS_TIER2_ONLY, quand l'adaptateur le déclare) -> à exclure du
        # taux d'incomplétude qui décide si le palier 1 est calculable (cf.
        # predict_batch). Sans ça, un fichier réel où seul le palier 2 manque
        # intégralement (ex. dataset public NF-BoT-IoT, palier 1 100%
        # complet) était rejeté en bloc -- palier 1 pourtant parfaitement
        # calculable, jamais de constante fabriquée pour le palier 2 (comme
        # déjà le cas pour zeek).
        self._raw_cols_tier2_only = {}
        for name, cls in ADAPTERS.items():
            mod = sys.modules[cls.__module__]
            cols = getattr(mod, 'RAW_COLS', None) or getattr(mod, 'COLS_VOULUES', None)
            if cols:
                self._raw_cols[name] = list(cols)
            tier2_only = getattr(mod, 'RAW_COLS_TIER2_ONLY', None)
            if tier2_only:
                self._raw_cols_tier2_only[name] = set(tier2_only)

    def _adapt(self, source: str, raw_df: pd.DataFrame) -> pd.DataFrame | None:
        attendu = self._raw_cols.get(source)
        if attendu:
            # Certains CSV reels (ex. export CICIDS2017 depuis Hugging Face)
            # ont des espaces en debut de nom de colonne (' Destination Port').
            # to_canonical() les nettoie en interne, mais AVANT ce nettoyage
            # cette detection croyait la colonne "manquante" -> la reindexait
            # en double (NaN) -> une fois strip() applique dans to_canonical(),
            # deux colonnes finissaient avec le meme nom ('Label' en double)
            # -> df_raw['Label'] retournait un DataFrame au lieu d'une Series
            # -> AttributeError sur .str, silencieusement avale en
            # format_non_reconnu par le except plus bas. Comparer sur les noms
            # nettoyes evite cette duplication.
            colonnes_nettoyees = {c.strip() for c in raw_df.columns}
            manquantes = [c for c in attendu if c not in colonnes_nettoyees]
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

        if source not in self.medianes_par_source:
            return out  # source inconnue/non validée -> tout en format_non_reconnu

        # Nettoyage des noms de colonnes une seule fois ici (CSV réels avec
        # espaces en tete, ex. export CICIDS2017) -- sinon la detection de
        # completude ci-dessous croit presque tout absent et rejette tout le
        # fichier avant meme d'appeler _adapt().
        raw_df = raw_df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

        # --- complétude des colonnes BRUTES (les adaptateurs remplissent déjà
        # le palier 1 canonique par 0 via finalize_dtypes -> la détection doit
        # se faire AVANT l'adaptation, sur les colonnes brutes attendues,
        # sinon aucune valeur manquante n'est plus jamais visible) ---
        colonnes_label = {'Label', 'Attack', 'detailed-label'}
        tier2_only = self._raw_cols_tier2_only.get(source, set())
        cols_features_attendues = [c for c in self._raw_cols.get(source, []) if c not in colonnes_label]
        # Le taux d'incomplétude qui décide si on rejette le scoring ne porte
        # que sur les colonnes palier 1 -- une source dont seul le palier 2
        # manque intégralement (colonnes dans tier2_only) reste scorable.
        cols_gate = [c for c in cols_features_attendues if c not in tier2_only]
        if cols_gate:
            presentes = [c for c in cols_gate if c in raw_df.columns]
            sous_df = raw_df[presentes] if presentes else pd.DataFrame(index=raw_df.index)
            manquantes_par_col = len(cols_gate) - len(presentes)
            taux_manquant = (sous_df.isna().sum(axis=1) + manquantes_par_col) / len(cols_gate)
        else:
            taux_manquant = pd.Series(0.0, index=raw_df.index)
        trop_incomplet = taux_manquant >= SEUIL_IMPUTATION
        a_impute = (taux_manquant > 0) & (~trop_incomplet)

        canon = self._adapt(source, raw_df)
        if canon is None:
            return out  # adaptation impossible -> tout en format_non_reconnu
        medianes = self.medianes_par_source[source]

        # Événements dégradés : le 0 déjà mis par l'adaptateur (finalize_dtypes)
        # est remplacé par la médiane de CETTE source (jamais une constante
        # fabriquée entre sources) -- UNIQUEMENT sur les cellules reellement
        # manquantes (canon[col].isna()), pas toute la ligne. Un ecrasement
        # de ligne entiere etait correct tant que le manque restait rare/
        # sporadique, mais devient catastrophique quand une colonne brute est
        # structurellement absente pour TOUT le fichier (ex. 'Protocol' et
        # 'Source Port' absents d'un export CICIDS2017 reel) : taux_manquant
        # > 0 sur 100% des lignes -> ecrasement de TOUT le palier 1 par la
        # mediane -> chaque ligne devient statistiquement identique -> le
        # modele ne voit plus aucun signal reel (score identique partout,
        # 0 attaque detectee malgre 2180 attaques reelles etiquetees dans le
        # fichier de test). Corrige : seules les cellules NaN sont comblees.
        if medianes and a_impute.any():
            for col, med in medianes.items():
                if col in canon.columns:
                    mask = a_impute & canon[col].isna()
                    canon.loc[mask, col] = med

        idx_scorable = ~trop_incomplet
        if idx_scorable.any():
            X = canon.loc[idx_scorable, cs.TIER1_FEATURES].values.astype(np.float32)
            # anomaly_score = probabilité d'attaque du classifieur supervisé
            # (0-1, PLUS HAUT = plus suspect) -- sens inversé par rapport à
            # l'ancien decision_function d'Isolation Forest (plus BAS = plus
            # suspect). Impact si des dashboards Kibana existants trient sur
            # ce champ : à adapter au moment de l'intégration ELK.
            proba = self.palier1['model'].predict_proba(X)[:, 1]
            # Seuil GLOBAL (pas par source) : un seuil calibré séparément par
            # source sur la validation interne (dérivée de la même
            # distribution que l'entraînement) a été testé et abandonné --
            # neutre à légèrement négatif sur données réellement nouvelles
            # (recall zeek 51%->50% au lieu de mieux) plutôt qu'une
            # amélioration.
            y_pred = (proba >= self.palier1['seuil']).astype(bool)

            out.loc[idx_scorable, 'ml_status'] = np.where(a_impute.loc[idx_scorable], 'scoring_degrade', 'ok')
            out.loc[idx_scorable, 'is_attack'] = y_pred
            out.loc[idx_scorable, 'anomaly_score'] = proba

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
        # NaN/pd.NA (attack_category et confidence pour un event non-attaque,
        # ou anomaly_score/is_attack si format_non_reconnu) -> None : le JSON
        # strict (RFC 8259, utilisé par FastAPI/Starlette) rejette NaN, donc
        # /predict lèverait une ValueError sur CHAQUE event normal sans ceci.
        for k, v in res.items():
            if v is pd.NA or (isinstance(v, float) and np.isnan(v)):
                res[k] = None
        return res
