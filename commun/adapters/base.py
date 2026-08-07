"""
Contrat commun à tout adaptateur de format de flux. Un adaptateur ne fait
QUE de la conversion de colonnes (aucun échantillonnage, aucun entraînement) :
il prend un DataFrame dans le format brut d'un outil et retourne un
DataFrame au schéma canonique (commun/canonical_schema.py).

Ajouter un nouveau format = sous-classer BaseFlowAdapter, implémenter
to_canonical(). Ni le schéma canonique ni les modèles n'ont besoin de changer.
Voir README.md pour un exemple minimal.
"""

import pandas as pd
from .. import canonical_schema as cs


class BaseFlowAdapter:
    SOURCE_NAME = None  # ex. 'netflow', 'cicflowmeter', 'zeek', 'suricata'

    def to_canonical(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """À implémenter par chaque adaptateur. Doit retourner un DataFrame
        avec toutes les colonnes de canonical_schema.ALL_OUTPUT_COLS."""
        raise NotImplementedError

    def validate(self, df: pd.DataFrame) -> None:
        """Vérifie que la sortie respecte le contrat : palier 1 complet et
        sans NaN, palier 2 présent (NaN toléré), colonnes méta présentes."""
        missing_t1 = [c for c in cs.TIER1_FEATURES if c not in df.columns]
        if missing_t1:
            raise ValueError(f"[{self.SOURCE_NAME}] colonnes palier 1 manquantes : {missing_t1}")
        if df[cs.TIER1_FEATURES].isna().any().any():
            bad = df[cs.TIER1_FEATURES].columns[df[cs.TIER1_FEATURES].isna().any()].tolist()
            raise ValueError(
                f"[{self.SOURCE_NAME}] le palier 1 doit toujours être rempli "
                f"(jamais de NaN) — colonnes en défaut : {bad}"
            )
        missing_t2 = [c for c in cs.TIER2_FEATURES if c not in df.columns]
        if missing_t2:
            raise ValueError(
                f"[{self.SOURCE_NAME}] colonnes palier 2 manquantes (doivent "
                f"exister même si non remplies, cf. NaN) : {missing_t2}"
            )
        missing_meta = [c for c in cs.META_COLS if c not in df.columns]
        if missing_meta:
            raise ValueError(f"[{self.SOURCE_NAME}] colonnes méta manquantes : {missing_meta}")

    def extract(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Point d'entrée : convertit + valide."""
        df = self.to_canonical(df_raw)
        self.validate(df)
        return df
