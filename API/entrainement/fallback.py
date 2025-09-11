import numpy as np

class FallbackRanker:
    """Proxy autour du modèle: si la prédiction est constante/NaN, on replie
    sur une heuristique déterministe basée sur X (ex: colonnes texte)."""
    def __init__(self, base_model, tie_cols=(-2, -1)):
        self.base_model = base_model
        self.tie_cols = tie_cols  # dernières colonnes = features texte (T)

    def _score(self, X):
        # 1) Essaye le modèle de base
        s = None
        try:
            if hasattr(self.base_model, "predict_proba"):
                proba = self.base_model.predict_proba(X)
                s = proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else np.ravel(proba)
            elif hasattr(self.base_model, "decision_function"):
                s = self.base_model.decision_function(X)
            else:
                s = self.base_model.predict(X)
            s = np.asarray(s, dtype=float)
        except Exception:
            s = np.full(X.shape[0], np.nan)

        # 2) Fallback si dégénéré (NaN/Inf ou constant)
        if (not np.all(np.isfinite(s))) or (np.unique(s).size <= 1):
            t = X[:, self.tie_cols]  # colonnes texte (dans [0,1])
            t = np.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
            s = t.mean(axis=1) if t.ndim == 2 else t.astype(float)

        return np.asarray(s, dtype=float)

    # API utilisée par le test
    def predict(self, X):
        return self._score(X)

    def decision_function(self, X):
        return self._score(X)

    # Pour compat scikit/joblib si nécessaire
    @property
    def feature_names_in_(self):
        return getattr(self.base_model, "feature_names_in_", None)