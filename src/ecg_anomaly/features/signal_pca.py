"""Path A: Representacion por senal directa + reduccion PCA."""

import logging

import numpy as np
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    PCA = None
    StandardScaler = None
    logging.warning("sklearn module not installed. PCA extraction will not be available.")

logger = logging.getLogger(__name__)


class SignalPCAExtractor:
    """Extractor Path A: senal directa escalada + PCA.

    Para los algoritmos de clustering, aplica StandardScaler + PCA
    reteniendo el porcentaje de varianza configurado.
    Para el autoencoder, retorna la senal escalada sin PCA
    (el autoencoder aprende su propia representacion comprimida).
    """

    def __init__(self, variance_threshold: float = 0.95, random_state: int = 42):
        self.variance_threshold = variance_threshold
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_threshold, random_state=random_state)
        self._is_fitted = False

    def fit(self, segments: np.ndarray) -> "SignalPCAExtractor":
        """Ajusta scaler y PCA sin transformar.

        Separado de fit_transform para poder ajustar solo con un subconjunto
        (p. ej. latidos normales) y evitar fuga de datos hacia la evaluacion.

        Args:
            segments: Array [N, beat_length] con latidos normalizados.
        """
        scaled = self.scaler.fit_transform(segments)
        self.pca.fit(scaled)
        self._is_fitted = True

        logger.info(
            "PCA: %d -> %d componentes (%.1f%% varianza retenida)",
            segments.shape[1],
            self.n_components,
            self.explained_variance_ratio * 100,
        )

        return self

    def fit_transform(self, segments: np.ndarray) -> np.ndarray:
        """Escala y reduce dimensionalidad de los segmentos.

        Args:
            segments: Array [N, beat_length] con latidos normalizados.

        Returns:
            Array [N, k] donde k componentes retienen el % de varianza.
        """
        self.fit(segments)
        return self.transform(segments)

    def transform(self, segments: np.ndarray) -> np.ndarray:
        """Transforma nuevos segmentos usando el PCA ya ajustado."""
        if not self._is_fitted:
            raise RuntimeError("Debe llamar fit_transform antes de transform")
        scaled = self.scaler.transform(segments)
        return self.pca.transform(scaled)

    def get_raw_for_autoencoder(self, segments: np.ndarray) -> np.ndarray:
        """Retorna datos escalados sin PCA para el autoencoder.

        El autoencoder trabaja con la senal completa (200 dimensiones)
        porque aprende su propia compresion en la capa de encoding.
        """
        if self._is_fitted:
            return self.scaler.transform(segments)
        return self.scaler.fit_transform(segments)

    @property
    def n_components(self) -> int:
        """Numero de componentes PCA seleccionados."""
        if not self._is_fitted:
            return 0
        return self.pca.n_components_

    @property
    def explained_variance_ratio(self) -> float:
        """Varianza total explicada por los componentes seleccionados."""
        if not self._is_fitted:
            return 0.0
        return float(np.sum(self.pca.explained_variance_ratio_))
