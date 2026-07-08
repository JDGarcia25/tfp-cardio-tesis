"""Configuracion centralizada del sistema de deteccion de anomalias ECG."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig:
    """Configuracion centralizada del sistema.

    Soporta carga desde YAML o JSON. Todos los parametros del pipeline
    de deteccion de anomalias se centralizan aqui.
    """

    # Reproducibilidad
    random_seed: int = 42

    # Dataset
    dataset_name: str = "mitbih"
    dataset_path: str = "./data/mitbih"

    # Modelos a ejecutar
    models: List[str] = field(
        default_factory=lambda: ["kmeans", "dbscan", "hdbscan", "autoencoder"]
    )

    # Representacion de datos: "signal_pca" (Path A) o "manual_features" (Path B)
    representation: str = "signal_pca"

    # Senal ECG
    sampling_rate: int = 360
    before_r_samples: int = 90
    after_r_samples: int = 110
    filter_lowcut: float = 0.5
    filter_highcut: float = 40.0
    filter_order: int = 4

    # PCA
    pca_variance_threshold: float = 0.95

    # Registros excluidos (ritmos de marcapasos)
    excluded_records: List[str] = field(
        default_factory=lambda: ["102", "104", "107", "217"]
    )

    # Hiperparametros por modelo
    kmeans_params: Dict = field(
        default_factory=lambda: {
            "n_clusters": 10,
            "random_state": 42,
            "n_init": 20,
            "distance_percentile": 89.5,
        }
    )
    dbscan_params: Dict = field(
        default_factory=lambda: {"eps": "auto", "min_samples": 5, "eps_percentile": 75}
    )
    hdbscan_params: Dict = field(
        default_factory=lambda: {
            "min_cluster_size": 30,
            "min_samples": 15,
            "cluster_selection_epsilon": 0.5,
        }
    )
    autoencoder_params: Dict = field(
        default_factory=lambda: {
            "encoding_dim": 6,
            "hidden_layers": [10, 8],
            "epochs": 30,
            "batch_size": 256,
            "anomaly_rate": 0.105,
            "learning_rate": 0.001,
        }
    )

    # Salida
    output_dir: str = "./results"
    generate_plots: bool = True
    logging_level: str = "INFO"

    @property
    def beat_length(self) -> int:
        """Longitud total del latido segmentado en muestras."""
        return self.before_r_samples + self.after_r_samples

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        """Carga configuracion desde archivo YAML.
        
        Resuelve dataset_path relativo a la raiz del proyecto
        (donde esta pyproject.toml) para evitar errores de CWD.
        """
        yaml_path = Path(path).resolve()
        # Buscar la raiz del proyecto (donde esta pyproject.toml)
        project_root = yaml_path.parent
        while not (project_root / "pyproject.toml").exists():
            if project_root.parent == project_root:
                break
            project_root = project_root.parent
        
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        # Resolver dataset_path relativo a la raiz del proyecto
        dp = config_dict.get("dataset_path", "")
        if dp and not Path(dp).is_absolute():
            config_dict["dataset_path"] = str((project_root / dp).resolve())
        
        return cls(**config_dict)

    @classmethod
    def from_json(cls, path: str) -> "SystemConfig":
        """Carga configuracion desde archivo JSON."""
        with open(path, "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        """Serializa la configuracion a diccionario."""
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "models": self.models,
            "representation": self.representation,
            "sampling_rate": self.sampling_rate,
            "beat_length": self.beat_length,
            "kmeans_params": self.kmeans_params,
            "dbscan_params": self.dbscan_params,
            "hdbscan_params": self.hdbscan_params,
            "autoencoder_params": self.autoencoder_params,
        }

    def save_yaml(self, path: str) -> None:
        """Guarda configuracion en archivo YAML."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def setup_logging(self) -> None:
        """Configura el nivel de logging global."""
        logging.basicConfig(
            level=getattr(logging, self.logging_level.upper(), logging.INFO),
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
