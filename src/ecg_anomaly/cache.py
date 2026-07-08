"""Cacheo de artefactos preprocesados para reproducibilidad entre notebooks."""

from pathlib import Path

import joblib

from ecg_anomaly.config import SystemConfig
from ecg_anomaly.preprocessing.pipeline import PreprocessedData


def get_or_build_preprocessed(
    config: SystemConfig, cache_dir: str = "../cache", force: bool = False
) -> PreprocessedData:
    """Carga el preprocesamiento desde disco o lo construye una sola vez.

    El nombre del cache incluye parametros que afectan el resultado, asi
    un cambio de configuracion invalida el cache automaticamente.
    """
    from ecg_anomaly.data.loader import MITBIHLoader
    from ecg_anomaly.preprocessing.pipeline import PreprocessingPipeline

    cache_path_dir = Path(cache_dir)
    cache_path_dir.mkdir(parents=True, exist_ok=True)

    # Firma: si cambias filtros/ventana/umbral, se regenera solo
    signature = (
        f"pp_lc{config.filter_lowcut}_hc{config.filter_highcut}"
        f"_ord{config.filter_order}_b{config.before_r_samples}"
        f"_a{config.after_r_samples}.joblib"
    )
    cache_path = cache_path_dir / signature

    if cache_path.exists() and not force:
        print(f"[cache] Cargando preprocesamiento desde {cache_path.name}")
        return joblib.load(cache_path)

    print("[cache] Construyendo preprocesamiento (esto tarda)...")
    loader = MITBIHLoader(config)
    dataset = loader.load(config.dataset_path)
    pipeline = PreprocessingPipeline(config)
    preprocessed = pipeline.run(dataset)

    joblib.dump(preprocessed, cache_path)
    print(f"[cache] Guardado en {cache_path.name}")
    return preprocessed
