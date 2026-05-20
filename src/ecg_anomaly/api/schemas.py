"""Esquemas Pydantic para el API de prediccion de anomalias ECG."""

from typing import List, Optional

from pydantic import BaseModel, Field


class BeatInput(BaseModel):
    """Latido ECG de entrada para clasificacion.

    El latido debe ser un segmento de 200 muestras centrado en el pico R
    (90 muestras antes + 110 despues, frecuencia de muestreo 360 Hz).

    El campo ``preprocessed`` controla si el API aplica normalizacion Z-score:

    - ``None`` (default): deteccion automatica por heuristica estadistica.
      Si ``|media| < 0.5`` y ``0.5 < std < 1.5``, se asume que ya esta
      normalizado; en caso contrario, el API lo normaliza.
    - ``True``: el beat ya tiene Z-score aplicado; el API no modifica nada.
    - ``False``: el beat NO esta normalizado; el API aplica Z-score antes
      de pasarlo al modelo.
    """

    beat: List[float] = Field(
        ...,
        min_length=200,
        max_length=200,
        description=(
            "200 muestras del latido ECG segmentado. "
            "Frecuencia de muestreo esperada: 360 Hz."
        ),
        examples=[[0.0] * 200],
    )
    preprocessed: Optional[bool] = Field(
        None,
        description=(
            "Estado de preprocesamiento del beat. "
            "True → ya normalizado por Z-score, el API no modifica nada. "
            "False → no normalizado, el API aplica Z-score automaticamente. "
            "None (default) → deteccion automatica por heuristica estadistica."
        ),
    )


class PredictionOutput(BaseModel):
    """Resultado de la prediccion de anomalia ECG."""

    prediction: int = Field(
        ...,
        description="Clasificacion binaria: 0 = normal, 1 = anomalia",
        examples=[0],
    )
    label: str = Field(
        ...,
        description="Etiqueta legible: 'normal' o 'anomalia'",
        examples=["normal"],
    )
    reconstruction_error: Optional[float] = Field(
        None,
        description="Error de reconstruccion MSE (solo disponible para autoencoder)",
    )
    threshold: Optional[float] = Field(
        None,
        description="Umbral de decision del modelo",
    )
    model_name: str = Field(
        ...,
        description="Nombre del modelo utilizado para la prediccion",
        examples=["autoencoder"],
    )
    normalization_applied: bool = Field(
        ...,
        description=(
            "True si el API aplico normalizacion Z-score al beat antes de inferencia. "
            "False si el beat ya estaba normalizado."
        ),
    )


class HealthOutput(BaseModel):
    """Estado del servicio."""

    status: str = Field(..., description="Estado del servicio: 'ok'")
    model_name: str = Field(..., description="Nombre del mejor modelo cargado")
    model_type: str = Field(..., description="Tipo: 'autoencoder' o 'clustering'")
