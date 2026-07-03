"""Metricas de eficiencia computacional.

Responden a: ¿Cual metodo es mas practico en terminos de tiempo y recursos?
"""

import time
import tracemalloc
from typing import Dict


class EfficiencyTracker:
    """Context manager para medir tiempo de ejecucion y uso de memoria.

    Uso:
        with EfficiencyTracker() as tracker:
            model.fit(X)
        print(tracker.to_dict())
    """

    def __init__(self):
        self.elapsed_seconds: float = 0.0
        self.peak_memory_mb: float = 0.0

    def __enter__(self) -> "EfficiencyTracker":
        tracemalloc.start()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start_time
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.peak_memory_mb = peak / (1024 * 1024)

    def to_dict(self) -> Dict[str, float]:
        """Retorna metricas como diccionario."""
        return {
            "time_seconds": round(self.elapsed_seconds, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }
