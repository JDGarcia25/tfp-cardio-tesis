"""Metricas de eficiencia computacional.

Responden a: ¿Cual metodo es mas practico en terminos de tiempo y recursos?
"""

import threading
import time
from typing import Dict, Optional

import psutil


class EfficiencyTracker:
    """Context manager para medir tiempo de ejecucion y uso de memoria.

    Mide el RSS (Resident Set Size) del proceso via psutil, muestreado en un
    hilo en segundo plano, en vez de tracemalloc. tracemalloc solo rastrea
    asignaciones en el heap de Python y no ve la memoria nativa reservada por
    extensiones en C/C++ (TensorFlow, BLAS de NumPy/SciPy), lo que subestimaba
    fuertemente el consumo del Autoencoder frente a los modelos de sklearn.

    Uso:
        with EfficiencyTracker() as tracker:
            model.fit(X)
        print(tracker.to_dict())
    """

    def __init__(self, sample_interval_seconds: float = 0.05):
        self.elapsed_seconds: float = 0.0
        self.peak_memory_mb: float = 0.0
        self._sample_interval = sample_interval_seconds
        self._process = psutil.Process()
        self._stop_event = threading.Event()
        self._sampler_thread: Optional[threading.Thread] = None
        self._peak_rss_bytes: int = 0

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            rss = self._process.memory_info().rss
            if rss > self._peak_rss_bytes:
                self._peak_rss_bytes = rss
            self._stop_event.wait(self._sample_interval)

    def __enter__(self) -> "EfficiencyTracker":
        self._peak_rss_bytes = self._process.memory_info().rss
        self._stop_event.clear()
        self._sampler_thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._sampler_thread.start()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed_seconds = time.perf_counter() - self._start_time
        self._stop_event.set()
        if self._sampler_thread is not None:
            self._sampler_thread.join(timeout=1.0)
        final_rss = self._process.memory_info().rss
        peak = max(self._peak_rss_bytes, final_rss)
        self.peak_memory_mb = peak / (1024 * 1024)

    def to_dict(self) -> Dict[str, float]:
        """Retorna metricas como diccionario."""
        return {
            "time_seconds": round(self.elapsed_seconds, 4),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
        }
