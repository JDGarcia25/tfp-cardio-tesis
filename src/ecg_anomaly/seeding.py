"""Fijacion de semillas para reproducibilidad total."""

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        try:
            # Fuerza que las operaciones de TF (reducciones, etc.) usen un
            # orden determinista. Sin esto, tf.random.set_seed no basta:
            # el mismo experimento (misma seed, mismos datos) puede dar
            # resultados distintos entre corridas por no-determinismo a
            # nivel de operacion (ver notebook 05, comparacion Seccion 2
            # vs Seccion 10 con el autoencoder).
            tf.config.experimental.enable_op_determinism()
        except Exception:
            pass
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass
