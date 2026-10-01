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
            # orden determinista, para que el mismo calculo con los mismos
            # pesos iniciales dé siempre el mismo resultado.
            #
            # No alcanza para que dos autoencoders entrenados en el mismo
            # proceso sean identicos: en Keras 3 los pesos iniciales toman su
            # semilla del `random` de Python, que esta funcion fija una sola
            # vez y cada modelo construido avanza. Por eso
            # AutoencoderDetector.fit() reinicia el estado aleatorio completo
            # (keras.utils.set_random_seed) al empezar cada entrenamiento.
            # Verificado: tres fit() consecutivos en el mismo proceso, sobre
            # latidos reales de MIT-BIH, dan umbral y pesos identicos.
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
