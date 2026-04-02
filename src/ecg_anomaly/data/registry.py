"""Registro de registros MIT-BIH y constantes AAMI para agrupacion de anotaciones."""

from typing import Dict, FrozenSet, List, Set


# Agrupacion AAMI (de Chazal, O'Dwyer & Reilly, 2004)
# Normal: latidos dentro de la variabilidad normal
AAMI_NORMAL: FrozenSet[str] = frozenset({"N", "L", "R", "e", "j"})

# Anomalo: arritmias de distintos tipos
AAMI_ANOMALOUS: FrozenSet[str] = frozenset({"A", "a", "J", "S", "V", "E", "F", "/", "f", "Q"})

# Simbolos que no representan latidos (se ignoran)
NON_BEAT_SYMBOLS: FrozenSet[str] = frozenset(
    {"+", "~", "!", "[", "]", "x", "(", ")", "|", '"', "p", "t", "u"}
)

# Registros con ritmos de marcapasos (excluir segun recomendacion AAMI)
PACEMAKER_RECORDS: FrozenSet[str] = frozenset({"102", "104", "107", "217"})


class RecordRegistry:
    """Catalogo de registros MIT-BIH Arrhythmia Database.

    Mantiene la lista completa de 48 registros, sus categorias por
    patologia, y los registros excluidos por marcapasos.
    """

    # 48 registros estandar de MIT-BIH
    ALL_RECORDS: List[str] = [
        "100", "101", "102", "103", "104", "105", "106", "107",
        "108", "109", "111", "112", "113", "114", "115", "116",
        "117", "118", "119", "121", "122", "123", "124",
        "200", "201", "202", "203", "205", "207", "208",
        "209", "210", "212", "213", "214", "215", "217",
        "219", "220", "221", "222", "223", "228", "230",
        "231", "232", "233", "234",
    ]

    # Clasificacion por tipo de patologia predominante
    RECORD_CATEGORIES: Dict[str, List[str]] = {
        "Normal": [
            "100", "101", "103", "112", "113", "115", "116",
            "117", "121", "122", "123", "220", "230", "231", "232",
        ],
        "PVC_Ventricular": [
            "105", "106", "108", "109", "119", "200", "201",
            "203", "205", "208", "210", "213", "214", "215",
            "219", "221", "228", "233", "234",
        ],
        "APC_Atrial": [
            "209", "222", "223",
        ],
        "Bloqueos": [
            "111", "118", "124", "207", "212",
        ],
        "Mixto_Complejo": [
            "114", "202",
        ],
    }

    def get_valid_records(self, excluded: Set[str] | None = None) -> List[str]:
        """Retorna registros validos (excluyendo marcapasos y otros).

        Args:
            excluded: Conjunto adicional de registros a excluir.
                      Por defecto excluye PACEMAKER_RECORDS.
        """
        exclude = PACEMAKER_RECORDS | (excluded or set())
        return [r for r in self.ALL_RECORDS if r not in exclude]

    def get_records_by_category(self, category: str) -> List[str]:
        """Retorna registros de una categoria especifica, sin marcapasos."""
        if category not in self.RECORD_CATEGORIES:
            raise ValueError(
                f"Categoria '{category}' no encontrada. "
                f"Disponibles: {list(self.RECORD_CATEGORIES.keys())}"
            )
        return [r for r in self.RECORD_CATEGORIES[category] if r not in PACEMAKER_RECORDS]

    def get_categories(self) -> List[str]:
        """Retorna lista de categorias disponibles."""
        return list(self.RECORD_CATEGORIES.keys())

    @staticmethod
    def classify_symbol(symbol: str) -> int:
        """Clasifica un simbolo de anotacion MIT-BIH en binario AAMI.

        Returns:
            0 para normal, 1 para anomalo, -1 para no-latido (ignorar).
        """
        if symbol in AAMI_NORMAL:
            return 0
        if symbol in AAMI_ANOMALOUS:
            return 1
        return -1

    @staticmethod
    def is_beat_symbol(symbol: str) -> bool:
        """Verifica si un simbolo representa un latido cardiaco."""
        return symbol not in NON_BEAT_SYMBOLS
