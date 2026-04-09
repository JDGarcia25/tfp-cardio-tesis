# 10 - Fundamentos de Python aplicados al proyecto

Mini curso de Python usando ejemplos reales del codigo del proyecto. Cada concepto se explica primero de forma general y luego se muestra donde aparece en `src/ecg_anomaly/`.

---

## 1. Variables y tipos de datos

Una variable es un nombre que apunta a un valor almacenado en memoria. En Python no se declara el tipo — se infiere automaticamente.

```python
# Tipos basicos
nombre = "MIT-BIH"          # str  (texto)
frecuencia = 360             # int  (entero)
umbral = 0.95                # float (decimal)
generar_graficos = True      # bool (verdadero/falso)
modelos = ["kmeans", "dbscan"]  # list (lista ordenada)
params = {"n_clusters": 2}   # dict (diccionario clave:valor)
```

### En el proyecto

En `src/ecg_anomaly/config.py`:

```python
class SystemConfig:
    dataset_name: str = "mitbih"           # str
    sampling_rate: int = 360               # int
    filter_lowcut: float = 0.5             # float
    generate_plots: bool = True            # bool
    models: List[str] = [...]              # lista de strings
    kmeans_params: Dict = {"n_clusters": 2}  # diccionario
```

Cada parametro del sistema es una variable con un tipo y un valor por defecto. Esto evita numeros magicos dispersos por el codigo.

---

## 2. Imports: `import` y `from ... import`

Los imports permiten usar codigo que vive en otros archivos (modulos) o librerias externas.

### `import` — Importa un modulo completo

```python
import numpy                    # Importa todo el modulo
datos = numpy.array([1, 2, 3])  # Hay que escribir el nombre completo

import numpy as np              # Alias corto (convencion estandar)
datos = np.array([1, 2, 3])     # Mas comodo
```

### `from ... import` — Importa algo especifico de un modulo

```python
from numpy import array         # Solo trae 'array'
datos = array([1, 2, 3])        # Se usa directamente, sin prefijo

from sklearn.cluster import KMeans  # Trae una clase de un subpaquete
modelo = KMeans(n_clusters=2)
```

### Diferencia clave

| Forma | Que hace | Cuando usar |
|---|---|---|
| `import numpy as np` | Trae todo, accedes con `np.algo` | Librerias grandes (numpy, pandas) — queda claro de donde viene cada funcion |
| `from sklearn.cluster import KMeans` | Trae solo `KMeans` | Cuando necesitas pocas cosas especificas y el nombre ya es descriptivo |

### En el proyecto

En `src/ecg_anomaly/models/kmeans.py`:

```python
from typing import Dict                         # 1. Libreria estandar de Python
import numpy as np                               # 2. Libreria externa (pip)
from sklearn.cluster import KMeans               # 3. Libreria externa (subpaquete)
from ecg_anomaly.models.base import BaseAnomalyDetector  # 4. Modulo del proyecto
```

Estas cuatro lineas muestran los 4 origenes tipicos de un import:

1. **`typing`** — Viene con Python, no hay que instalar nada
2. **`numpy`** — Libreria externa instalada con `pip` o `poetry`
3. **`sklearn.cluster`** — Subpaquete de scikit-learn; solo traemos la clase `KMeans`
4. **`ecg_anomaly.models.base`** — Otro archivo de nuestro propio proyecto

### Como funciona la ruta de un import del proyecto

```python
from ecg_anomaly.models.base import BaseAnomalyDetector
#    ───────────── ────── ────         ───────────────────
#    paquete raiz  carpeta archivo     clase que queremos
#    src/ecg_anomaly/models/base.py    class BaseAnomalyDetector
```

Python busca el archivo siguiendo la ruta de puntos:
- `ecg_anomaly` → carpeta `src/ecg_anomaly/`
- `.models` → subcarpeta `models/`
- `.base` → archivo `base.py`
- `import BaseAnomalyDetector` → la clase definida dentro de ese archivo

### El archivo `__init__.py`

Cada carpeta que contiene un `__init__.py` se convierte en un **paquete** importable:

```
src/ecg_anomaly/
├── __init__.py          ← Hace que 'ecg_anomaly' sea importable
├── models/
│   ├── __init__.py      ← Hace que 'ecg_anomaly.models' sea importable
│   └── kmeans.py
```

Sin `__init__.py`, Python no reconoce la carpeta como paquete y el `from ecg_anomaly.models import ...` falla.

---

## 3. Funciones

Una funcion es un bloque de codigo reutilizable que recibe datos (parametros) y devuelve un resultado.

```python
def sumar(a, b):
    """Suma dos numeros."""   # Docstring: documentacion de la funcion
    resultado = a + b
    return resultado

total = sumar(3, 5)  # total = 8
```

### Parametros con valores por defecto

```python
def saludar(nombre, idioma="es"):
    if idioma == "es":
        return f"Hola, {nombre}"
    return f"Hello, {nombre}"

saludar("Ana")             # "Hola, Ana"  (usa el default "es")
saludar("Ana", idioma="en")  # "Hello, Ana"
```

### En el proyecto

En `src/ecg_anomaly/preprocessing/filters.py`:

```python
def butterworth_bandpass(signal, lowcut, highcut, fs, order=4):
    """Aplica filtro Butterworth pasa-banda."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal, axis=0)
```

- `signal`, `lowcut`, `highcut`, `fs` → parametros obligatorios
- `order=4` → parametro con valor por defecto (si no se pasa, usa 4)
- `return filtfilt(...)` → devuelve la senal filtrada

Se llama asi:

```python
# Usa order=4 por defecto (el de config)
senal_filtrada = butterworth_bandpass(senal, 0.5, 40.0, 360)

# Se puede cambiar el orden explicitamente
senal_filtrada = butterworth_bandpass(senal, 0.5, 40.0, 360, order=6)
```

---

## 4. Clases y objetos

Una **clase** es un plano (plantilla) que define que datos tiene algo y que puede hacer. Un **objeto** es una instancia concreta creada a partir de esa clase.

### Analogia

```
Clase = Plano de una casa     →  define: cuartos, puertas, ventanas
Objeto = Casa construida      →  una casa real con sus propios muebles
```

### Sintaxis basica

```python
class Perro:
    def __init__(self, nombre, raza):    # Se ejecuta al crear el objeto
        self.nombre = nombre             # Atributo: dato del objeto
        self.raza = raza

    def ladrar(self):                    # Metodo: accion del objeto
        return f"{self.nombre} dice: Guau!"

# Crear objetos (instancias)
mi_perro = Perro("Max", "Labrador")
mi_perro.nombre    # "Max"
mi_perro.ladrar()  # "Max dice: Guau!"
```

### `self` — La referencia al propio objeto

`self` es el objeto sobre el que se esta trabajando. Python lo pasa automaticamente.

```python
class Contador:
    def __init__(self):
        self.valor = 0         # self.valor pertenece a ESTE contador

    def incrementar(self):
        self.valor += 1        # Modifica el valor de ESTE contador

c1 = Contador()
c2 = Contador()
c1.incrementar()
c1.incrementar()
# c1.valor = 2, c2.valor = 0  → cada objeto tiene su propio estado
```

### En el proyecto

En `src/ecg_anomaly/data/loader.py`:

```python
class MITBIHLoader:
    def __init__(self, config: SystemConfig):   # Recibe la configuracion
        self.config = config                    # La guarda como atributo
        self.registry = RecordRegistry()        # Crea otro objeto interno

    def load(self, path, records=None, channel=0):
        # Usa self.config y self.registry que se guardaron en __init__
        if records is None:
            records = self.registry.get_valid_records(
                set(self.config.excluded_records)
            )
        ...
```

1. `__init__` guarda la configuracion y crea el registro
2. `load()` usa ambos (`self.config`, `self.registry`) para cargar datos
3. Cada `MITBIHLoader` tiene su propia configuracion independiente

---

## 5. Herencia

Una clase puede **heredar** de otra, recibiendo todos sus atributos y metodos. Esto evita repetir codigo.

```python
class Animal:
    def __init__(self, nombre):
        self.nombre = nombre

    def respirar(self):
        return f"{self.nombre} esta respirando"

class Gato(Animal):            # Gato HEREDA de Animal
    def maullar(self):         # Agrega su propio metodo
        return f"{self.nombre} dice: Miau!"

gato = Gato("Luna")
gato.respirar()   # "Luna esta respirando"  ← heredado de Animal
gato.maullar()    # "Luna dice: Miau!"      ← propio de Gato
```

### En el proyecto

En `src/ecg_anomaly/models/kmeans.py`:

```python
class KMeansDetector(BaseAnomalyDetector):  # Hereda de BaseAnomalyDetector
    """Detector K-Means (Nivel 1 - Baseline)."""

    def fit(self, X):                        # Implementa el metodo abstracto
        self.model = KMeans(**self.params)    # self.params viene de la clase padre
        self.labels_ = self.model.fit_predict(X)
        self._assign_anomalies()
        return self
```

- `KMeansDetector` hereda `__init__`, `self.name`, `self.params`, `self.model` de `BaseAnomalyDetector`
- Solo necesita implementar `fit()`, `predict_anomalies()` y `get_params()`
- Los otros 3 detectores (DBSCAN, HDBSCAN, Autoencoder) hacen lo mismo: heredan la misma base y solo cambian **como** detectan anomalias

---

## 6. Clases abstractas (ABC)

Una **clase abstracta** es una clase que **no se puede usar directamente** — solo sirve como plantilla que obliga a las clases hijas a implementar ciertos metodos.

```python
from abc import ABC, abstractmethod

class Figura(ABC):                # ABC = Abstract Base Class
    @abstractmethod               # Este decorador marca el metodo como obligatorio
    def area(self):
        """Las clases hijas DEBEN implementar este metodo."""

class Circulo(Figura):
    def __init__(self, radio):
        self.radio = radio

    def area(self):               # Obligatorio — si falta, Python lanza error
        return 3.14159 * self.radio ** 2

# figura = Figura()   ← ERROR: no se puede instanciar una clase abstracta
circulo = Circulo(5)
circulo.area()        # 78.54
```

### En el proyecto

En `src/ecg_anomaly/models/base.py`:

```python
from abc import ABC, abstractmethod

class BaseAnomalyDetector(ABC):
    """Interfaz base para todos los detectores."""

    def __init__(self, name, params):
        self.name = name
        self.params = params
        self.model = None
        self.anomaly_labels_ = None

    @abstractmethod
    def fit(self, X):
        """Entrenar el modelo — CADA detector lo implementa diferente."""

    @abstractmethod
    def predict_anomalies(self, X):
        """Predecir anomalias — CADA detector lo implementa diferente."""

    @abstractmethod
    def get_params(self):
        """Retornar parametros."""
```

**Por que es util:** Garantiza que cualquier detector nuevo tenga los 3 metodos. Si alguien crea `NuevoDetector(BaseAnomalyDetector)` y olvida implementar `fit()`, Python lanza un error inmediato en vez de fallar misteriosamente despues.

---

## 7. Decoradores

Un **decorador** es una funcion que modifica el comportamiento de otra funcion o clase. Se escribe con `@` encima de lo que se quiere decorar.

### `@dataclass` — Genera codigo automaticamente

Sin dataclass hay que escribir `__init__` manualmente:

```python
# Sin @dataclass (tedioso)
class Punto:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# Con @dataclass (automatico)
from dataclasses import dataclass

@dataclass
class Punto:
    x: float
    y: float
    z: float

# Python genera __init__, __repr__, __eq__ automaticamente
p = Punto(1.0, 2.0, 3.0)
print(p)  # Punto(x=1.0, y=2.0, z=3.0)
```

**En el proyecto** — `src/ecg_anomaly/data/loader.py`:

```python
@dataclass
class ECGRecord:
    record_id: str
    signal: np.ndarray
    r_peak_positions: np.ndarray
    symbols: np.ndarray
    binary_labels: np.ndarray
    sampling_rate: int = 360

# Se crea asi, sin escribir __init__:
registro = ECGRecord(
    record_id="100",
    signal=senal,
    r_peak_positions=picos,
    symbols=simbolos,
    binary_labels=etiquetas,
)
registro.record_id  # "100"
```

### `@property` — Convierte un metodo en atributo

Permite acceder a un calculo como si fuera un dato, sin parentesis.

```python
class Rectangulo:
    def __init__(self, ancho, alto):
        self.ancho = ancho
        self.alto = alto

    @property
    def area(self):
        return self.ancho * self.alto

r = Rectangulo(3, 4)
r.area    # 12  ← sin parentesis, parece un atributo
# r.area()  ← esto daria error
```

**En el proyecto** — `src/ecg_anomaly/config.py`:

```python
@property
def beat_length(self) -> int:
    """Longitud total del latido segmentado en muestras."""
    return self.before_r_samples + self.after_r_samples

# Se usa como atributo:
config.beat_length  # 200 (= 90 + 110)
```

### `@classmethod` — Metodo de la clase, no del objeto

Un metodo normal recibe `self` (el objeto). Un classmethod recibe `cls` (la clase) y puede crear instancias de formas alternativas.

```python
class Fecha:
    def __init__(self, dia, mes, anio):
        self.dia = dia
        self.mes = mes
        self.anio = anio

    @classmethod
    def desde_string(cls, texto):       # cls = la clase Fecha
        dia, mes, anio = texto.split("-")
        return cls(int(dia), int(mes), int(anio))  # Crea una Fecha

# Dos formas de crear el mismo objeto:
f1 = Fecha(9, 4, 2026)
f2 = Fecha.desde_string("09-04-2026")   # Alternativa mas legible
```

**En el proyecto** — `src/ecg_anomaly/config.py`:

```python
@classmethod
def from_yaml(cls, path: str) -> "SystemConfig":
    """Carga configuracion desde archivo YAML."""
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)
    return cls(**config_dict)      # cls = SystemConfig

# Se usa asi:
config = SystemConfig.from_yaml("config/default.yaml")
```

`cls(**config_dict)` equivale a `SystemConfig(dataset_name="mitbih", sampling_rate=360, ...)` — crea un `SystemConfig` con los valores del archivo YAML.

### `@abstractmethod` — Obliga a implementar en las clases hijas

Ya visto en la seccion 6. Marca metodos que las clases hijas **deben** definir.

---

## 8. Type hints (anotaciones de tipo)

Python no obliga a declarar tipos, pero las **anotaciones** documentan que tipo de dato espera cada variable. No cambian el comportamiento — son para legibilidad y herramientas del IDE.

```python
# Sin type hints
def filtrar(senal, frecuencia):
    ...

# Con type hints → queda claro que espera y que devuelve
def filtrar(senal: np.ndarray, frecuencia: float) -> np.ndarray:
    ...
```

### Tipos comunes del proyecto

```python
from typing import Dict, List, Optional

nombre: str = "kmeans"                  # Texto
tasa: int = 360                         # Entero
umbral: float = 0.95                    # Decimal
modelos: List[str] = ["kmeans"]         # Lista de strings
params: Dict[str, int] = {"k": 2}      # Diccionario string→entero
resultado: Optional[str] = None         # Puede ser str o None
```

**En el proyecto** — `src/ecg_anomaly/models/base.py`:

```python
def fit(self, X: np.ndarray) -> "BaseAnomalyDetector":
#               ───────────    ────────────────────────
#               X debe ser     devuelve el propio objeto
#               un array       (para encadenar llamadas)
```

---

## 9. El operador `**` (desempaquetado de diccionarios)

El doble asterisco `**` "abre" un diccionario y lo convierte en argumentos con nombre.

```python
params = {"n_clusters": 2, "random_state": 42}

# Estas dos lineas son equivalentes:
modelo = KMeans(**params)
modelo = KMeans(n_clusters=2, random_state=42)
```

### En el proyecto

En `src/ecg_anomaly/models/kmeans.py`:

```python
def fit(self, X):
    self.model = KMeans(**self.params)  # Desempaqueta el diccionario de config
```

Si `self.params = {"n_clusters": 2, "random_state": 42, "n_init": 10}`, entonces `KMeans(**self.params)` equivale a `KMeans(n_clusters=2, random_state=42, n_init=10)`.

Esto permite que los hiperparametros vengan del YAML sin que el codigo los conozca de antemano.

---

## 10. Context managers (`with`)

Un **context manager** ejecuta codigo de preparacion al entrar y codigo de limpieza al salir, garantizando que la limpieza siempre ocurra (incluso si hay errores).

### El caso mas comun: archivos

```python
# Sin with → hay que cerrar manualmente (y se olvida)
f = open("datos.txt")
contenido = f.read()
f.close()

# Con with → se cierra automaticamente al salir del bloque
with open("datos.txt") as f:
    contenido = f.read()
# Aqui f ya esta cerrado, pase lo que pase
```

### Crear un context manager propio

Se definen los metodos `__enter__` (al entrar) y `__exit__` (al salir):

```python
class Cronometro:
    def __enter__(self):
        self.inicio = time.time()
        return self

    def __exit__(self, *args):
        self.duracion = time.time() - self.inicio

with Cronometro() as c:
    hacer_algo_lento()
print(f"Tardo {c.duracion} segundos")
```

### En el proyecto

En `src/ecg_anomaly/evaluation/efficiency.py`:

```python
class EfficiencyTracker:
    def __enter__(self):
        tracemalloc.start()               # Empieza a medir memoria
        self._start_time = time.perf_counter()  # Marca el tiempo inicial
        return self

    def __exit__(self, *args):
        self.elapsed_seconds = time.perf_counter() - self._start_time  # Calcula tiempo
        _, peak = tracemalloc.get_traced_memory()  # Lee pico de memoria
        tracemalloc.stop()                 # Para de medir (limpieza)
        self.peak_memory_mb = peak / (1024 * 1024)
```

Se usa en `src/ecg_anomaly/evaluation/comparator.py`:

```python
with EfficiencyTracker() as tracker:
    detector.fit(X)                     # Entrena el modelo

# Al salir del with, tracker ya tiene el tiempo y la memoria
detector.fit_time_seconds = tracker.elapsed_seconds
detector.peak_memory_mb = tracker.peak_memory_mb
```

---

## 11. List comprehensions

Una forma compacta de crear listas aplicando una operacion a cada elemento.

```python
# Forma larga (for tradicional)
cuadrados = []
for n in [1, 2, 3, 4]:
    cuadrados.append(n ** 2)
# cuadrados = [1, 4, 9, 16]

# Forma corta (list comprehension)
cuadrados = [n ** 2 for n in [1, 2, 3, 4]]

# Con filtro
pares = [n for n in range(10) if n % 2 == 0]  # [0, 2, 4, 6, 8]
```

### En el proyecto

En `src/ecg_anomaly/data/loader.py`:

```python
# Clasificar cada simbolo como 0 (normal) o 1 (anomalo)
binary_labels = np.array(
    [RecordRegistry.classify_symbol(s) for s in beat_symbols]
)
```

Recorre cada simbolo en `beat_symbols`, lo clasifica, y crea un array con los resultados.

---

## 12. `field()` en dataclasses

Cuando un atributo de un dataclass tiene un valor mutable por defecto (lista, diccionario), hay que usar `field(default_factory=...)` para que cada instancia tenga **su propia copia**.

```python
from dataclasses import dataclass, field

# MAL — todas las instancias compartirian la MISMA lista
@dataclass
class Config:
    modelos: list = ["kmeans"]   # ← Python lanza error

# BIEN — cada instancia crea su propia lista
@dataclass
class Config:
    modelos: list = field(default_factory=lambda: ["kmeans"])
```

### En el proyecto

En `src/ecg_anomaly/config.py`:

```python
models: List[str] = field(
    default_factory=lambda: ["kmeans", "dbscan", "hdbscan", "autoencoder"]
)

kmeans_params: Dict = field(
    default_factory=lambda: {"n_clusters": 2, "random_state": 42, "n_init": 10}
)
```

`lambda:` crea una funcion anonima que se ejecuta cada vez que se crea un `SystemConfig` nuevo, asegurando que cada configuracion tenga su propia copia de la lista y del diccionario.

---

## 13. Patron Factory resumido

El Factory combina varios conceptos de Python en un patron util:

```python
class DetectorFactory:
    # Diccionario de clase: nombre → clase del detector
    _detectors = {
        "kmeans": KMeansDetector,       # Guarda la CLASE, no un objeto
        "dbscan": DBSCANDetector,
    }

    @classmethod
    def create(cls, name, params):
        detector_class = cls._detectors[name]  # Busca la clase por nombre
        return detector_class(name, params)     # La instancia (crea objeto)
```

Paso a paso:

```python
# 1. El diccionario mapea strings a clases
_detectors["kmeans"]  # → KMeansDetector (la clase, no un objeto)

# 2. create() busca la clase
detector_class = _detectors["kmeans"]  # detector_class = KMeansDetector

# 3. La llama como constructor (crea el objeto)
detector_class("kmeans", {"n_clusters": 2})
# Equivale a:
KMeansDetector("kmeans", {"n_clusters": 2})
```

Esto permite crear objetos por nombre (string del YAML) sin usar cadenas de `if/elif`.

---

## 14. Metodos especiales (`__dunder__`)

Los metodos que empiezan y terminan con `__` (llamados "dunder" — double underscore) son metodos especiales que Python llama automaticamente en ciertas situaciones.

| Metodo | Cuando se ejecuta | Ejemplo |
|---|---|---|
| `__init__(self, ...)` | Al crear un objeto: `obj = Clase()` | Inicializar atributos |
| `__repr__(self)` | Al imprimir: `print(obj)` o en el REPL | Representacion legible |
| `__enter__(self)` | Al entrar en un `with` | Preparar recursos |
| `__exit__(self, ...)` | Al salir de un `with` | Liberar recursos |
| `__len__(self)` | Al llamar `len(obj)` | Retornar tamanio |

### En el proyecto

En `src/ecg_anomaly/models/base.py`:

```python
def __repr__(self) -> str:
    return f"{self.__class__.__name__}(name={self.name!r})"

# Cuando imprimes un detector:
detector = KMeansDetector("kmeans", {"n_clusters": 2})
print(detector)  # KMeansDetector(name='kmeans')
```

---

## 15. Resumen visual: como se conecta todo

```
┌─────────────────────────────────────────────────────────────┐
│  config/default.yaml                                        │
│  → Se carga con SystemConfig.from_yaml() [@classmethod]     │
│  → Produce un objeto SystemConfig [@dataclass]              │
└───────────────────────────┬─────────────────────────────────┘
                            │
                  from ecg_anomaly.config import SystemConfig
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  pipeline.py                                                │
│  class ECGAnomalyPipeline:     [Facade]                     │
│      def __init__(self, config):                            │
│          self.config = config  [guarda como atributo]       │
│      def run(self):            [orquesta todo el flujo]     │
│          loader = MITBIHLoader(self.config)  [crea objeto]  │
│          dataset = loader.load(...)          [llama metodo] │
│          ...                                                │
└───────────────────────────┬─────────────────────────────────┘
                            │
                  from ecg_anomaly.models.factory
                       import DetectorFactory
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  factory.py                                                 │
│  DetectorFactory.create("kmeans", params)                   │
│      → busca en _detectors["kmeans"] = KMeansDetector       │
│      → retorna KMeansDetector("kmeans", params)             │
│                                                             │
│  KMeansDetector(BaseAnomalyDetector)  [herencia]            │
│      def fit(self, X):          [implementa @abstractmethod]│
│          self.model = KMeans(**self.params)  [** desempaq.] │
└───────────────────────────┬─────────────────────────────────┘
                            │
                  with EfficiencyTracker() as tracker:
                      detector.fit(X)
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  comparator.py                                              │
│  → Recolecta metricas de cada modelo                        │
│  → Genera DataFrame comparativo (pandas)                    │
│  → Identifica el mejor modelo por F1                        │
└─────────────────────────────────────────────────────────────┘
```

Cada flecha es un `from ... import` — el mecanismo que conecta los modulos entre si.

---

**Anterior:** [09 - Manual de Ejecucion](09_manual_ejecucion.md) | **Indice:** [Documentacion](README.md)
