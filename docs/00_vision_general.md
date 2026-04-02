# 00 - Vision General del Proyecto

## Que es este proyecto

Este proyecto es un **estudio comparativo de viabilidad tecnica** que evalua cuatro metodos de deteccion no supervisada de anomalias en senales electrocardiograficas (ECG). No es un producto clinico ni pretende reemplazar diagnosticos medicos. Es una investigacion que responde una pregunta concreta:

> **¿Cual es el rendimiento comparativo de K-Means, DBSCAN, HDBSCAN y un Autoencoder para detectar anomalias en ECG, y cual ofrece la mejor relacion rendimiento-costo computacional?**

## Por que importa

Las enfermedades cardiovasculares (ECV) son la **primera causa de muerte a nivel mundial** segun la OMS. El electrocardiograma (ECG) es la herramienta diagnostica mas comun para detectar arritmias cardiacas, pero:

- **Depende de especialistas:** Interpretar un ECG requiere un cardiologo entrenado. En regiones como Narino (Colombia), el acceso a cardiologos es limitado.
- **Arritmias intermitentes:** Muchas arritmias no ocurren durante el examen, requiriendo monitoreo continuo (Holter de 24-48 horas) que genera miles de latidos a revisar manualmente.
- **Modelos supervisados requieren datos etiquetados:** Entrenar un clasificador supervisado necesita grandes volumenes de datos anotados por expertos, lo cual es costoso y escaso fuera de centros de investigacion.

Los metodos **no supervisados** ofrecen una alternativa: detectar patrones anomalos sin necesidad de etiquetas de entrenamiento. Pero la pregunta clave que nadie ha respondido de forma sistematica es: **¿funcionan lo suficientemente bien como para ser viables? ¿Y cuanto se gana al usar metodos mas complejos?**

## Que NO es este proyecto

Es importante ser honestos sobre el alcance:

- **NO es una implementacion clinica.** No se lleva a un hospital ni se usa con pacientes.
- **NO promete mejorar la atencion en salud directamente.** Es un estudio de viabilidad tecnica.
- **NO usa datos de la region de Narino.** Usa MIT-BIH (datos de Boston, anos 80) que es el benchmark internacional. La referencia regional es una **motivacion**, no el **alcance**.

Esta honestidad intelectual es una fortaleza, no una debilidad. Un proyecto que sabe lo que es y lo que no es siempre sera mejor evaluado.

## El enfoque: comparacion progresiva de complejidad

La estrategia es comparar cuatro metodos de **complejidad creciente**, todos bajo las mismas condiciones de preprocesamiento. Esto es lo que hace valioso el proyecto: **la comparacion justa**.

```
Nivel 1: K-Means          (1957) → Baseline. Simple, rapido, asume clusters esfericos.
Nivel 2: DBSCAN            (1996) → Mejora: no asume forma, detecta ruido.
Nivel 3: HDBSCAN           (2013) → Evolucion: auto-configura parametros.
Nivel 4: Autoencoder (Deep Learning) → Moderno: aprende representacion comprimida.
```

**La pregunta que esto responde con valor genuino:**
Si HDBSCAN alcanza F1 de 0.82 en 5 segundos y el autoencoder 0.87 en 30 segundos, ¿cuando vale la pena el costo extra? Esa informacion es practicamente util para contextos de recursos limitados.

## Flujo general del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    MIT-BIH Arrhythmia Database                  │
│              44 registros, ~100,000 latidos, 360 Hz             │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│               PREPROCESAMIENTO (Paso 1-3)                       │
│  Filtrado Butterworth → Segmentacion 200 muestras → Z-score    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
┌──────────────────────┐  ┌──────────────────────────┐
│   Path A: Senal+PCA  │  │  Path B: 12 Features     │
│   200 dim → ~8 dim   │  │  Morfologicas+RR+Stats   │
└──────────┬───────────┘  └────────────┬─────────────┘
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    4 MODELOS (Paso 4-6)                          │
│  K-Means │ DBSCAN │ HDBSCAN │ Autoencoder                      │
│  Cada uno genera: etiquetas binarias (normal/anomalia)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUACION (Paso 7)                           │
│  Intrinsecas (sin etiquetas) + Extrinsecas (ground truth AAMI) │
│  + Eficiencia computacional (tiempo, memoria)                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              TABLA COMPARATIVA FINAL                             │
│  4 modelos × 2 representaciones = 8 configuraciones             │
│  × metricas intrinsecas + extrinsecas + eficiencia              │
└─────────────────────────────────────────────────────────────────┘
```

## Valor del proyecto en tres dimensiones

### 1. Comparacion progresiva de complejidad
Nadie en la literatura ha presentado una comparacion limpia de K-Means, DBSCAN, HDBSCAN y autoencoder bajo las mismas condiciones de preprocesamiento para ECG.

### 2. Analisis de costo-beneficio
¿Cuanto rendimiento extra se obtiene al usar metodos mas complejos? ¿Justifica el costo computacional adicional? Esta informacion es practicamente util.

### 3. Efecto de la representacion de datos
¿La ingenieria de caracteristicas manual sigue siendo relevante o es mejor la senal cruda con PCA? Otra pregunta con valor practico real.

---

**Siguiente:** [01 - Datos MIT-BIH y Agrupacion AAMI](01_datos_mitbih.md)
