# ============================================
# SISTEMA MODULAR DE DETECCIÓN DE ARRITMIAS
# Arquitectura Profesional para Google Colab
# Universidad CESMAG - 2026
# ============================================

import os
import json
import yaml
import numpy as np
import pandas as pd
import wfdb
import pywt
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, calinski_harabasz_score, 
                            davies_bouldin_score, adjusted_rand_score)
from sklearn.neighbors import NearestNeighbors

# Visualización
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 1. CONFIGURACIÓN CENTRALIZADA (YAML/JSON)
# =============================================================================

@dataclass
class SystemConfig:
    """Configuración centralizada del sistema"""
    # Dataset seleccionado
    dataset_name: str = "mitbih"
    dataset_path: str = "./data"
    
    # Modelos a ejecutar
    models: List[str] = field(default_factory=lambda: ["kmeans", "dbscan", "optics"])
    
    # Preprocesamiento
    sampling_rate: int = 360
    window_size: int = 180  # Muestras por latido
    filter_lowcut: float = 0.5
    filter_highcut: float = 40.0
    
    # Hiperparámetros por modelo
    kmeans_params: Dict = field(default_factory=lambda: {"n_clusters": 5, "random_state": 42})
    dbscan_params: Dict = field(default_factory=lambda: {"eps": 0.5, "min_samples": 10})
    optics_params: Dict = field(default_factory=lambda: {"min_samples": 10, "xi": 0.05})
    
    # Evaluación
    metrics: List[str] = field(default_factory=lambda: [
        "silhouette", "calinski_harabasz", "davies_bouldin", "ari"
    ])
    
    # Reportes
    output_dir: str = "./results"
    save_models: bool = True
    generate_plots: bool = True
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self):
        return {
            'dataset_name': self.dataset_name,
            'models': self.models,
            'sampling_rate': self.sampling_rate,
            'kmeans_params': self.kmeans_params,
            'dbscan_params': self.dbscan_params,
            'optics_params': self.optics_params
        }

# =============================================================================
# 2. MÓDULO DE CARGA DE DATOS (PATRÓN ADAPTADOR)
# =============================================================================

class BaseDataLoader(ABC):
    """Clase base abstracta para cargadores de datos"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data = None
        self.labels = None
        self.metadata = {}
    
    @abstractmethod
    def load(self, path: str) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
        """
        Carga datos y retorna: (señales, etiquetas, metadatos)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict:
        """Retorna información del dataset cargado"""
        pass

class MITBIHLoader(BaseDataLoader):
    """Adaptador para MIT-BIH Arrhythmia Database"""
    
    def load(self, path: str, records: List[str] = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Carga registros de MIT-BIH
        
        Args:
            path: Directorio con archivos .dat, .hea, .atr
            records: Lista de nombres de registros (ej: ['100', '101'])
        """
        if records is None:
            # Registros estándar recomendados
            records = ['100', '101', '102', '103', '104', '105', '106', '107', 
                      '108', '109', '111', '112', '113', '114', '115']
        
        all_signals = []
        all_labels = []
        total_beats = 0
        
        for record_name in records:
            try:
                # Cargar señal y anotaciones
                record = wfdb.rdrecord(os.path.join(path, record_name))
                annotation = wfdb.rdann(os.path.join(path, record_name), 'atr')
                
                # Canal II (generalmente más limpio)
                signal = record.p_signal[:, 0]
                
                # Extraer etiquetas en cada posición de anotación
                for i, sample in enumerate(annotation.sample):
                    if sample < len(signal):
                        all_signals.append(signal[max(0, sample-90):min(len(signal), sample+90)])
                        all_labels.append(annotation.symbol[i])
                        total_beats += 1
                
            except Exception as e:
                print(f"⚠️ Error cargando {record_name}: {e}")
                continue
        
        # Convertir a arrays numpy consistentes
        min_len = min(len(s) for s in all_signals) if all_signals else 0
        all_signals = np.array([s[:min_len] for s in all_signals])
        
        self.data = all_signals
        self.labels = np.array(all_labels)
        self.metadata = {
            'dataset': 'MIT-BIH',
            'records': records,
            'total_beats': total_beats,
            'sampling_rate': 360,
            'classes': list(set(all_labels))
        }
        
        return self.data, self.labels, self.metadata
    
    def get_info(self) -> Dict:
        return self.metadata

class CSVLoader(BaseDataLoader):
    """Adaptador para archivos CSV con señales ECG"""
    
    def load(self, path: str, signal_column: str = 'signal', 
             label_column: str = 'label', **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
        
        df = pd.read_csv(path)
        
        # Asumir que cada fila es un latido o ventana temporal
        if signal_column in df.columns:
            # Si hay una columna con listas de valores
            signals = df[signal_column].apply(lambda x: np.array(eval(x)) if isinstance(x, str) else x)
            self.data = np.vstack(signals.values)
        else:
            # Asumir que todas las columnas numéricas son la señal
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            self.data = df[numeric_cols].values
        
        if label_column in df.columns:
            self.labels = df[label_column].values
        else:
            self.labels = None
            
        self.metadata = {
            'dataset': 'CSV_Custom',
            'path': path,
            'shape': self.data.shape,
            'sampling_rate': kwargs.get('sampling_rate', 360)
        }
        
        return self.data, self.labels, self.metadata
    
    def get_info(self) -> Dict:
        return self.metadata

class DataLoaderFactory:
    """Fábrica de cargadores de datos"""
    
    _loaders = {
        'mitbih': MITBIHLoader,
        'csv': CSVLoader,
        # Fácilmente extensible para nuevos formatos
        # 'physionet': PhysioNetLoader,
        # 'h5': H5Loader,
    }
    
    @classmethod
    def get_loader(cls, dataset_type: str, config: SystemConfig) -> BaseDataLoader:
        if dataset_type not in cls._loaders:
            raise ValueError(f"Dataset '{dataset_type}' no soportado. "
                           f"Disponibles: {list(cls._loaders.keys())}")
        return cls._loaders[dataset_type](config)
    
    @classmethod
    def register_loader(cls, name: str, loader_class: type):
        """Permite registrar nuevos cargadores en tiempo de ejecución"""
        cls._loaders[name] = loader_class

# =============================================================================
# 3. MÓDULO DE PREPROCESAMIENTO (PIPELINE REUTILIZABLE)
# =============================================================================

class PreprocessingPipeline:
    """Pipeline modular de preprocesamiento"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.steps = []
        
    def add_step(self, name: str, func, **params):
        """Agrega un paso al pipeline"""
        self.steps.append({'name': name, 'func': func, 'params': params})
        return self
    
    def apply(self, signals: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Aplica pipeline completo y retorna resultados intermedios
        
        Returns:
            Dict con señales en cada etapa: raw, filtered, normalized, features
        """
        results = {'raw': signals.copy()}
        current = signals.copy()
        
        # 1. Filtrado
        current = self._bandpass_filter(current)
        results['filtered'] = current.copy()
        
        # 2. Detección de picos y segmentación
        segments, r_peaks = self._segment_beats(current)
        results['segments'] = segments
        results['r_peaks'] = r_peaks
        
        # 3. Normalización
        normalized = self._normalize(segments)
        results['normalized'] = normalized
        
        # 4. Extracción de características
        features = self._extract_features(normalized)
        results['features'] = features
        
        # 5. Reducción de dimensionalidad
        pca = PCA(n_components=0.95)
        features_reduced = pca.fit_transform(features)
        results['features_pca'] = features_reduced
        results['pca_explained_var'] = sum(pca.explained_variance_ratio_)
        
        return results
    
    def _bandpass_filter(self, signals: np.ndarray) -> np.ndarray:
        """Filtro pasa banda Butterworth"""
        from scipy.signal import butter, filtfilt
        
        nyquist = 0.5 * self.config.sampling_rate
        low = self.config.filter_lowcut / nyquist
        high = self.config.filter_highcut / nyquist
        
        b, a = butter(5, [low, high], btype='band')
        
        # Aplicar a cada señal si es 2D
        if signals.ndim == 1:
            return filtfilt(b, a, signals)
        else:
            return np.array([filtfilt(b, a, sig) for sig in signals])
    
    def _segment_beats(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detecta picos R y segmenta latidos
        Simplificado para señales ya segmentadas o continuas
        """
        if signal.ndim == 2:
            # Ya está segmentado
            return signal, np.arange(len(signal))
        
        # Si es señal continua, detectar picos
        from scipy.signal import find_peaks
        
        distance = int(0.4 * self.config.sampling_rate)
        peaks, _ = find_peaks(signal, distance=distance, prominence=0.5)
        
        # Extraer ventanas alrededor de picos
        window = self.config.window_size
        segments = []
        valid_peaks = []
        
        for peak in peaks:
            start = max(0, peak - window//2)
            end = min(len(signal), peak + window//2)
            if end - start == window:
                segments.append(signal[start:end])
                valid_peaks.append(peak)
        
        return np.array(segments), np.array(valid_peaks)
    
    def _normalize(self, segments: np.ndarray) -> np.ndarray:
        """Normalización Z-score por latido"""
        normalized = []
        for seg in segments:
            norm = (seg - np.mean(seg)) / (np.std(seg) + 1e-10)
            normalized.append(norm)
        return np.array(normalized)
    
    def _extract_features(self, segments: np.ndarray) -> np.ndarray:
        """
        Extracción de características morfológicas y estadísticas
        """
        features = []
        
        for seg in segments:
            feat = {}
            
            # Estadísticas básicas
            feat['mean'] = np.mean(seg)
            feat['std'] = np.std(seg)
            feat['max'] = np.max(seg)
            feat['min'] = np.min(seg)
            feat['range'] = feat['max'] - feat['min']
            
            # Momentos de orden superior
            feat['skewness'] = pd.Series(seg).skew()
            feat['kurtosis'] = pd.Series(seg).kurtosis()
            
            # Características de frecuencia (FFT)
            fft_vals = np.abs(np.fft.fft(seg))
            feat['dominant_freq'] = np.argmax(fft_vals[:len(fft_vals)//2])
            feat['spectral_energy'] = np.sum(fft_vals**2)
            
            # Wavelet
            coeffs = pywt.wavedec(seg, 'db4', level=3)
            for i, coef in enumerate(coeffs):
                feat[f'wavelet_energy_{i}'] = np.sum(coef**2)
            
            features.append(list(feat.values()))
        
        return np.array(features)

# =============================================================================
# 4. MÓDULO DE MODELOS (ESTRATEGIA)
# =============================================================================

class BaseClusteringModel(ABC):
    """Interfaz base para modelos de clustering"""
    
    def __init__(self, name: str, config: Dict):
        self.name = name
        self.config = config
        self.model = None
        self.labels = None
        self.metrics = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Entrena modelo y retorna etiquetas"""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """Retorna parámetros del modelo"""
        pass
    
    def evaluate(self, X: np.ndarray, true_labels: Optional[np.ndarray] = None) -> Dict:
        """Evalúa el modelo con métricas de clustering"""
        if self.labels is None:
            self.fit(X)
        
        unique_labels = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        
        if unique_labels > 1:
            self.metrics['silhouette'] = silhouette_score(X, self.labels)
            self.metrics['calinski_harabasz'] = calinski_harabasz_score(X, self.labels)
            self.metrics['davies_bouldin'] = davies_bouldin_score(X, self.labels)
        else:
            self.metrics['silhouette'] = -1
            self.metrics['calinski_harabasz'] = 0
            self.metrics['davies_bouldin'] = float('inf')
        
        # Métricas específicas de anomalías
        n_noise = list(self.labels).count(-1)
        self.metrics['n_clusters'] = unique_labels
        self.metrics['n_noise'] = n_noise
        self.metrics['noise_ratio'] = n_noise / len(self.labels)
        
        # Si hay etiquetas reales, calcular ARI
        if true_labels is not None:
            # Mapear etiquetas de clustering a reales (simplificado)
            from scipy.stats import mode
            mapped = self._map_clusters_to_labels(true_labels)
            self.metrics['ari'] = adjusted_rand_score(true_labels, mapped)
        
        return self.metrics
    
    def _map_clusters_to_labels(self, true_labels: np.ndarray) -> np.ndarray:
        """Mapea clusters a etiquetas reales usando moda"""
        mapped = np.full_like(self.labels, -1)
        for cluster in np.unique(self.labels):
            if cluster == -1:
                continue
            mask = self.labels == cluster
            mode_label = mode(true_labels[mask], keepdims=True).mode[0]
            mapped[mask] = mode_label
        return mapped

class KMeansModel(BaseClusteringModel):
    def fit(self, X: np.ndarray) -> np.ndarray:
        self.model = KMeans(**self.config)
        self.labels = self.model.fit_predict(X)
        return self.labels
    
    def get_params(self):
        return self.config

class DBSCANModel(BaseClusteringModel):
    def fit(self, X: np.ndarray) -> np.ndarray:
        # Auto-optimizar eps si es 'auto'
        if self.config.get('eps') == 'auto':
            self.config['eps'] = self._optimize_eps(X)
        
        self.model = DBSCAN(**self.config)
        self.labels = self.model.fit_predict(X)
        return self.labels
    
    def _optimize_eps(self, X: np.ndarray) -> float:
        from sklearn.neighbors import NearestNeighbors
        neigh = NearestNeighbors(n_neighbors=self.config.get('min_samples', 5))
        neigh.fit(X)
        distances, _ = neigh.kneighbors(X)
        distances = np.sort(distances[:, -1])
        return np.percentile(distances, 90)
    
    def get_params(self):
        return self.config

class OPTICSModel(BaseClusteringModel):
    def fit(self, X: np.ndarray) -> np.ndarray:
        self.model = OPTICS(**self.config)
        self.labels = self.model.fit_predict(X)
        return self.labels
    
    def get_params(self):
        return self.config

class ModelFactory:
    """Fábrica de modelos de clustering"""
    
    _models = {
        'kmeans': KMeansModel,
        'dbscan': DBSCANModel,
        'optics': OPTICSModel,
    }
    
    @classmethod
    def get_model(cls, name: str, config: Dict) -> BaseClusteringModel:
        if name not in cls._models:
            raise ValueError(f"Modelo '{name}' no disponible")
        return cls._models[name](name, config)
    
    @classmethod
    def list_models(cls):
        return list(cls._models.keys())

# =============================================================================
# 5. MÓDULO DE EVALUACIÓN Y COMPARACIÓN
# =============================================================================

class ModelEvaluator:
    """Evaluación comparativa de múltiples modelos"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.results = []
    
    def evaluate_all(self, X: np.ndarray, true_labels: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Ejecuta y evalúa todos los modelos configurados"""
        
        for model_name in self.config.models:
            print(f"🔧 Entrenando {model_name}...")
            
            # Obtener configuración específica
            model_config = getattr(self.config, f"{model_name}_params", {})
            
            # Crear y entrenar modelo
            model = ModelFactory.get_model(model_name, model_config)
            labels = model.fit(X)
            
            # Evaluar
            metrics = model.evaluate(X, true_labels)
            
            # Guardar resultados
            result = {
                'model': model_name,
                'labels': labels,
                'n_clusters': metrics['n_clusters'],
                'n_noise': metrics['n_noise'],
                'noise_ratio': metrics['noise_ratio'],
                'silhouette': metrics.get('silhouette', -1),
                'calinski_harabasz': metrics.get('calinski_harabasz', 0),
                'davies_bouldin': metrics.get('davies_bouldin', float('inf')),
                'ari': metrics.get('ari', None)
            }
            self.results.append(result)
            
            print(f"   ✅ Clusters: {result['n_clusters']}, "
                  f"Anomalías: {result['n_noise']} ({result['noise_ratio']:.1%})")
        
        return pd.DataFrame(self.results)
    
    def get_best_model(self, metric: str = 'silhouette') -> Dict:
        """Retorna el mejor modelo según una métrica"""
        df = pd.DataFrame(self.results)
        if metric in ['davies_bouldin']:
            return df.loc[df[metric].idxmin()].to_dict()  # Menor es mejor
        return df.loc[df[metric].idxmax()].to_dict()  # Mayor es mejor

# =============================================================================
# 6. MÓDULO DE REPORTES Y VISUALIZACIÓN
# =============================================================================

class ReportGenerator:
    """Generación de reportes y visualizaciones"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
    
    def generate_comparison_report(self, results_df: pd.DataFrame, 
                                  preprocessing_results: Dict,
                                  save_path: Optional[str] = None):
        """Genera reporte comparativo completo"""
        
        # 1. Tabla de métricas
        print("\n" + "="*60)
        print("REPORTE COMPARATIVO DE MODELOS")
        print("="*60)
        print(results_df[['model', 'n_clusters', 'n_noise', 'silhouette', 
                         'davies_bouldin', 'ari']].to_string(index=False))
        
        # 2. Gráfico de barras comparativo
        self._plot_metrics_comparison(results_df)
        
        # 3. Visualización de clusters del mejor modelo
        best_model = results_df.loc[results_df['silhouette'].idxmax(), 'model']
        best_labels = results_df[results_df['model'] == best_model]['labels'].iloc[0]
        self._plot_clusters(preprocessing_results['features_pca'], best_labels, best_model)
        
        # 4. Guardar resultados
        if save_path:
            results_df.to_csv(os.path.join(self.config.output_dir, 
                                          f"results_{datetime.now():%Y%m%d_%H%M%S}.csv"), 
                            index=False)
            print(f"\n💾 Resultados guardados en: {self.config.output_dir}")
    
    def _plot_metrics_comparison(self, results_df: pd.DataFrame):
        """Gráfico comparativo de métricas"""
        fig = go.Figure()
        
        metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        
        for metric, color in zip(metrics, colors):
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=results_df['model'],
                y=results_df[metric],
                marker_color=color
            ))
        
        fig.update_layout(
            title='Comparación de Métricas por Modelo',
            barmode='group',
            template='plotly_white',
            height=500
        )
        fig.show()
    
    def _plot_clusters(self, X: np.ndarray, labels: np.ndarray, model_name: str):
        """Visualización 2D de clusters"""
        fig = px.scatter(
            x=X[:, 0], y=X[:, 1],
            color=labels.astype(str),
            title=f'Clusters detectados por {model_name.upper()}',
            labels={'x': 'Componente 1', 'y': 'Componente 2', 'color': 'Cluster'},
            opacity=0.7
        )
        fig.update_layout(template='plotly_white', height=600)
        fig.show()

# =============================================================================
# 7. ORQUESTADOR PRINCIPAL (FACADE)
# =============================================================================

class ArrhythmiaDetectionSystem:
    """
    Sistema principal que orquesta todos los módulos
    Patrón Facade para interfaz simplificada
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.data_loader = None
        self.preprocessor = PreprocessingPipeline(config)
        self.evaluator = ModelEvaluator(config)
        self.reporter = ReportGenerator(config)
        
        self.raw_data = None
        self.labels = None
        self.preprocessed = None
        self.results = None
    
    def load_data(self, dataset_type: str = None, path: str = None, **kwargs):
        """Carga datos usando el adaptador apropiado"""
        dataset_type = dataset_type or self.config.dataset_name
        path = path or self.config.dataset_path
        
        print(f"📥 Cargando dataset: {dataset_type}")
        self.data_loader = DataLoaderFactory.get_loader(dataset_type, self.config)
        self.raw_data, self.labels, metadata = self.data_loader.load(path, **kwargs)
        
        print(f"   ✅ {metadata.get('total_beats', len(self.raw_data))} latidos cargados")
        return self
    
    def preprocess(self):
        """Ejecuta pipeline de preprocesamiento"""
        print("🔧 Preprocesando señales...")
        self.preprocessed = self.preprocessor.apply(self.raw_data)
        
        print(f"   ✅ Características extraídas: {self.preprocessed['features'].shape}")
        print(f"   ✅ Reducción PCA: {self.preprocessed['features_pca'].shape} "
              f"({self.preprocessed['pca_explained_var']:.1%} varianza)")
        return self
    
    def run_models(self):
        """Ejecuta todos los modelos configurados"""
        print("🚀 Ejecutando modelos de clustering...")
        X = self.preprocessed['features_pca']
        self.results = self.evaluator.evaluate_all(X, self.labels)
        return self
    
    def generate_report(self):
        """Genera reporte final"""
        self.reporter.generate_comparison_report(
            self.results, 
            self.preprocessed,
            save_path=True
        )
        return self
    
    def get_best_model(self):
        """Retorna el mejor modelo"""
        return self.evaluator.get_best_model()
    
    def full_pipeline(self, dataset_type: str = None, path: str = None, **kwargs):
        """Ejecuta pipeline completo"""
        return (self
                .load_data(dataset_type, path, **kwargs)
                .preprocess()
                .run_models()
                .generate_report())

# =============================================================================
# EJEMPLO DE USO EN GOOGLE COLAB
# =============================================================================

def demo_mitbih():
    """Demostración completa con MIT-BIH"""
    
    # 1. Crear configuración
    config = SystemConfig(
        dataset_name="mitbih",
        dataset_path="./mitbih_data",
        models=["kmeans", "dbscan", "optics"],
        sampling_rate=360,
        kmeans_params={"n_clusters": 5, "random_state": 42, "n_init": 10},
        dbscan_params={"eps": "auto", "min_samples": 10},
        optics_params={"min_samples": 10, "xi": 0.05, "min_cluster_size": 0.05},
        output_dir="./results"
    )
    
    # 2. Inicializar sistema
    system = ArrhythmiaDetectionSystem(config)
    
    # 3. Ejecutar pipeline completo
    try:
        system.full_pipeline(
            dataset_type="mitbih",
            path="./mitbih_data",
            records=['100', '101', '102', '103', '104']  # Subset para demo
        )
        
        # 4. Ver resultados
        best = system.get_best_model()
        print(f"\n🏆 Mejor modelo: {best['model']}")
        print(f"   Silhouette: {best['silhouette']:.3f}")
        print(f"   Anomalías detectadas: {best['n_noise']} ({best['noise_ratio']:.1%})")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Asegúrate de tener los datos MIT-BIH descargados")

# Ejecutar demostración
if __name__ == "__main__":
    print("🎓 Sistema de Detección de Arritmias - Universidad CESMAG")
    print("="*60)
    demo_mitbih()