#!/usr/bin/env python3
"""
=============================================================================
SISTEMA DINÁMICO DE DETECCIÓN DE ARRITMIAS - EVALUADOR INTERACTIVO
=============================================================================

Características:
- Menú interactivo para seleccionar registros
- Base de datos de registros procesados
- Evaluación en tiempo real de nuevos registros
- Dashboard de visualización
- Exportación de resultados comparativos

Autor: Sistema ECG Avanzado
Versión: 2.0 - Interactivo
=============================================================================
"""

import subprocess
import sys
import os
import json
import pickle
from datetime import datetime
from pathlib import Path

# =============================================================================
# INSTALADOR AUTOMÁTICO
# =============================================================================

def instalar_paquetes():
    paquetes_requeridos = {
        'numpy': 'numpy', 'scipy': 'scipy', 'matplotlib': 'matplotlib',
        'seaborn': 'seaborn', 'pandas': 'pandas', 'sklearn': 'scikit-learn',
        'joblib': 'joblib', 'wfdb': 'wfdb', 'pywt': 'PyWavelets'
    }
    
    print("=" * 70)
    print("VERIFICANDO DEPENDENCIAS...")
    print("=" * 70)
    
    faltantes = []
    for modulo, paquete in paquetes_requeridos.items():
        try:
            __import__(modulo)
            print(f"✓ {paquete}")
        except ImportError:
            print(f"✗ {paquete} - INSTALANDO...")
            faltantes.append(paquete)
    
    if faltantes:
        print(f"\nInstalando {len(faltantes)} paquetes...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + faltantes)
        print("✓ Instalación completada\n")
    
    # Opcionales
    opcionales = {'hdbscan': 'hdbscan', 'umap': 'umap-learn', 'nk': 'neurokit2'}
    for modulo, paquete in opcionales.items():
        try:
            __import__(modulo)
            print(f"✓ {paquete} (opcional)")
        except ImportError:
            print(f"○ {paquete} (no instalado)")
    
    return True

instalar_paquetes()

# =============================================================================
# IMPORTACIONES
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, TextBox
import warnings
warnings.filterwarnings('ignore')

from scipy import signal
from scipy.stats import skew, kurtosis, entropy

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

import wfdb
import pywt
from joblib import dump, load

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    from sklearn.decomposition import PCA as UMAP_Fallback

try:
    import neurokit2 as nk
    HAS_NEUROKIT = True
except ImportError:
    HAS_NEUROKIT = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Compatibilidad NumPy 2.0+
TRAPZ = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz

# =============================================================================
# CATÁLOGO DE REGISTROS MIT-BIH
# =============================================================================

class MITBIHCatalog:
    """
    Catálogo completo de registros MIT-BIH con metadatos.
    """
    
    # Todos los registros disponibles en MIT-BIH Arrhythmia Database
    ALL_RECORDS = [
        '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
        '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
        '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
        '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
        '222', '223', '228', '230', '231', '232', '233', '234'
    ]
    
    # Clasificación por tipo de arritmia dominante (según anotaciones MIT-BIH)
    RECORD_CATEGORIES = {
        'Normal': ['100', '101', '103', '117', '122', '212'],
        'PVC/Ventricular': ['105', '106', '116', '119', '201', '203', '208', '210', '213', '215', '221', '228'],
        'APC/Atrial': ['200', '202', '219', '220', '234'],
        'Bloqueos': ['104', '108', '111', '112', '123', '217'],
        'Mixto/Complejo': ['102', '107', '109', '113', '114', '115', '118', '121', '124', 
                          '205', '207', '209', '214', '222', '223', '230', '231', '232', '233']
    }
    
    def __init__(self):
        self.processed_records = set()
        self.current_session_records = set()
        self.load_history()
    
    def load_history(self):
        """Carga historial de registros procesados."""
        history_file = Path('ecg_processing_history.json')
        if history_file.exists():
            with open(history_file, 'r') as f:
                data = json.load(f)
                self.processed_records = set(data.get('processed', []))
                print(f"📚 Historial cargado: {len(self.processed_records)} registros previos")
    
    def save_history(self):
        """Guarda historial actualizado."""
        with open('ecg_processing_history.json', 'w') as f:
            json.dump({
                'processed': list(self.processed_records),
                'last_update': datetime.now().isoformat()
            }, f)
    
    def get_available_records(self, category=None, exclude_processed=True):
        """
        Obtiene registros disponibles con filtros opcionales.
        
        Args:
            category: 'Normal', 'PVC/Ventricular', 'APC/Atrial', 'Bloqueos', 'Mixto/Complejo'
            exclude_processed: Si True, excluye registros ya procesados
        """
        if category:
            base_records = self.RECORD_CATEGORIES.get(category, [])
        else:
            base_records = self.ALL_RECORDS
        
        if exclude_processed:
            available = [r for r in base_records if r not in self.processed_records]
        else:
            available = base_records
        
        return available
    
    def get_record_info(self, record_id):
        """Obtiene información sobre un registro específico."""
        info = {
            'id': record_id,
            'category': None,
            'processed': record_id in self.processed_records,
            'in_session': record_id in self.current_session_records
        }
        
        for cat, records in self.RECORD_CATEGORIES.items():
            if record_id in records:
                info['category'] = cat
                break
        
        return info
    
    def mark_as_processed(self, record_id):
        """Marca un registro como procesado."""
        self.processed_records.add(record_id)
        self.current_session_records.add(record_id)
        self.save_history()
    
    def get_statistics(self):
        """Estadísticas de uso."""
        total = len(self.ALL_RECORDS)
        processed = len(self.processed_records)
        session = len(self.current_session_records)
        
        return {
            'total_available': total,
            'total_processed': processed,
            'current_session': session,
            'remaining': total - processed,
            'percentage_used': (processed / total) * 100
        }


# =============================================================================
# SISTEMA DE PROCESAMIENTO ECG (CLASES ANTERIORES OPTIMIZADAS)
# =============================================================================

class ECGDataLoader:
    def __init__(self):
        self.fs = 360
        self.loaded_data = {}
    
    def load_record(self, record_id, cache=True):
        """Carga un registro específico."""
        if cache and record_id in self.loaded_data:
            return self.loaded_data[record_id]
        
        try:
            record = wfdb.rdrecord(record_id, pn_dir='mitdb')
            
            if record.n_sig > 1:
                sig_names = [s.lower() for s in record.sig_name]
                channel = sig_names.index('mlii') if 'mlii' in sig_names else 0
            else:
                channel = 0
            
            ecg_signal = record.p_signal[:, channel]
            annotations = wfdb.rdann(record_id, 'atr', pn_dir='mitdb')
            
            data = {
                'signal': ecg_signal,
                'annotations': annotations,
                'fs': record.fs,
                'duration': len(ecg_signal) / record.fs,
                'n_beats': len(annotations.sample)
            }
            
            if cache:
                self.loaded_data[record_id] = data
            
            return data
            
        except Exception as e:
            print(f"❌ Error cargando {record_id}: {e}")
            return None
    
    def clear_cache(self):
        """Limpia caché de memoria."""
        self.loaded_data.clear()


class ECGPreprocessor:
    def __init__(self, fs=360):
        self.fs = fs
    
    def bandpass_filter(self, sig, low=0.5, high=40, order=4):
        nyq = 0.5 * self.fs
        b, a = signal.butter(order, [low/nyq, high/nyq], btype='band')
        return signal.filtfilt(b, a, sig)
    
    def pan_tompkins(self, sig):
        """Detección QRS con Pan-Tompkins."""
        nyq = 0.5 * self.fs
        b, a = signal.butter(2, [5/nyq, 15/nyq], btype='band')
        filtered = signal.filtfilt(b, a, sig)
        
        derivative = np.diff(filtered)
        derivative = np.append(derivative, 0)
        squared = derivative ** 2
        
        window = int(0.150 * self.fs)
        integrated = np.convolve(squared, np.ones(window)/window, mode='same')
        
        peaks, _ = signal.find_peaks(integrated, distance=int(0.2*self.fs), 
                                     height=np.mean(integrated))
        
        # Refinar picos
        qrs_peaks = []
        for p in peaks:
            start = max(0, p - int(0.05*self.fs))
            end = min(len(sig), p + int(0.05*self.fs))
            true_peak = start + np.argmax(sig[start:end])
            qrs_peaks.append(true_peak)
        
        return np.array(qrs_peaks)
    
    def segment_beats(self, sig, peaks, before=0.2, after=0.4):
        """Segmenta latidos."""
        beats, valid = [], []
        b_samples = int(before * self.fs)
        a_samples = int(after * self.fs)
        
        for p in peaks:
            start, end = p - b_samples, p + a_samples
            if start >= 0 and end < len(sig):
                beat = sig[start:end]
                beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-10)
                beats.append(beat)
                valid.append(p)
        
        return beats, np.array(valid)


class ECGFeatureExtractor:
    def __init__(self, fs=360):
        self.fs = fs
    
    def extract_features(self, beats, peaks):
        """Extrae features de un conjunto de latidos."""
        if len(beats) < 2:
            return None
        
        # Features RR
        rr_intervals = np.diff(peaks) / self.fs * 1000  # ms
        
        features = {
            'rr_mean': np.mean(rr_intervals),
            'rr_std': np.std(rr_intervals),
            'rr_min': np.min(rr_intervals),
            'rr_max': np.max(rr_intervals),
            'rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)) if len(rr_intervals) > 1 else 0,
            'n_beats': len(beats)
        }
        
        # Features morfológicos agregados
        all_beats = np.array(beats)
        features['beat_mean_amp'] = np.mean([np.max(b) - np.min(b) for b in beats])
        features['beat_std_amp'] = np.std([np.max(b) for b in beats])
        
        # Entropía de la variabilidad
        if len(rr_intervals) > 10:
            hist, _ = np.histogram(rr_intervals, bins=10, density=True)
            features['rr_entropy'] = entropy(hist + 1e-10)
        else:
            features['rr_entropy'] = 0
        
        return features


# =============================================================================
# SISTEMA DE CLUSTERING DINÁMICO
# =============================================================================

class DynamicClusteringSystem:
    """
    Sistema de clustering que se actualiza incrementalmente.
    """
    
    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.scaler = RobustScaler()
        self.model = None
        self.is_fitted = False
        self.feature_history = []
        self.record_history = []
        
    def partial_fit(self, features, record_id):
        """Añade datos y reentrena si es necesario."""
        self.feature_history.append(features)
        self.record_history.append(record_id)
        
        # Reentrenar cada 5 registros nuevos o si es el primero
        if len(self.feature_history) % 5 == 1 or not self.is_fitted:
            self._retrain()
    
    def _retrain(self):
        """Reentrena el modelo con todos los datos acumulados."""
        if len(self.feature_history) < 2:
            return
        
        # Convertir a matriz
        X = pd.DataFrame(self.feature_history).values
        
        # Escalar
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        
        # Reducir dimensionalidad si hay muchas features
        if X.shape[1] > 10:
            pca = PCA(n_components=8)
            X_reduced = pca.fit_transform(X_scaled)
        else:
            X_reduced = X_scaled
        
        # Clustering
        self.model = KMeans(n_clusters=min(self.n_clusters, len(X)), 
                           random_state=RANDOM_STATE, n_init=10)
        self.model.fit(X_reduced)
        self.is_fitted = True
        
        print(f"🔄 Modelo reentrenado con {len(self.feature_history)} registros")
    
    def predict(self, features):
        """Predice cluster para nuevos features."""
        if not self.is_fitted:
            return None
        
        X = np.array(list(features.values())).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return self.model.predict(X_scaled)[0]
    
    def get_cluster_statistics(self):
        """Estadísticas por cluster."""
        if not self.is_fitted or len(self.feature_history) < 2:
            return {}
        
        X = pd.DataFrame(self.feature_history).values
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        
        stats = {}
        for i in range(self.n_clusters):
            mask = labels == i
            if np.any(mask):
                records_in_cluster = [self.record_history[j] for j, m in enumerate(mask) if m]
                stats[i] = {
                    'size': np.sum(mask),
                    'records': records_in_cluster,
                    'percentage': (np.sum(mask) / len(labels)) * 100
                }
        return stats


# =============================================================================
# EVALUADOR INTERACTIVO
# =============================================================================

class InteractiveECGEvaluator:
    """
    Sistema interactivo para evaluación dinámica de registros ECG.
    """
    
    def __init__(self):
        self.catalog = MITBIHCatalog()
        self.loader = ECGDataLoader()
        self.preprocessor = ECGPreprocessor()
        self.extractor = ECGFeatureExtractor()
        self.clustering = DynamicClusteringSystem()
        self.results_db = {}
        
        # Crear directorio de resultados
        self.results_dir = Path('ecg_results')
        self.results_dir.mkdir(exist_ok=True)
    
    def show_menu(self):
        """Muestra menú principal."""
        stats = self.catalog.get_statistics()
        
        print("\n" + "="*70)
        print("  SISTEMA DE EVALUACIÓN ECG - MENÚ PRINCIPAL")
        print("="*70)
        print(f"📊 ESTADÍSTICAS:")
        print(f"   Total registros MIT-BIH: {stats['total_available']}")
        print(f"   Procesados anteriormente: {stats['total_processed']} ({stats['percentage_used']:.1f}%)")
        print(f"   Disponibles: {stats['remaining']}")
        print(f"   Sesión actual: {stats['current_session']}")
        print("-"*70)
        print("OPCIONES:")
        print("  1. 🔍 Evaluar registro específico (por ID)")
        print("  2. 📁 Evaluar por categoría (Normal, PVC, APC, etc.)")
        print("  3. 🎲 Evaluar registro aleatorio no procesado")
        print("  4. 📊 Ver registros ya procesados")
        print("  5. 📈 Ver análisis comparativo de clusters")
        print("  6. 💾 Exportar resultados de sesión")
        print("  7. 🧹 Limpiar historial y empezar de nuevo")
        print("  8. ❌ Salir")
        print("="*70)
        
        choice = input("Seleccione opción (1-8): ").strip()
        return choice
    
    def evaluate_specific_record(self):
        """Evalúa un registro específico por ID."""
        record_id = input("\nIngrese ID del registro (ej. 100, 203, 117): ").strip()
        
        if record_id not in self.catalog.ALL_RECORDS:
            print(f"❌ Registro {record_id} no existe en MIT-BIH")
            return
        
        info = self.catalog.get_record_info(record_id)
        print(f"\n📋 Info: Categoría={info['category']}, Procesado={info['processed']}")
        
        if info['processed'] and not input("¿Reprocesar? (s/n): ").lower().startswith('s'):
            return
        
        self._process_record(record_id)
    
    def evaluate_by_category(self):
        """Muestra registros por categoría."""
        print("\nCATEGORÍAS DISPONIBLES:")
        for i, cat in enumerate(self.catalog.RECORD_CATEGORIES.keys(), 1):
            available = len(self.catalog.get_available_records(category=cat))
            print(f"  {i}. {cat} ({available} disponibles)")
        
        choice = input("\nSeleccione categoría (número o nombre): ").strip()
        
        # Mapear selección
        categories = list(self.catalog.RECORD_CATEGORIES.keys())
        try:
            if choice.isdigit():
                cat = categories[int(choice)-1]
            else:
                cat = choice
        except:
            print("❌ Categoría inválida")
            return
        
        available = self.catalog.get_available_records(category=cat)
        print(f"\n📋 Registros disponibles en {cat}: {available[:10]}...")
        print(f"   Total: {len(available)}")
        
        if available:
            record_id = available[0]  # Tomar el primero
            print(f"🎯 Evaluando: {record_id}")
            self._process_record(record_id)
    
    def evaluate_random(self):
        """Evalúa registro aleatorio no procesado."""
        available = self.catalog.get_available_records()
        
        if not available:
            print("⚠️  Todos los registros han sido procesados")
            print("   Use opción 7 para limpiar historial o evalúe por categoría")
            return
        
        record_id = np.random.choice(available)
        print(f"\n🎲 Registro aleatorio seleccionado: {record_id}")
        self._process_record(record_id)
    
    def show_processed_records(self):
        """Muestra registros ya procesados."""
        if not self.catalog.processed_records:
            print("\n📭 No hay registros procesados aún")
            return
        
        print(f"\n📊 REGISTROS PROCESADOS ({len(self.catalog.processed_records)}):")
        
        # Agrupar por categoría
        by_category = {}
        for rec in sorted(self.catalog.processed_records):
            info = self.catalog.get_record_info(rec)
            cat = info['category'] or 'Desconocida'
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(rec)
        
        for cat, records in by_category.items():
            print(f"\n  {cat}:")
            print(f"    {', '.join(records)}")
        
        # Mostrar sesión actual
        if self.catalog.current_session_records:
            print(f"\n🆕 SESIÓN ACTUAL ({len(self.catalog.current_session_records)}):")
            print(f"    {', '.join(sorted(self.catalog.current_session_records))}")
    
    def show_cluster_analysis(self):
        """Muestra análisis de clusters."""
        if not self.clustering.is_fitted:
            print("\n⚠️  Aún no hay suficientes datos para clustering")
            print("   Procese al menos 2 registros")
            return
        
        stats = self.clustering.get_cluster_statistics()
        
        print(f"\n📈 ANÁLISIS DE CLUSTERS:")
        print(f"Total registros en modelo: {len(self.clustering.feature_history)}")
        print("-" * 50)
        
        for cluster_id, info in stats.items():
            print(f"\n🔷 CLUSTER {cluster_id}:")
            print(f"   Tamaño: {info['size']} registros ({info['percentage']:.1f}%)")
            print(f"   Registros: {', '.join(info['records'])}")
            
            # Interpretación médica básica
            if info['percentage'] > 50:
                interpretation = "Ritmo Normal (dominante)"
            elif info['size'] == 1:
                interpretation = "Anomalía única / Outlier"
            else:
                interpretation = "Subgrupo de arritmias"
            print(f"   Interpretación: {interpretation}")
    
    def export_session_results(self):
        """Exporta resultados de la sesión actual."""
        if not self.catalog.current_session_records:
            print("\n⚠️  No hay registros en la sesión actual")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"session_{timestamp}.json"
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'records_processed': list(self.catalog.current_session_records),
            'clustering_stats': self.clustering.get_cluster_statistics() if self.clustering.is_fitted else {},
            'individual_results': {}
        }
        
        for rec_id in self.catalog.current_session_records:
            if rec_id in self.results_db:
                export_data['individual_results'][rec_id] = self.results_db[rec_id]
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"\n💾 Resultados exportados a: {filename}")
        
        # También crear CSV resumen
        csv_file = self.results_dir / f"summary_{timestamp}.csv"
        rows = []
        for rec_id, data in self.results_db.items():
            if rec_id in self.catalog.current_session_records:
                row = {
                    'record_id': rec_id,
                    'category': self.catalog.get_record_info(rec_id)['category'],
                    'cluster': data.get('cluster', 'N/A'),
                    'n_beats': data.get('n_beats', 0),
                    'rr_mean': data.get('rr_mean', 0),
                    'rr_std': data.get('rr_std', 0)
                }
                rows.append(row)
        
        if rows:
            pd.DataFrame(rows).to_csv(csv_file, index=False)
            print(f"📄 Resumen CSV: {csv_file}")
    
    def clear_history(self):
        """Limpia historial de procesamiento."""
        confirm = input("\n⚠️  ¿Eliminar TODO el historial? (escriba 'SI' para confirmar): ")
        if confirm == 'SI':
            self.catalog.processed_records.clear()
            self.catalog.current_session_records.clear()
            self.catalog.save_history()
            self.clustering = DynamicClusteringSystem()  # Resetear clustering
            self.results_db.clear()
            print("✅ Historial eliminado. Sistema reiniciado.")
        else:
            print("❌ Cancelado")
    
    def _process_record(self, record_id):
        """Procesa un registro completo."""
        print(f"\n{'='*60}")
        print(f"PROCESANDO REGISTRO {record_id}")
        print(f"{'='*60}")
        
        # 1. Cargar
        print("📥 Cargando datos...")
        data = self.loader.load_record(record_id)
        if not data:
            return
        
        print(f"   Duración: {data['duration']:.1f}s")
        print(f"   Muestras: {len(data['signal'])}")
        print(f"   Latidos anotados: {data['n_beats']}")
        
        # 2. Preprocesar
        print("🔧 Preprocesando...")
        filtered = self.preprocessor.bandpass_filter(data['signal'])
        peaks = self.preprocessor.pan_tompkins(filtered)
        beats, valid_peaks = self.preprocessor.segment_beats(filtered, peaks)
        
        print(f"   Picos R detectados: {len(peaks)}")
        print(f"   Latidos válidos: {len(beats)}")
        
        # 3. Extraer features
        print("🧠 Extrayendo características...")
        features = self.extractor.extract_features(beats, valid_peaks)
        
        if not features:
            print("❌ No se pudieron extraer features")
            return
        
        for key, val in features.items():
            print(f"   {key}: {val:.3f}" if isinstance(val, float) else f"   {key}: {val}")
        
        # 4. Clustering
        print("🎯 Clasificando...")
        self.clustering.partial_fit(features, record_id)
        cluster = self.clustering.predict(features)
        
        print(f"   Cluster asignado: {cluster}")
        
        # 5. Guardar resultados
        self.results_db[record_id] = {
            'timestamp': datetime.now().isoformat(),
            'cluster': int(cluster) if cluster is not None else None,
            **features
        }
        
        # 6. Marcar como procesado
        self.catalog.mark_as_processed(record_id)
        
        # 7. Visualización rápida
        self._quick_visualization(record_id, data['signal'], filtered, peaks, beats)
        
        print(f"\n✅ Registro {record_id} procesado exitosamente")
    
    def _quick_visualization(self, record_id, raw, filtered, peaks, beats):
        """Genera visualización rápida del registro."""
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # Señal cruda
        t = np.arange(len(raw)) / 360
        axes[0].plot(t, raw, 'gray', alpha=0.7, label='Raw')
        axes[0].set_title(f'Registro {record_id} - Señal Cruda')
        axes[0].set_ylabel('mV')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Señal filtrada con picos
        axes[1].plot(t, filtered, 'b-', linewidth=0.8, label='Filtered')
        axes[1].plot(peaks/360, filtered[peaks], 'ro', markersize=4, label='R-peaks')
        axes[1].set_title('Señal Filtrada + Detección QRS')
        axes[1].set_ylabel('mV')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Latidos superpuestos
        if len(beats) > 0:
            beat_len = len(beats[0])
            t_beat = np.arange(beat_len) / 360
            for i, beat in enumerate(beats[:50]):  # Max 50 latidos
                axes[2].plot(t_beat, beat, alpha=0.3, color='blue')
            axes[2].set_title(f'Superposición de Latidos (n={min(len(beats), 50)})')
            axes[2].set_xlabel('Tiempo (s)')
            axes[2].set_ylabel('Amplitud normalizada')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Guardar
        save_path = self.results_dir / f"{record_id}_analysis.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   📊 Gráfico guardado: {save_path}")
        
        plt.show(block=False)
        plt.pause(0.1)  # No bloquear ejecución
    
    def run(self):
        """Bucle principal del sistema."""
        print("""
        ╔══════════════════════════════════════════════════════════════════╗
        ║     SISTEMA DINÁMICO DE EVALUACIÓN ECG - MIT-BIH                 ║
        ║     Clustering No Supervisado en Tiempo Real                     ║
        ╚══════════════════════════════════════════════════════════════════╝
        """)
        
        while True:
            choice = self.show_menu()
            
            if choice == '1':
                self.evaluate_specific_record()
            elif choice == '2':
                self.evaluate_by_category()
            elif choice == '3':
                self.evaluate_random()
            elif choice == '4':
                self.show_processed_records()
            elif choice == '5':
                self.show_cluster_analysis()
            elif choice == '6':
                self.export_session_results()
            elif choice == '7':
                self.clear_history()
            elif choice == '8':
                print("\n👋 ¡Hasta luego!")
                break
            else:
                print("\n❌ Opción inválida")
            
            input("\nPresione Enter para continuar...")


# =============================================================================
# EJECUCIÓN
# =============================================================================

if __name__ == "__main__":
    evaluator = InteractiveECGEvaluator()
    evaluator.run()