"""Visualizacion de clusters: PCA scatter, t-SNE, distribucion."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_pca_scatter(
    X_pca: np.ndarray,
    labels: np.ndarray,
    title: str = "Clusters en espacio PCA",
    anomaly_labels: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> None:
    """Scatter 2D usando las primeras dos componentes PCA.

    Args:
        X_pca: Features reducidas [N, >=2].
        labels: Etiquetas de clustering.
        title: Titulo del grafico.
        anomaly_labels: Si se proporcionan, usa colores normal/anomalo.
        save_path: Ruta para guardar la imagen.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    if anomaly_labels is not None:
        colors = np.where(anomaly_labels == 0, "#2E86AB", "#E74C3C")
        scatter = ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=colors,
            alpha=0.4,
            s=10,
        )
        # Leyenda manual
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2E86AB", markersize=8, label="Normal"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#E74C3C", markersize=8, label="Anomalia"),
        ]
        ax.legend(handles=legend_elements)
    else:
        unique_labels = sorted(set(labels))
        palette = sns.color_palette("husl", len(unique_labels))
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            name = "Ruido" if label == -1 else f"Cluster {label}"
            ax.scatter(
                X_pca[mask, 0],
                X_pca[mask, 1],
                c=[palette[idx]],
                alpha=0.4 if label >= 0 else 0.2,
                s=10 if label >= 0 else 5,
                label=name,
            )
        ax.legend(fontsize=8)

    ax.set_xlabel("Componente Principal 1")
    ax.set_ylabel("Componente Principal 2")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_anomaly_distribution(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    model_name: str = "",
    save_path: Optional[str] = None,
) -> None:
    """Distribucion comparativa de anomalias reales vs detectadas."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, data, title in [
        (axes[0], true_labels, "Ground Truth (AAMI)"),
        (axes[1], pred_labels, f"Prediccion ({model_name})"),
    ]:
        unique, counts = np.unique(data, return_counts=True)
        colors = ["#2E86AB" if u == 0 else "#E74C3C" for u in unique]
        labels = ["Normal" if u == 0 else "Anomalo" for u in unique]
        ax.bar(labels, counts, color=colors)
        ax.set_title(title)
        ax.set_ylabel("Cantidad de latidos")

        for i, (label, count) in enumerate(zip(labels, counts)):
            ax.text(i, count + 50, str(count), ha="center", fontsize=10)

    plt.suptitle("Distribucion de anomalias", fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
