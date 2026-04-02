"""Visualizacion de senales ECG: raw, filtradas, latidos."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_raw_vs_filtered(
    raw: np.ndarray,
    filtered: np.ndarray,
    fs: int = 360,
    duration_seconds: float = 5.0,
    title: str = "Senal ECG: Cruda vs Filtrada",
    save_path: Optional[str] = None,
) -> None:
    """Grafica comparativa de senal cruda y filtrada."""
    n_samples = int(duration_seconds * fs)
    t = np.arange(min(n_samples, len(raw))) / fs

    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    axes[0].plot(t, raw[: len(t)], color="#2E86AB", linewidth=0.5)
    axes[0].set_title("Senal cruda")
    axes[0].set_ylabel("Amplitud (mV)")

    axes[1].plot(t, filtered[: len(t)], color="#A23B72", linewidth=0.5)
    axes[1].set_title("Senal filtrada (Butterworth 0.5-40 Hz)")
    axes[1].set_ylabel("Amplitud (mV)")
    axes[1].set_xlabel("Tiempo (s)")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_signal_with_peaks(
    signal: np.ndarray,
    peaks: np.ndarray,
    fs: int = 360,
    duration_seconds: float = 5.0,
    title: str = "Deteccion de picos R",
    save_path: Optional[str] = None,
) -> None:
    """Grafica senal ECG con picos R marcados."""
    n_samples = int(duration_seconds * fs)
    t = np.arange(min(n_samples, len(signal))) / fs
    sig = signal[: len(t)]

    valid_peaks = peaks[peaks < len(t)]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, sig, color="#2E86AB", linewidth=0.5, label="ECG")
    ax.plot(
        valid_peaks / fs,
        sig[valid_peaks],
        "rv",
        markersize=8,
        label=f"Picos R ({len(valid_peaks)})",
    )
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Amplitud")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_beat_overlay(
    segments: np.ndarray,
    n_beats: int = 50,
    fs: int = 360,
    title: str = "Superposicion de latidos",
    save_path: Optional[str] = None,
) -> None:
    """Superpone multiples latidos segmentados para visualizar morfologia."""
    n = min(n_beats, len(segments))
    t = np.arange(segments.shape[1]) / fs * 1000  # ms

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(n):
        ax.plot(t, segments[i], alpha=0.15, color="#2E86AB", linewidth=0.5)

    # Media de todos los latidos
    ax.plot(t, np.mean(segments[:n], axis=0), color="#F18F01", linewidth=2, label="Media")

    ax.set_xlabel("Tiempo (ms)")
    ax.set_ylabel("Amplitud normalizada")
    ax.set_title(f"{title} (n={n})")
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
