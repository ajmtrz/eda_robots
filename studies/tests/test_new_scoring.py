import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # backend para pruebas sin GUI
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from tester_lib import evaluate_report

PERIODS_PER_YEAR = 6240.0  # H1


def generate_curves(n=1000, seed=42):
    """Genera un conjunto de curvas de ejemplo con distintos comportamientos."""
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 100, n)

    # Curva ideal: tendencia clara y poco ruido
    smooth_noise = rng.normal(0, 1.5, n)
    ideal = np.cumsum(np.concatenate([[0], np.diff(trend) + smooth_noise[1:]]))

    # Curva volátil: fuerte tendencia pero con mucha volatilidad
    vol_trend = np.linspace(0, 400, n)
    volatile_noise = rng.normal(0, 8, n) + 0.5  # sesgo positivo para cierre alcista
    volatile = np.cumsum(np.concatenate([[0], np.diff(vol_trend) + volatile_noise[1:]]))

    # Curva con drawdown severo (colapso del 75 %)
    phase1 = np.linspace(0, 60, 400)
    phase2 = np.linspace(60, 15, 200)
    phase3 = np.linspace(15, 45, 400)
    dd_severe = np.concatenate([phase1, phase2, phase3])

    # Curva lateral: sin tendencia (termina cerca de cero)
    lateral_noise = rng.normal(0, 3, n)
    lateral_noise -= lateral_noise.mean()
    lateral = np.cumsum(lateral_noise)

    # Curva perdedora: tendencia negativa
    losing_trend = np.linspace(100, 30, n)
    losing_noise = rng.normal(0, 2, n)
    losing = np.cumsum(np.concatenate([[100], np.diff(losing_trend) + losing_noise[1:]]))

    # Curva moderada: buena tendencia con drawdown moderado
    moderate_noise = rng.normal(0, 1.5, n)
    moderate = np.cumsum(np.concatenate([[0], np.diff(trend) + moderate_noise[1:]]))
    moderate[400:600] -= np.linspace(0, 15, 200)
    moderate[600:] -= 15

    return {
        "ideal": ideal,
        "volatil": volatile,
        "dd_severo": dd_severe,
        "lateral": lateral,
        "perdedora": losing,
        "moderada": moderate,
    }


def plot_curves(curves, scores, path):
    plt.figure(figsize=(12, 7))
    titles = {
        name: f"{name} (Score: {score:.3f})" for name, score in scores.items()
    }
    for i, (name, curve) in enumerate(curves.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(curve)
        plt.title(titles[name])
        plt.grid(True)
        plt.xlabel("Períodos")
        plt.ylabel("P&L Acumulado")
    plt.tight_layout()    plt.savefig(path)
    plt.close()


def test_scoring_order(tmp_path):
    curves = generate_curves()
    scores = {}
    for name, curve in curves.items():
        score, _ = evaluate_report(curve, PERIODS_PER_YEAR)
        scores[name] = score

    # Guardar gráfico para inspección manual en caso de fallo
    plot_curves(curves, scores, tmp_path / "curves.png")

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ordered_names = [name for name, _ in ranking]

    expected = [
        "ideal",
        "moderada",
        "volatil",
        "dd_severo",
        "lateral",
        "perdedora",
    ]

    assert ordered_names == expected, f"Ranking obtenido {ordered_names} != {expected}"
