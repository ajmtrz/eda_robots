import traceback
from threading import Lock
from numba import jit, njit, prange
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import onnxruntime as rt
from catboost import CatBoostClassifier
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes
from modules.export_lib import (
    skl2onnx_parser_catboost_classifier,
    skl2onnx_convert_catboost,
)
from modules.labeling_lib import get_features
rt.set_default_logger_severity(4)

@njit(cache=True, fastmath=True)
def process_data(close, labels, metalabels, meta_thr=0.5):
    last_deal  = 2
    last_price = 0.0
    report, chart = [0.0], [0.0]

    for i in range(len(close)):
        pred, pr, pred_meta = labels[i], close[i], metalabels[i]

        # ── abrir posición
        if last_deal == 2 and pred_meta > meta_thr:
            last_price = pr
            last_deal  = 0 if pred < 0.5 else 1
            continue

        # ── cerrar por señal opuesta
        if last_deal == 0 and pred > 0.5:
            last_deal = 2
            profit = (pr - last_price)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + profit)
            continue

        if last_deal == 1 and pred < 0.5:
            last_deal = 2
            profit = (last_price - pr)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + (pr - last_price))
            continue

    # Cierre forzoso al final si sigue abierta
    if last_deal == 0:
        profit = close[-1] - last_price
        report.append(report[-1] + profit)
        chart.append(chart[-1] + profit)
    elif last_deal == 1:
        profit = last_price - close[-1]
        report.append(report[-1] + profit)
        chart.append(chart[-1] + (close[-1] - last_price))

    return np.array(report), np.array(chart)

@njit(cache=True, fastmath=True)
def process_data_one_direction(close, main_labels, meta_labels, direction):
    last_deal  = 2            # 2 = flat, 1 = position open
    last_price = 0.0
    report = [0.0]
    chart  = [0.0]
    long_side = (direction == 'buy')
    min_prob  = 0.5

    for i in range(close.size):
        pred_main = main_labels[i]
        pr        = close[i]
        pred_meta = meta_labels[i]

        # ── abrir posición ───────────────────────────────
        if last_deal == 2 and pred_meta > min_prob and pred_main > min_prob:
            last_deal  = 1
            last_price = pr
            continue

        # ── cerrar posición ──────────────────────────────
        if last_deal == 1 and pred_main < min_prob:
            last_deal = 2
            profit = (pr - last_price) if long_side else (last_price - pr)
            report.append(report[-1] + profit)
            chart.append(chart[-1]  + profit)

    # Cierre forzoso al final si sigue abierta
    if last_deal == 1:
        profit = (close[-1] - last_price) if long_side else (last_price - close[-1])
        report.append(report[-1] + profit)
        chart.append(chart[-1]  + profit)

    return np.asarray(report, dtype=np.float64), np.asarray(chart, dtype=np.float64)


# ───────────────────────────────────────────────────────────────────
# 2)  Wrappers del tester
# ───────────────────────────────────────────────────────────────────
def tester(
        ds_main: np.ndarray,
        ds_meta: np.ndarray,
        close: np.ndarray,
        model_main: object,
        model_meta: object,
        direction: str = 'both',
        plot: bool = False,
        prd: str = '') -> float:

    """Evalúa una estrategia para una o ambas direcciones.

    Parameters
    ----------
    ds_main : np.ndarray
        Conjunto de características para el modelo principal.
    ds_meta : np.ndarray
        Conjunto de características para el meta-modelo.
    close : np.ndarray
        Serie de precios de cierre.
    model_main : object
        Modelo principal entrenado con ``predict_proba``.
    model_meta : object
        Meta-modelo entrenado con ``predict_proba``.
    direction : str, optional
        ``'buy'``, ``'sell'`` o ``'both'``. Por defecto ``'both'``.
    plot : bool, optional
        Si ``True`` muestra la curva de equity.  Por defecto ``False``.
    prd : str, optional
        Etiqueta del periodo a mostrar en el gráfico.

    Returns
    -------
    float
        Puntuación de la estrategia según :func:`evaluate_report`.
    """

    # Calcular probabilidades usando ambos modelos (sin binarizar)
    main = model_main.predict_proba(ds_main)[:, 1]
    meta = model_meta.predict_proba(ds_meta)[:, 1]

    # Asegurar contigüidad en memoria
    close = np.ascontiguousarray(close)
    main = np.ascontiguousarray(main)
    meta = np.ascontiguousarray(meta)

    if direction == 'both':
        rpt, _ = process_data(close, main, meta)
    else:
        rpt, _ = process_data_one_direction(close, main, meta, direction)

    if rpt.size < 2:
        return -1.0

    score = evaluate_report(rpt)

    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(rpt, label='Equity Curve')
        plt.xlabel("Operations")
        plt.ylabel("Cumulative Profit")
        if prd:
            plt.title(f"Period: {prd} | Score: {score:.2f}")
        else:
            plt.title(f"Score: {score:.2f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    return score

@njit(cache=True, fastmath=True)
def max_gap_between_highs(equity: np.ndarray) -> int:
    last_high = equity[0]
    gap = 0
    max_gap = 0
    for x in equity:
        if x > last_high:
            last_high = x
            if gap > max_gap:
                max_gap = gap
            gap = 0
        else:
            gap += 1
    if gap > max_gap:
        max_gap = gap
    return max_gap

@njit(cache=True, fastmath=True)
def evaluate_report(report: np.ndarray) -> float:
    if report.size < 3:
        return -1.0

    ret = np.diff(report)
    n   = ret.size
    if n < 5:
        return -1.0                     # curva demasiado corta

    # 1) slope * signed-R²
    sr2 = _signed_r2(report)           # ∈ [-1,1]
    if sr2 <= 0.0:                     # exigimos pendiente positiva
        return -1.0
    sr2_norm = 0.5 * (sr2 + 1.0)       # → [0,1]

    # 2) profit factor y return/DD
    gains  = ret[ret > 0.0].sum()
    losses = -ret[ret < 0.0].sum()
    if losses == 0.0:
        losses = 1e-9
    pf  = gains / losses
    pf_norm  = 1.0 - np.exp(-pf / 2.0)

    peak = report[0]
    max_dd = 0.0
    for x in report:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > max_dd:
            max_dd = dd
    rdd = (report[-1] - report[0]) / (max_dd + 1e-9)
    rdd_norm = 1.0 - np.exp(-rdd / 4.0)

    # 3) expected payoff por trade
    exp_payoff = ret.mean()
    payoff_norm = (np.tanh(exp_payoff * 100.0) + 1.0) / 2.0

    # 4) estancamiento
    stagn = max_gap_between_highs(report)
    stagn_pen = 1.0 / (1.0 + stagn / 1000.0)

    # 5) bonus de trades (logística)
    TRG   = 300.0        # n trades donde bonus = 0.5
    SCALE = 100.0        # controla la pendiente
    bonus = 1.0 / (1.0 + np.exp(-(n - TRG) / SCALE))

    # 6) score agregado (ponderaciones iguales)
    core  = (pf_norm + rdd_norm + sr2_norm + payoff_norm) / 4.0
    score = core * bonus * stagn_pen

    return score

def tester_one_direction(
        ds_main: np.ndarray,
        ds_meta: np.ndarray,
        close: np.ndarray,
        model_main: object,
        model_meta: object,
        direction: str = 'buy',
        plot: bool = False,
        prd: str = '') -> float:
    """Mantiene compatibilidad con la API anterior."""
    return tester(
        ds_main=ds_main,
        ds_meta=ds_meta,
        close=close,
        model_main=model_main,
        model_meta=model_meta,
        direction=direction,
        plot=plot,
        prd=prd,
    )

# ─────────────────────────────────────────────
# tester_slow  (mantenerlo o borrarlo)
# ─────────────────────────────────────────────
def tester_slow(dataset, markup, plot=False):
    last_deal = 2
    last_price = 0.0
    report, chart = [0.0], [0.0]

    close = dataset['close'].to_numpy()
    main = dataset['labels_main'].to_numpy()
    metalabels = dataset['labels_meta'].to_numpy()

    for i in range(dataset.shape[0]):
        pred, pr, pred_meta = main[i], close[i], metalabels[i]

        if last_deal == 2 and pred_meta == 1:
            last_price = pr
            last_deal  = 0 if pred < 0.5 else 1
            continue

        if last_deal == 0 and pred > 0.5:
            last_deal = 2
            profit = -markup + (pr - last_price)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + profit)
            continue

        if last_deal == 1 and pred < 0.5:
            last_deal = 2
            profit = -markup + (last_price - pr)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + (pr - last_price))
            continue

    y = np.array(report).reshape(-1, 1)
    X = np.arange(len(report)).reshape(-1, 1)
    lr = _signed_r2(report)

    l = 1 if lr >= 0 else -1

    if plot:
        plt.plot(report)
        plt.plot(chart)
        plt.plot(lr.predict(X))
        plt.title("Strategy performance R^2 " + str(format(lr.score(X, y) * l, ".2f")))
        plt.xlabel("the number of trades")
        plt.ylabel("cumulative profit in pips")
        plt.show()

    return lr.score(X, y) * l

# ─────────────────────────────────────────────
# Wrappers
# ─────────────────────────────────────────────
def test_model(dataset: pd.DataFrame,
               result: list,
               backward: datetime,
               forward: datetime,
               plt: bool = False) -> float:

    ext_dataset = dataset.copy()
    mask = (ext_dataset.index > backward) & (ext_dataset.index < forward)
    ext_dataset = ext_dataset[mask].reset_index(drop=True)

    X = ext_dataset.iloc[:, 1:].to_numpy()
    close = ext_dataset['close'].to_numpy()

    return tester(
        ds_main=X,
        ds_meta=X,
        close=close,
        model_main=result[0],
        model_meta=result[1],
        direction='both',
        plot=plt,
    )

# ───────────────────────────────────────────────────────────────────
# Monte-Carlo robustness utilities
# ───────────────────────────────────────────────────────────────────

# ---------- helpers ------------------------------------------------
@njit(cache=True, fastmath=True)
def _bootstrap_returns(returns: np.ndarray,
                       block_size: int) -> np.ndarray:
    """
    Resamplea los *returns* preservando (opcionalmente) dependencia local
    mediante bootstrapping por bloques.
    """
    n = returns.shape[0]
    resampled = np.empty_like(returns)
    
    if block_size <= 1 or block_size > n:
        # Bootstrap simple
        for i in range(n):
            resampled[i] = returns[np.random.randint(0, n)]
        return resampled
    
    # Llenar el array resampled bloque por bloque
    pos = 0
    while pos < n:
        # Seleccionar inicio del bloque
        start = np.random.randint(0, n - block_size + 1)
        
        # Calcular cuántos elementos podemos copiar
        remaining = n - pos
        to_copy = min(block_size, remaining)
        
        # Copiar el bloque
        resampled[pos:pos + to_copy] = returns[start:start + to_copy]
        pos += to_copy
    
    return resampled


@njit(cache=True, fastmath=True)
def _equity_from_returns(resampled_returns: np.ndarray) -> np.ndarray:
    """Crea curva de equity partiendo de 0."""
    n = resampled_returns.size
    equity = np.empty(n + 1, dtype=np.float64)
    equity[0] = 0.0
    cumsum = 0.0
    for i in range(n):
        cumsum += resampled_returns[i]
        equity[i + 1] = cumsum
    return equity


@njit(cache=True, fastmath=True)
def _signed_r2(equity: np.ndarray) -> float:
    n = equity.size
    x_mean = (n - 1) * 0.5
    y_mean = equity.mean()
    cov = varx = 0.0
    for i in range(n):
        dx = i - x_mean
        cov += dx * (equity[i] - y_mean)
        varx += dx * dx
    slope = cov / varx if varx else 0.0
    sse = sst = 0.0
    for i in range(n):
        pred = y_mean + slope * (i - x_mean)
        diff = equity[i] - pred
        sse += diff * diff
        diff2 = equity[i] - y_mean
        sst += diff2 * diff2
    r2 = 1.0 - sse / sst if sst else 0.0
    return r2 if slope >= 0 else -r2

@njit(cache=True, fastmath=True)
def _make_noisy_signals(close: np.ndarray,
                       labels: np.ndarray,
                       meta: np.ndarray) -> tuple:
    """Añade ruido a precios, labels y meta-labels."""
    n = close.size
    volatility = np.std(np.diff(close) / close[:-1])
    price_noise_range = (volatility * 0.5, volatility * 2.0)

    # ------ precios ------------------------------------------------
    # Añadir ruido a los precios para simular slippage y volatilidad
    price_noise = np.random.uniform(price_noise_range[0], price_noise_range[1])
    close_noisy = np.empty_like(close)
    if price_noise > 0:
        for i in range(n):
            close_noisy[i] = close[i] * (1 + np.random.laplace(0.0, price_noise))
    else:
        close_noisy[:] = close

    # Labels y meta-labels se mantienen sin distorsionar
    return close_noisy, labels, meta

def _make_noisy_close(close: np.ndarray) -> np.ndarray:
    """Devuelve una serie de precios alterada con ruido laplaciano."""
    n = close.size
    if n < 2:
        return close.copy()
    volatility = np.std(np.diff(close) / close[:-1])
    price_noise = np.random.uniform(volatility * 0.5, volatility * 2.0)
    if price_noise <= 0:
        return close.copy()
    noise = np.random.laplace(0.0, price_noise, size=n)
    return close * (1.0 + noise)

@njit(cache=True, fastmath=True, parallel=True)
def _simulate_batch(close, l_all, m_all, block_size, direction):
    n_sim, n = l_all.shape
    scores = np.full(n_sim, -1.0)
    for i in prange(n_sim):
        c_n, l_n, m_n = _make_noisy_signals(close, l_all[i], m_all[i])
        if direction == "both":
            rpt, _ = process_data(c_n, l_n, m_n)
        else:
            rpt, _ = process_data_one_direction(c_n, l_n, m_n, direction)
        if rpt.size < 2:
            continue
        ret = np.diff(rpt)
        if ret.size == 0:
            continue
        eq = _equity_from_returns(_bootstrap_returns(ret, block_size))
        score = evaluate_report(eq)
        scores[i] = score
    return scores

# -----------------------------------------------------------------------------
#  ONNX-accelerated batch prediction
# -----------------------------------------------------------------------------
update_registered_converter(
    CatBoostClassifier,
    "CatBoostClassifier",
    calculate_linear_classifier_output_shapes,
    skl2onnx_convert_catboost,
    parser=skl2onnx_parser_catboost_classifier,
    options={"nocl": [True, False], "zipmap": [True, False]}
)

def _predict_batch(model, X_base, noise_levels):
    try:
        # 1. Convertir modelo a ONNX
        onnx_model = convert_sklearn(
            model,
            initial_types=[('x', FloatTensorType([None, X_base.shape[1]]))],
            target_opset={"": 18, "ai.onnx.ml": 2},
            options={id(model): {'zipmap': False}}
        )
        sess = rt.InferenceSession(
            onnx_model.SerializeToString(),
            providers=['CPUExecutionProvider']
        )
        iname = sess.get_inputs()[0].name
        onnx_run = sess.run

        # 2. Preparar los datos base
        n_sim, n_samples, n_feat = len(noise_levels), *X_base.shape
        X_big = np.repeat(X_base[None, :, :], n_sim, axis=0)
        
        # 3. Calcular sensibilidad de cada feature
        fi = model.get_feature_importance()
        max_fi = fi.max() or 1.0
        sensitivity = fi / max_fi
        
        # 4. Generar y aplicar el ruido ajustado
        std = X_base.std(axis=0, keepdims=True)
        eps = np.random.normal(0.0, std, size=X_big.shape)
        adjusted_noise = noise_levels[:, None, None] * sensitivity[None, None, :]
        X_big += eps * adjusted_noise

        # 5. Realizar la predicción
        proba_flat = onnx_run(None, {iname: X_big.reshape(-1, n_feat).astype(np.float32)})[0]
        proba = proba_flat.reshape(n_sim, n_samples)
        return proba
    except Exception as e:
        print(f"\nError en _predict_batch:")
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
    finally:
        if 'sess' in locals():
            sess = None
        else:
            print("No session to close.")

def monte_carlo_full(
    model_main: object,
    model_meta: object,
    ds_test: pd.DataFrame,
    hp: dict,
    direction: str = "both",
    n_sim: int = 100,
    block_size: int = 20,
    plot: bool = False,
    prd: str = "",
) -> dict:
    """Lanza Monte Carlo alterando solo el cierre y recalculando las features."""

    # extraer serie de cierre original
    close = np.ascontiguousarray(ds_test["close"].to_numpy())

    try:
        base_df = ds_test.copy()
        feats = get_features(base_df, dict(hp))
        if feats.empty:
            return {"scores": np.array([-1.0]), "p_positive": 0.0, "quantiles": np.array([-1.0, -1.0, -1.0])}

        X_main = feats.filter(regex=r"(?<!_meta)_feature$").to_numpy()
        X_meta = feats.filter(like="_meta_feature").to_numpy()
        close_feat = feats["close"].to_numpy()

        main = model_main.predict_proba(X_main)[:, 1]
        meta = model_meta.predict_proba(X_meta)[:, 1]
        if direction == "both":
            rpt_original, _ = process_data(close_feat, main, meta)
        else:
            rpt_original, _ = process_data_one_direction(close_feat, main, meta, direction)

        scores = np.full(n_sim, -1.0)
        equity_curves = [] if plot else None

        for i in range(n_sim):
            c_n = _make_noisy_close(close)
            df_n = ds_test.copy()
            df_n["close"] = c_n
            feats_n = get_features(df_n, dict(hp))
            if feats_n.empty:
                continue
            X_m = feats_n.filter(regex=r"(?<!_meta)_feature$").to_numpy()
            X_me = feats_n.filter(like="_meta_feature").to_numpy()
            c_feat = feats_n["close"].to_numpy()

            m_main = model_main.predict_proba(X_m)[:, 1]
            m_meta = model_meta.predict_proba(X_me)[:, 1]

            if direction == "both":
                rpt, _ = process_data(c_feat, m_main, m_meta)
            else:
                rpt, _ = process_data_one_direction(c_feat, m_main, m_meta, direction)

            if len(rpt) < 2:
                continue
            ret = np.diff(rpt)
            if ret.size == 0:
                continue
            eq = _equity_from_returns(_bootstrap_returns(ret, block_size))
            scores[i] = evaluate_report(eq)
            if equity_curves is not None:
                equity_curves.append(rpt)

        valid = scores[scores > 0.0]

        if plot:
            try:
                import matplotlib.pyplot as plt

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])

                if valid.size > 0 and equity_curves:
                    for eq in equity_curves:
                        ax1.plot(eq, color='gray', alpha=0.1)
                    ax1.plot(rpt_original, color='blue', linewidth=2, label='Original')

                    min_len = min(len(curve) for curve in equity_curves)
                    curves_array = np.array([curve[:min_len] for curve in equity_curves])
                    p05 = np.percentile(curves_array, 5, axis=0)
                    p95 = np.percentile(curves_array, 95, axis=0)
                    median = np.median(curves_array, axis=0)
                    ax1.plot(p05, 'r--', alpha=0.5, label='5%')
                    ax1.plot(p95, 'r--', alpha=0.5, label='95%')
                    ax1.plot(median, 'g-', alpha=0.5, label='Mediana')

                    if prd:
                        ax1.set_title(f'Simulaciones Monte Carlo - {prd}\n(n={valid.size}, {direction})')
                    else:
                        ax1.set_title(f'Simulaciones Monte Carlo\n(n={valid.size}, {direction})')
                else:
                    ax1.text(0.5, 0.5, 'No hay simulaciones válidas', ha='center', va='center')

                ax1.set_xlabel('Operaciones')
                ax1.set_ylabel('Beneficio Acumulado')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                if valid.size > 0:
                    ax2.hist(valid, bins=30, density=True, alpha=0.6, color='skyblue')
                    ax2.axvline(np.median(valid), color='red', linestyle='--', label=f'Mediana: {np.median(valid):.2f}')
                    q05, q50, q95 = np.quantile(valid, [0.05, 0.5, 0.95])
                    ax2.axvline(q05, color='orange', linestyle=':', label=f'Q05: {q05:.2f}')
                    ax2.axvline(q95, color='green', linestyle=':', label=f'Q95: {q95:.2f}')

                ax2.set_xlabel('Score')
                ax2.set_ylabel('Densidad')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.show()

            except Exception as viz_error:
                print(f"Error en visualización: {viz_error}")

        return {
            "scores": scores,
            "p_positive": np.mean(scores > 0),
            "quantiles": np.quantile(valid, [0.05, 0.5, 0.95]) if valid.size else [-1, -1, -1],
        }
    except Exception as e:
        print(f"\nError en monte_carlo_full:")
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())


def robust_oos_score(
        ds_test: pd.DataFrame,
        model_main: object,
        model_meta: object,
        hp: dict,
        direction: str = "both",
        n_sim: int = 100,
        block_size: int = 20,
        agg: str = "q05",
        plot: bool = False,
        prd: str = "") -> float:

    """Calcula un score robusto en out-of-sample mediante Monte Carlo."""

    try:
        mc = monte_carlo_full(
            model_main=model_main,
            model_meta=model_meta,
            ds_test=ds_test,
            hp=hp,
            direction=direction,
            n_sim=n_sim,
            block_size=block_size,
            plot=plot,
            prd=prd,
        )

        if agg == "q05":
            return float(mc["quantiles"][0])
        elif agg == "q50":
            return float(mc["quantiles"][1])
        elif agg == "q95":
            return float(mc["quantiles"][2])
        else:
            raise ValueError("agg must be 'q05', 'q50' or 'q95'")

    except Exception as e:
        print(f"\nError en robust_oos_score: {e}")
        return -1.0

def walk_forward_robust_score(
    ds_test: pd.DataFrame,
    model_main: object,
    model_meta: object,
    hp: dict,
    direction: str = "both",
    n_splits: int = 3,
    agg: str = "min",
    plot: bool = False,
) -> float:
    """Calcula un score OOS mediante validación walk-forward."""

    try:
        n = len(ds_test)
        if n_splits <= 1 or n <= 1:
            return robust_oos_score(
                ds_test=ds_test,
                model_main=model_main,
                model_meta=model_meta,
                hp=hp,
                direction=direction,
                plot=plot,
                prd="oos",
            )

        split_idx = np.linspace(0, n, n_splits + 1, dtype=int)
        scores = []
        for i in range(n_splits):
            start = split_idx[i]
            end = split_idx[i + 1]
            if end - start < 2:
                continue
            s = robust_oos_score(
                ds_test=ds_test.iloc[start:end],
                model_main=model_main,
                model_meta=model_meta,
                hp=hp,
                direction=direction,
                plot=plot,
                prd=f"oos_{i+1}",
            )
            if np.isfinite(s):
                scores.append(s)

        if not scores:
            return -1.0

        if agg == "min":
            return float(np.min(scores))
        elif agg == "median":
            return float(np.median(scores))
        else:
            raise ValueError("agg must be 'min' or 'median'")

    except Exception as e:
        print(f"\nError en walk_forward_robust_score: {e}")
        return -1.0
