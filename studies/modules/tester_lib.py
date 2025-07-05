import traceback
from numba import njit, prange
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import onnxruntime as rt
from functools import lru_cache
from modules.labeling_lib import get_features
rt.set_default_logger_severity(4)

def audit_index(
        idx: pd.DatetimeIndex,
        name: str = "train",
        show_examples: int = 5,
    ) -> None:
        """
        Comprueba orden, duplicados y huecos de un DatetimeIndex.
        Imprime detalle de los gaps distintos al paso modal.
        """
        assert idx.is_monotonic_increasing,  f"{name}: índice no ordenado"
        assert idx.is_unique,               f"{name}: índices duplicados"

        # diferencias entre ticks contiguos
        deltas = idx.to_series().diff().dropna()

        # paso modal → el considerado "normal"
        mode_delta = deltas.mode().iloc[0]

        # máscara de huecos
        gap_mask = deltas != mode_delta

        if not gap_mask.any():
            print(f"✅ {name}: sin huecos – paso constante = {mode_delta}")
        else:
            gap_counts = deltas[gap_mask].value_counts().sort_index()
            print(f"⚠️ {name}: {gap_counts.size} tipo(s) de gap encontrados "
                f"(paso normal = {mode_delta})")

            for gap_delta, count in gap_counts.items():
                print(f"   • gap {gap_delta}  →  {count} veces")
                # mostrar algunos ejemplos
                examples = deltas[deltas == gap_delta].head(show_examples)
                for ts, gap in examples.items():
                    prev = ts - gap
                    print(f"        {prev}  →  {ts}")

        print(f"{name}: {len(idx):,} filas  ({idx[0]}  →  {idx[-1]})\n")

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
        dataset: pd.DataFrame,
        model_main: object,
        model_meta: object,
        model_main_cols: list[str],
        model_meta_cols: list[str],
        direction: str = 'both',
        plot: bool = False,
        prd: str = '') -> float:

    """Evalúa una estrategia para una o ambas direcciones.

    Parameters
    ----------
    dataset : pd.DataFrame
        DataFrame con las columnas de cierre y etiquetas.
    model_main_cols : list[str]
        Lista de nombres de columnas para el modelo principal.
    model_meta_cols : list[str]
        Lista de nombres de columnas para el meta-modelo.
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
    # Preparación de datos
    ds_main = dataset[model_main_cols].to_numpy()
    ds_meta = dataset[model_meta_cols].to_numpy()
    close = dataset['close'].to_numpy()

    # Calcular probabilidades usando ambos modelos (sin binarizar)
    main = _predict_one(model_main, ds_main)
    meta = _predict_one(model_meta, ds_meta)

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
        dataset: pd.DataFrame,
        model_main: object,
        model_meta: object,
        model_main_cols: list[str],
        model_meta_cols: list[str],
        direction: str = 'buy',
        plot: bool = False,
        prd: str = '') -> float:
    """Mantiene compatibilidad con la API anterior."""
    return tester(
        dataset=dataset,
        model_main=model_main,
        model_meta=model_meta,
        model_main_cols=model_main_cols,
        model_meta_cols=model_meta_cols,
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

@lru_cache(maxsize=2)
def _ort_session(model_path:str):
    sess  = rt.InferenceSession(model_path,
                                providers=['CPUExecutionProvider'])
    iname = sess.get_inputs()[0].name
    return sess, iname

def _predict_onnx(model_path:str, X_3d:np.ndarray) -> np.ndarray:
    n_sim, n_rows, n_feat = X_3d.shape
    sess, iname = _ort_session(model_path)

    raw = sess.run(None, {iname: X_3d.reshape(-1, n_feat).astype(np.float32)})[0]

    # ─── des-ZipMap / distintos formatos de salida ─────────────────────
    if raw.dtype == object:                     # lista de dicts
        prob_pos = np.fromiter((row[b'1'] for row in raw), dtype=np.float32)

    elif raw.ndim == 2:                         # matriz (n,2)
        prob_pos = raw[:, 1].astype(np.float32)

    elif raw.ndim == 1:                         # vector (n,)  → ya es proba+
        prob_pos = raw.astype(np.float32)

    else:
        raise RuntimeError(f"Formato de salida ONNX no soportado: {raw.shape}")

    return prob_pos.reshape(n_sim, n_rows)

def _predict_one(model_any, X_2d: np.ndarray) -> np.ndarray:
    """
    Devuelve la probabilidad de la clase positiva para una sola matriz 2-D.
      · Si 'model_any' es CatBoost -> usa predict_proba.
      · Si es ruta .onnx, bytes, o ModelProto -> usa _predict_onnx.
    Resultado shape: (n_rows,)
    """
    if hasattr(model_any, "predict_proba"):
        return model_any.predict_proba(X_2d)[:, 1]
    else:
        # _predict_onnx espera tensor 3-D: (n_sim, n_rows, n_feat)
        return _predict_onnx(model_any, X_2d[None, :, :])[0]

@njit(cache=True, fastmath=True, parallel=True)
def _score_batch(close_all, l_all, m_all, block_size, direction):
    """
    Calcula el score para cada simulación *tal cual*, sin volver a alterar
    precios ni señales (ya vienen ruidosos).
    Ahora también devuelve las curvas de equity de cada simulación.
    """
    n_sim, n = l_all.shape
    scores = np.full(n_sim, -1.0)
    # Usar array 2-D en lugar de lista de arrays para evitar múltiples firmas
    equity_curves = np.zeros((n_sim, n+1), dtype=np.float64)
    
    for i in prange(n_sim):
        if direction == "both":
            rpt, _ = process_data(close_all[i], l_all[i], m_all[i])
        else:
            rpt, _ = process_data_one_direction(close_all[i],
                                                l_all[i],
                                                m_all[i],
                                                direction)
        if rpt.size < 2:
            continue
        ret = np.diff(rpt)
        if ret.size == 0:
            continue
        eq = _equity_from_returns(_bootstrap_returns(ret, block_size))
        scores[i] = evaluate_report(eq)
        # Copiar la curva de equity al array 2-D
        eq_len = min(len(eq), n+1)
        equity_curves[i, :eq_len] = eq[:eq_len]
    
    return scores, equity_curves

def monte_carlo_full(
    dataset: pd.DataFrame,
    model_main: object,
    model_meta: object,
    model_main_cols: list[str],
    model_meta_cols: list[str],
    hp: dict,
    direction: str = "both",
    n_sim: int = 100,
    block_size: int = 1,
    plot: bool = False,
    prd: str = "",
) -> dict:
    """Lanza Monte Carlo alterando solo el cierre y recalculando las features."""

    # extraer serie de cierre original
    close = np.ascontiguousarray(dataset["close"].to_numpy())

    try:
        base_df = dataset.copy()
        feats = get_features(base_df, dict(hp))
        if feats.empty:
            return {"scores": np.array([-1.0]), "p_positive": 0.0, "quantiles": np.array([-1.0, -1.0, -1.0])}

        X_main = feats[model_main_cols].to_numpy()
        X_meta = feats[model_meta_cols].to_numpy()
        close_feat = feats["close"].to_numpy()

        main = _predict_one(model_main, X_main)
        meta = _predict_one(model_meta, X_meta)
        if direction == "both":
            rpt_original, _ = process_data(close_feat, main, meta)
        else:
            rpt_original, _ = process_data_one_direction(close_feat, main, meta, direction)

        # Score de la original
        score_original = evaluate_report(rpt_original)

        # ───── 3-D batch build ──────────────────────────────────────────────
        X_main_batch, X_meta_batch, close_batch = [], [], []

        for _ in range(n_sim):
            noisy_close = _make_noisy_close(close)
            df_sim      = dataset.copy()
            df_sim["close"] = noisy_close

            feats_sim = get_features(df_sim, dict(hp))
            if feats_sim.empty:
                continue                          # descarta simulaciones inválidas

            X_main_batch.append(feats_sim[model_main_cols].to_numpy())
            X_meta_batch.append(feats_sim[model_meta_cols].to_numpy())
            close_batch.append(feats_sim["close"].to_numpy())

        if not X_main_batch:                     # ninguna simulación válida
            return {"scores": np.array([-1.0]), "p_positive": 0.0,
                    "quantiles": np.array([-1.0, -1.0, -1.0])}

        # → tensors 3-D
        X_main_arr  = np.stack(X_main_batch)              # (n_sim, n_rows, n_featM)
        X_meta_arr  = np.stack(X_meta_batch)              # (n_sim, n_rows, n_featm)
        close_arr   = np.stack(close_batch)               # (n_sim, n_rows)

        # ───── inferencia ONNX en bloque ────────────────────────────────────
        pred_main = _predict_onnx(model_main, X_main_arr)   # ← ruta .onnx
        pred_meta = _predict_onnx(model_meta, X_meta_arr)

        # ───── score vectorizado (Numba) ────────────────────────────────────
        scores, equity_curves = _score_batch(close_arr,
                                    pred_main,
                                    pred_meta,
                                    block_size,
                                    direction)

        # Incluir la original al principio del array de scores
        scores = np.concatenate(([score_original], scores))
        # Convertir el array 2-D de equity_curves a lista para mantener compatibilidad
        equity_curves_list = [rpt_original] + [equity_curves[i] for i in range(equity_curves.shape[0])]

        valid  = scores[scores > 0.0]

        if plot:
            if valid.size == 0:
                # No hay datos válidos, no graficar nada
                pass
            else:
                try:
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
                    # Solo graficar las curvas correspondientes a scores válidos
                    valid_indices = np.where(scores > 0.0)[0]
                    for idx in valid_indices:
                        ax1.plot(equity_curves_list[idx], color='gray', alpha=0.1)
                    ax1.plot(equity_curves_list[0], color='blue', linewidth=2, label='Original')

                    min_len = min(len(equity_curves_list[idx]) for idx in valid_indices) if len(valid_indices) > 0 else 0
                    if min_len > 0:
                        curves_array = np.array([equity_curves_list[idx][:min_len] for idx in valid_indices])
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

                    ax1.set_xlabel('Operaciones')
                    ax1.set_ylabel('Beneficio Acumulado')
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)

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
        dataset: pd.DataFrame,
        model_main: object,
        model_meta: object,
        model_main_cols: list[str],
        model_meta_cols: list[str],
        hp: dict,
        direction: str = "both",
        n_sim: int = 100,
        block_size: int = 1,
        agg: str = "q05",
        plot: bool = False,
        prd: str = "") -> float:

    """Calcula un score robusto en out-of-sample mediante Monte Carlo."""

    try:
        mc = monte_carlo_full(
            dataset=dataset,
            model_main=model_main,
            model_meta=model_meta,
            model_main_cols=model_main_cols,
            model_meta_cols=model_meta_cols,
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
    dataset: pd.DataFrame,
    model_main: object,
    model_meta: object,
    model_main_cols: list[str],
    model_meta_cols: list[str],
    hp: dict,
    direction: str = "both",
    n_splits: int = 3,
    agg: str = "min",
    plot: bool = False,
) -> float:
    """Calcula un score OOS mediante validación walk-forward."""

    try:
        n = len(dataset)
        if n_splits <= 1 or n <= 1:
            return robust_oos_score(
                dataset=dataset,
                model_main=model_main,
                model_meta=model_meta,
                model_main_cols=model_main_cols,
                model_meta_cols=model_meta_cols,
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
                dataset=dataset.iloc[start:end],
                model_main=model_main,
                model_meta=model_meta,
                model_main_cols=model_main_cols,
                model_meta_cols=model_meta_cols,
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
