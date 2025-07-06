from numba import njit, prange, float64, int64
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import onnxruntime as rt
from functools import lru_cache
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
def process_data_one_direction(close, main_labels, meta_labels, direction_int):
    last_deal  = 2            # 2 = flat, 1 = position open
    last_price = 0.0
    report = [0.0]
    chart  = [0.0]
    long_side = (direction_int == 0)  # 0=buy, 1=sell
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
        direction_map = {'buy': 0, 'sell': 1}
        direction_int = direction_map.get(direction, 0)
        rpt, _ = process_data_one_direction(close, main, meta, direction_int)

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
def _signed_r2(y):
    n = y.size
    t = np.arange(n, dtype=np.float64)
    t_mean = t.mean()
    y_mean = y.mean()
    cov = np.sum((t - t_mean) * (y - y_mean))
    var_t = np.sum((t - t_mean)**2)
    var_y = np.sum((y - y_mean)**2)
    if var_t == 0 or var_y == 0:
        return 0.0
    slope = cov / var_t
    r2 = (cov**2) / (var_t * var_y)
    return np.sign(slope) * r2          #  ∈ [-1,1]

@njit(cache=True, fastmath=True)
def _max_dd_curve(eq):
    peak = eq[0]
    mdd  = 0.0
    for x in eq:
        if x > peak:
            peak = x
        dd = peak - x
        if dd > mdd:
            mdd = dd
    return mdd

@njit(cache=True, fastmath=True)
def _bars_since_last_high(eq):
    last_high = eq[0]
    gap = 0
    max_gap = 0
    for x in eq:
        if x >= last_high:
            last_high = x
            max_gap = max(max_gap, gap)
            gap = 0
        else:
            gap += 1
    return gap          # barras desde el último máximo

@njit(cache=True, fastmath=True, parallel=True)
def uniform_filter1d(x: np.ndarray,
                                    size: int) -> np.ndarray:
    if size < 1:
        raise ValueError("`size` debe ser >= 1")

    n = x.size
    out = np.empty_like(x)

    left  = size // 2
    right = size - left          # incluye posición i

    # ---------- bucle paralelo ----------
    for i in prange(n):
        acc = 0.0
        for k in range(-left, right):
            idx = i + k

            # modo reflect
            if idx < 0:
                idx = -idx - 1
            elif idx >= n:
                idx = (2 * n - 1) - idx

            acc += x[idx]

        out[i] = acc / size

    return out

@njit(cache=True, fastmath=True)
def evaluate_report(eq: np.ndarray) -> float:
    """
    Puntúa una curva de equity de forma relativa (0–1).
    Favorece:
      • pendiente positiva y estable,
      • buena relación beneficio/riesgo,
      • muchos trades,
      • superar máximos pronto.
    """
    if eq.size < 15 or not np.isfinite(eq).all():
        return -1.0

    ret = np.diff(eq)
    n   = ret.size
    if n < 10:
        return -1.0
    mean_ret = ret.mean()
    std_ret  = ret.std() + 1e-12
    skew = np.mean(((ret - mean_ret) / std_ret) ** 3)
    kurt = np.mean(((ret - mean_ret) / std_ret) ** 4)
    skew_penalty = 1.0 / (1.0 + np.exp(-skew))           # favorece skew > 0
    kurt_penalty = 1.0 / (1.0 + 0.25 * (kurt - 3.0)**2)  # penaliza kurtosis ≠ 3
    shape_penalty = skew_penalty * kurt_penalty
    # ---------- 1) Tendencia --------------------------------------------
    slope = (eq[-1] - eq[0]) / n
    sigma = ret.std() + 1e-12
    eq_smooth = uniform_filter1d(eq, size=10)
    sr2 = _signed_r2(eq_smooth)
    if sr2 <= 0.0:
        return -1.0
    ratio = min(5.0, max(-5.0, slope / sigma))
    trend = 0.5 * (1.0 + sr2)
    trend *= 1.0 / (1.0 + np.exp(-ratio))

    # ---------- 2) Eficiencia -------------------------------------------
    gains  = ret[ret > 0.0].sum()
    losses = -ret[ret < 0.0].sum() + 1e-12
    pf   = gains / losses               # profit factor
    pf_n = 1.0 - np.exp(-pf / 3.0)

    dd   = _max_dd_curve(eq) + 1e-12
    rdd = min(10.0, max(0.0, (eq[-1] - eq[0]) / dd))
    rdd_n = 1.0 - np.exp(-rdd / 6.0)

    effic = pf_n * rdd_n                # ∈ (0,1)
    stability = 1.0 / (1.0 + ret.std() / (np.abs(ret.mean()) + 1e-12))

    # ---------- 3) Agilidad ---------------------------------------------
    gap  = _bars_since_last_high(eq)
    g    = 250.0                        # barras para penalizar al 50 %
    agil = 1.0 / (1.0 + gap / g)        # ↓ al crecer gap

    # ---------- 4) Madurez real por número de operaciones ----------------
    ops = np.count_nonzero(ret)
    N0  = 150.0                        # operaciones para madurez media
    s   = 40.0                         # pendiente
    maturity = 1.0 / (1.0 + np.exp(-(ops - N0) / s))

    # ---------- Score ----------------------------------------------------
    wT = 0.6                            # peso tendencia
    wE = 0.4
    core = wT * trend + wE * effic
    core *= stability
    score = core * agil * maturity * shape_penalty
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
