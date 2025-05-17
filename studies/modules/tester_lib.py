from numba import jit, njit, prange
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import Callable, Literal

@jit(fastmath=True, cache=True)
def process_data(close, labels, metalabels):
    last_deal  = 2          # 2 = flat, 0 = short, 1 = long
    last_price = 0.0
    report, chart = [0.0], [0.0]

    for i in range(len(close)):
        pred, pr, pred_meta = labels[i], close[i], metalabels[i]

        # ── abrir posición
        if last_deal == 2 and pred_meta == 1:
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

    return np.array(report), np.array(chart)

@jit(fastmath=True, cache=True)
def process_data_one_direction(close, labels, metalabels, direction):
    last_deal = 2           # 2 = flat, 1 = in‑market (única dirección)
    last_price = 0.0
    report, chart = [0.0], [0.0]
    long_side = (direction == 'buy')

    for i in range(len(close)):
        pred, pr, pred_meta = labels[i], close[i], metalabels[i]

        # Abrir posición si:
        # 1. No hay posición abierta
        # 2. El metamodelo da señal positiva
        # 3. El modelo principal da señal positiva (>0.5)
        if last_deal == 2 and pred_meta == 1 and pred > 0.5:
            last_deal = 1
            last_price = pr
            continue

        # Cerrar posición si el modelo principal da señal negativa (<0.5)
        if last_deal == 1 and pred < 0.5:
            last_deal = 2
            # Calcular beneficio según dirección
            profit = (pr - last_price) if long_side else (last_price - pr)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + profit)
            continue

    return np.array(report), np.array(chart)

# ───────────────────────────────────────────────────────────────────
# 2)  Wrappers del tester
# ───────────────────────────────────────────────────────────────────
def tester(dataset, plot=False):

    close = dataset['close'].to_numpy()
    lab   = dataset['labels'].to_numpy()
    meta  = dataset['meta_labels'].to_numpy()

    # Pasamos los índices relativos al dataset filtrado
    rpt, ch= process_data(close, lab, meta)

    # regresión lineal sobre el equity
    y = rpt.reshape(-1, 1)
    X = np.ascontiguousarray(np.arange(len(rpt))).reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    sign = 1 if lr.coef_[0][0] >= 0 else -1

    if plot:
        plt.plot(rpt)
        plt.plot(ch)
        plt.plot(lr.predict(X))
        plt.title(f"R² {lr.score(X, y) * sign:.2f}")
        plt.show()

    return lr.score(X, y) * sign

@njit(fastmath=True, cache=True)
def evaluate_report(report: np.ndarray, r2_raw: float) -> float:
    # Verificación básica - necesitamos al menos 2 puntos para un reporte válido
    if len(report) < 2:
        return -1.0

    # Calcular los retornos individuales
    returns = np.diff(report)
    num_trades = len(returns)

    # ────────────────────────
    # MÉTRICAS BASE
    gains = returns[returns > 0]
    losses = -returns[returns < 0]
    
    # Calcular profit factor (relación entre ganancias y pérdidas)
    if np.sum(losses) > 0:
        profit_factor = np.sum(gains) / np.sum(losses)
    else:
        profit_factor = 0.0  # Sin pérdidas, consideramos profit factor 0

    # Calcular maximum drawdown
    equity_curve = report
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        if x > peak:
            peak = x
        current_dd = peak - x
        if current_dd > max_dd:
            max_dd = current_dd
    
    # Calcular return-to-drawdown ratio
    total_return = equity_curve[-1] - equity_curve[0]
    return_dd_ratio = total_return / max(max_dd, 1e-6)

    # ────────────────────────
    # PUNTAJE COMPUESTO BASE
    base_score = (
        (profit_factor * 0.4) +  # 40% del peso al profit factor
        (return_dd_ratio * 0.6)  # 60% del peso al return/DD ratio
    )

    # Penalizaciones por métricas débiles
    penalization = 1.0
    if profit_factor < 2.0: penalization *= 0.8  # Penalizamos PF < 2
    if return_dd_ratio < 2.0: penalization *= 0.8  # Penalizamos RDD < 2

    # ────────────────────────
    # AJUSTE POR NÚMERO DE TRADES
    min_trades = 200  # Mínimo de trades deseado
    trade_weight = min(1.0, num_trades / min_trades)  # Penalización lineal
    if num_trades > 50:
        trade_weight = min(2.0, 1.0 + (num_trades - 50) / 100)  # Bonus gradual
    
    # Aplicar penalizaciones y ajustes
    base_score *= penalization * trade_weight

    # ────────────────────────
    # Verificamos que el R² sea positivo
    if r2_raw <= 0:
        return -1.0  # Rechazamos estrategias con tendencia negativa

    # ────────────────────────
    # Score final (50% métricas de trading, 50% calidad de regresión)
    final_score = 0.5 * base_score + 0.5 * r2_raw
    
    return final_score

def tester_one_direction(dataset, direction='buy', plot=False):

    # Extraer datos necesarios
    close = np.ascontiguousarray(dataset['close'].values)
    lab = np.ascontiguousarray(dataset['labels'].values)
    meta = np.ascontiguousarray(dataset['meta_labels'].values)

    # Pasamos los índices numéricos en lugar de fechas
    rpt, ch= process_data_one_direction(
        close, lab, meta, direction)
    
    # Si no hay suficientes operaciones, devolver valor negativo
    if len(rpt) < 2:
        return -1.0

    # Calcular regresión lineal para el R²
    y = rpt.reshape(-1, 1)
    X = np.ascontiguousarray(np.arange(len(rpt))).reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    sign = 1 if lr.coef_[0][0] >= 0 else -1
    r2_raw = lr.score(X, y) * sign

    # Visualizar resultados si se solicita
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(rpt, label='Equity Curve')
        plt.plot(lr.predict(X), 'g--', alpha=0.7, label='Regression Line')
        plt.title(f"Strategy Performance: R² {r2_raw:.2f}")
        plt.xlabel("Operations")
        plt.ylabel("Cumulative Profit")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # Evaluar el reporte para obtener una puntuación completa
    return evaluate_report(rpt, r2_raw)

# ─────────────────────────────────────────────
# tester_slow  (mantenerlo o borrarlo)
# ─────────────────────────────────────────────
def tester_slow(dataset, markup, plot=False):
    last_deal = 2
    last_price = 0.0
    report, chart = [0.0], [0.0]

    close = dataset['close'].to_numpy()
    labels = dataset['labels'].to_numpy()
    metalabels = dataset['meta_labels'].to_numpy()

    for i in range(dataset.shape[0]):
        pred, pr, pred_meta = labels[i], close[i], metalabels[i]

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
    lr = LinearRegression()
    lr.fit(X, y)

    l = 1 if lr.coef_ >= 0 else -1

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
               plt=False):

    ext_dataset = dataset.copy()
    mask = (ext_dataset.index > backward) & (ext_dataset.index < forward)
    ext_dataset = ext_dataset[mask].reset_index(drop=True)
    X = ext_dataset.iloc[:, 1:]

    ext_dataset['labels']      = result[0].predict_proba(X)[:, 1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X)[:, 1]
    ext_dataset[['labels', 'meta_labels']] = ext_dataset[['labels', 'meta_labels']].gt(0.5).astype(float)

    return tester(ext_dataset, plot=plt)


def test_model_one_direction(dataset: pd.DataFrame,
                           result: list,
                           backward: datetime,
                           forward: datetime,
                           direction: str = 'buy',
                           plt=False):
    # Copiar dataset para no modificar el original
    ext_dataset = dataset.copy()
    mask = (ext_dataset.index > backward) & (ext_dataset.index < forward)
    ext_dataset = ext_dataset[mask]#.reset_index(drop=True)

    # Extraer características regulares y meta-features
    X_main = ext_dataset.loc[:, ext_dataset.columns.str.contains('_feature') & ~ext_dataset.columns.str.contains('_meta_feature')]
    X_meta = ext_dataset.loc[:, ext_dataset.columns.str.contains('_meta_feature')]

    # Calcular probabilidades usando ambos modelos
    ext_dataset['labels'] = result[0].predict_proba(X_main)[:, 1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X_meta)[:, 1]
    
    # Convertir probabilidades a señales binarias
    ext_dataset[['labels', 'meta_labels']] = ext_dataset[['labels', 'meta_labels']].gt(0.5).astype(float)

    return tester_one_direction(ext_dataset, direction, plot=plt)



# ───────────────────────────────────────────────────────────────────
# Monte-Carlo robustness utilities
# ───────────────────────────────────────────────────────────────────

# ---------- helpers ------------------------------------------------
@njit(fastmath=True, cache=True)
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
        
    # ---- block bootstrap  ----------------------------------------
    # Calcular número exacto de bloques necesarios
    n_blocks = (n + block_size - 1) // block_size  # división entera hacia arriba
    
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


@njit(fastmath=True, cache=True)
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


@njit(fastmath=True, cache=True)
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

@njit(fastmath=True, cache=True)
def _make_noisy_signals(close: np.ndarray,
                       labels: np.ndarray,
                       meta: np.ndarray,
                       price_noise_range: tuple[float, float] = (0.001, 0.005),
                       prob_noise_range: tuple[float, float] = (0.01, 0.03),
                       correlation: float = 0.3) -> tuple:
    """Añade ruido a precios, labels y meta-labels."""
    n = close.size
    # ------ precios ------------------------------------------------
    price_noise = np.random.uniform(price_noise_range[0], price_noise_range[1])
    close_noisy = np.empty_like(close)
    if price_noise > 0:
        for i in range(n):
            close_noisy[i] = close[i] * (1 + np.random.laplace(0.0, price_noise))
    else:
        close_noisy[:] = close

    # ------ labels / meta_labels -----------------------------------
    prob_noise = price_noise * correlation + np.random.uniform(prob_noise_range[0], prob_noise_range[1]) * (1 - correlation)
    labels_noisy = np.empty_like(labels)
    meta_noisy = np.empty_like(meta)
    
    if prob_noise > 0:
        for i in range(n):
            labels_noisy[i] = min(1.0, max(0.0, labels[i] + np.random.normal(0.0, prob_noise)))
            meta_noisy[i] = min(1.0, max(0.0, meta[i] + np.random.normal(0.0, prob_noise)))
    else:
        labels_noisy[:] = labels
        meta_noisy[:] = meta

    # binarizamos tras añadir ruido
    for i in range(n):
        labels_noisy[i] = 1.0 if labels_noisy[i] > 0.5 else 0.0
        meta_noisy[i] = 1.0 if meta_noisy[i] > 0.5 else 0.0

    return close_noisy, labels_noisy, meta_noisy

@njit(parallel=True, fastmath=True, cache=True)
def _simulate_batch(close, l_all, m_all, block_size, direction):
    n_sim, n = l_all.shape
    scores = np.full(n_sim, -1.0)
    for i in prange(n_sim):                      # ← paralelo
        c_n, l_n, m_n = _make_noisy_signals(close, l_all[i], m_all[i])
        rpt, _ = process_data_one_direction(c_n, l_n, m_n, direction)
        if rpt.size < 2:
            continue
        ret = np.diff(rpt)
        if ret.size == 0:
            continue
        eq  = _equity_from_returns(_bootstrap_returns(ret, block_size))
        score = evaluate_report(eq, _signed_r2(eq))
        scores[i] = score
    return scores

def _predict_batch(model, X_base, noise_levels):
    """
    Devuelve (n_sim, n_samples) con las probabilidades para cada ruido.
    """
    n_sim, n_samples, n_feat = len(noise_levels), *X_base.shape
    # 1) replicar matriz base
    X_big = np.repeat(X_base[None, :, :], n_sim, axis=0)
    # 2) añadir un ruido distinto por simulación
    std = X_base.std(axis=0, keepdims=True)     # (1, n_feat)
    eps = np.random.normal(0.0, std, size=X_big.shape)
    X_big += eps * noise_levels[:, None, None]  # broadcast n_sim
    # 3) llamar al modelo **una vez**
    probs = model.predict_proba(X_big.reshape(-1, n_feat))[:, 1]
    return probs.reshape(n_sim, n_samples)

def monte_carlo_full(
    close: np.ndarray,
    *,
    n_sim: int = 100,
    block_size: int | None = 20,
    direction: str = 'buy',
    # --- modelos y features ---
    model_main=None,
    model_meta=None,
    X_main=None,
    X_meta=None,
) -> dict:
    """
    Lanza Monte-Carlo combinando ruido en inputs y bootstrapping de retornos.
    Si se pasan modelos y features, añade ruido a las features y predice en cada simulación.
    """
    # Validación de inputs
    if X_main is not None and X_meta is not None:
        if X_main.shape[0] != close.shape[0] or X_meta.shape[0] != close.shape[0]:
            raise ValueError("X_main y X_meta deben tener el mismo número de filas que close")
        if model_main is None or model_meta is None:
            raise ValueError("Si se pasan features, también deben pasarse los modelos")

    # ---------- curva original ------------------------------------
    try:
        # Predecir con los modelos originales
        labels = model_main.predict_proba(X_main)[:, 1]
        meta = model_meta.predict_proba(X_meta)[:, 1]
        labels = (labels > 0.5).astype(np.float64)
        meta = (meta > 0.5).astype(np.float64)

        report_orig, _ = process_data_one_direction(close, labels, meta, direction=direction)
        if len(report_orig) < 2:
            return {"scores": np.array([-1.0]), "p_positive": 0.0, "quantiles": np.array([-1.0, -1.0, -1.0])}

        scores = np.empty(n_sim, dtype=np.float64)

        noise_levels = np.random.uniform(0.005, 0.02, n_sim)
        l_all = (_predict_batch(model_main, X_main, noise_levels) > 0.5).astype(np.float64)
        m_all = (_predict_batch(model_meta, X_meta, noise_levels) > 0.5).astype(np.float64)

        scores = _simulate_batch(close, l_all, m_all, block_size, direction)

        valid = scores[scores > 0.0]
        return {
            "scores": scores,
            "p_positive": np.mean(scores > 0),
            "quantiles": np.quantile(valid, [0.05, 0.5, 0.95]) if valid.size else [-1,-1,-1],
        }
    except Exception as e:
        return {"scores": np.array([-1.0]), "p_positive": 0.0, "quantiles": np.array([-1.0, -1.0, -1.0])}

def robust_oos_score_one_direction(dataset: pd.DataFrame,
                                   models: list,
                                   backward: datetime | None = None,
                                   forward: datetime | None = None,
                                   *,
                                   direction: str = 'buy',
                                   n_sim: int = 100,
                                   block_size: int | None = 20,
                                   agg: str = "q05") -> float:
    """
    Devuelve un score robusto (float) listo para Optuna.
    Solo prepara los datos y pasa los modelos y features a monte_carlo_full,
    que se encargará de aplicar ruido y hacer predicciones en cada simulación.
    """
    # 1) filtra ventana OOS ---------------------------------------
    if backward is None or forward is None:
        ext_ds = dataset.copy()
    else:
        ext_ds = dataset.loc[(dataset.index >= backward) & (dataset.index <= forward)].copy()

    if ext_ds.empty:
        return -1.0

    # 2) preparar datos --------------------------------------------
    # Pre-calcular índices de características
    feature_cols = ext_ds.columns.str.contains('_feature')
    meta_feature_cols = ext_ds.columns.str.contains('_meta_feature')
    main_feature_cols = feature_cols & ~meta_feature_cols

    # Pre-calcular arrays base
    close_arr = ext_ds['close'].to_numpy(copy=True)
    X_main = ext_ds.loc[:, main_feature_cols].to_numpy(copy=True)
    X_meta = ext_ds.loc[:, meta_feature_cols].to_numpy(copy=True)

    if X_main.size == 0 or X_meta.size == 0:
        return -1.0

    # 3) Monte Carlo robusto -------------------------------------
    try:
        mc = monte_carlo_full(
            close_arr,
            n_sim=n_sim,
            block_size=block_size,
            direction=direction,
            model_main=models[0],
            model_meta=models[1],
            X_main=X_main,
            X_meta=X_meta
        )
        
        if mc["scores"].size == 0:
            return -1.0
            
        if agg == "q05":
            return float(mc["quantiles"][0])
        elif agg == "median":
            return float(mc["quantiles"][1])
        elif agg == "p_pos":
            return float(mc["p_positive"])
        else:
            raise ValueError("agg must be 'q05', 'median' or 'p_pos'")
            
    except Exception as e:
        return -1.0