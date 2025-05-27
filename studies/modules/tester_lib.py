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
    skl2onnx_parser_castboost_classifier,
    skl2onnx_convert_catboost,
)
rt.set_default_logger_severity(4)

@jit(fastmath=True)
def process_data(close, labels, metalabels):
    last_deal  = 2
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

@jit(fastmath=True)
def process_data_one_direction(close, main_labels, meta_labels, direction):
    """
    Procesa los datos para una dirección específica (compra o venta).
    
    Args:
        close: Array de precios de cierre
        main_labels: Array de probabilidades del modelo principal (sin binarizar)
        meta_labels: Array de probabilidades del modelo meta (sin binarizar)
        direction: 'buy' o 'sell'
    """
    last_deal = 2           # 2 = flat, 1 = in‑market (única dirección)
    last_price = 0.0
    report, chart = [0.0], [0.0]
    long_side = (direction == 'buy')
    min_prob = 0.5         # Umbral de probabilidad

    for i in range(len(close)):
        pred_main, pr, pred_meta = main_labels[i], close[i], meta_labels[i]

        # Abrir posición si:
        # 1. No hay posición abierta (last_deal == 2)
        # 2. El metamodelo da señal positiva (prob >= 0.5)
        # 3. El modelo principal da señal positiva (prob >= 0.5)
        if last_deal == 2 and pred_meta > min_prob and pred_main > min_prob:
            last_deal = 1
            last_price = pr
            continue

        # Cerrar posición si el modelo principal da señal negativa (prob < 0.5)
        if last_deal == 1 and pred_main < min_prob:
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
    main   = dataset['main_labels'].to_numpy()
    meta  = dataset['meta_labels'].to_numpy()

    # Pasamos los índices relativos al dataset filtrado
    rpt, ch= process_data(close, main, meta)

    # regresión lineal sobre el equity
    y = rpt.reshape(-1, 1)
    X = np.ascontiguousarray(np.arange(len(rpt))).reshape(-1, 1)
    lr = _signed_r2(rpt)
    sign = 1 if lr.coef_[0][0] >= 0 else -1

    if plot:
        plt.plot(rpt)
        plt.plot(ch)
        plt.plot(lr.predict(X))
        plt.title(f"R² {lr.score(X, y) * sign:.2f}")
        plt.show()

    return lr.score(X, y) * sign

# ───────────────────────────────────────────────
# NUEVA FUNCIÓN evaluate_report
# ───────────────────────────────────────────────
@njit(fastmath=True, nogil=True)
def evaluate_report(report: np.ndarray) -> float:
    eps = 1e-6

    # Verificación básica - necesitamos al menos 2 puntos para un reporte válido
    if len(report) < 2:
        return -1.0

    # Calcular los retornos individuales
    returns = np.diff(report)
    num_trades = len(returns)
    if num_trades < 3:
        return -1.0
    
    # Calcular R² y sea positivo
    r2_raw = _signed_r2(report)
    if r2_raw > 0:
        r2_raw *= 1.0
    else:
        return -1.0
    
    # ────────────────────────
    # MÉTRICAS BASE
    gains = returns[returns > 0]
    losses = -returns[returns < 0]
    profit_factor = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else eps

    equity_curve = report
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        max_dd = max(max_dd, peak - x)
    total_return = equity_curve[-1] - equity_curve[0]
    return_dd_ratio = total_return / max_dd if max_dd > 0 else eps

    # ────────────────────────
    # PUNTAJE COMPUESTO BASE
    base_score = (
        (profit_factor * 0.4) +
        (return_dd_ratio * 0.6)
    )

    # ────────────────────────
    # Factor de trades
    trade_multiplier = np.log10(num_trades) / np.log10(100)
    
    # ────────────────────────
    # Score final
    final_score = (0.5 * base_score + 0.5 * r2_raw) * trade_multiplier
    
    return final_score

def tester_one_direction(dataset, direction='buy', plot=False, prd=''):

    # Extraer datos necesarios
    close = np.ascontiguousarray(dataset['close'].values)
    main = np.ascontiguousarray(dataset['main_labels'].values)
    meta = np.ascontiguousarray(dataset['meta_labels'].values)

    # Pasamos los índices numéricos en lugar de fechas
    rpt, ch= process_data_one_direction(
        close, main, meta, direction)

    # Si no hay suficientes operaciones, devolver valor negativo
    if len(rpt) < 2:
        return -1.0
    
    # Calcular score
    score = evaluate_report(rpt)

    # Visualizar resultados si se solicita
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(rpt, label='Equity Curve')
        plt.xlabel("Operations")
        plt.ylabel("Cumulative Profit")
        plt.title(f"Period: {prd} | Score: {score:.2f}")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    # Evaluar el reporte para obtener una puntuación completa
    return score

# ─────────────────────────────────────────────
# tester_slow  (mantenerlo o borrarlo)
# ─────────────────────────────────────────────
def tester_slow(dataset, markup, plot=False):
    last_deal = 2
    last_price = 0.0
    report, chart = [0.0], [0.0]

    close = dataset['close'].to_numpy()
    main = dataset['main_labels'].to_numpy()
    metalabels = dataset['meta_labels'].to_numpy()

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
               plt=False):

    ext_dataset = dataset.copy()
    mask = (ext_dataset.index > backward) & (ext_dataset.index < forward)
    ext_dataset = ext_dataset[mask].reset_index(drop=True)
    X = ext_dataset.iloc[:, 1:]

    ext_dataset['main_labels'] = result[0].predict_proba(X)[:, 1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X)[:, 1]

    return tester(ext_dataset, plot=plt)


def test_model_one_direction(dataset: pd.DataFrame,
                           model_main: object,
                           model_meta: object,
                           direction: str = 'buy',
                           plt=False,
                           prd= ''):
    # Copiar dataset para no modificar el original
    ext_dataset = dataset.copy()

    # Extraer características regulares y meta-features
    X_main = ext_dataset.loc[:, ext_dataset.columns.str.contains('_feature') & ~ext_dataset.columns.str.contains('_meta_feature')].to_numpy('float32')
    X_meta = ext_dataset.loc[:, ext_dataset.columns.str.contains('_meta_feature')].to_numpy('float32')

    # Calcular probabilidades usando ambos modelos (sin binarizar)
    ext_dataset['main_labels'] = model_main.predict_proba(X_main)[:, 1]
    ext_dataset['meta_labels'] = model_meta.predict_proba(X_meta)[:, 1]

    return tester_one_direction(ext_dataset, direction, plot=plt, prd=prd)



# ───────────────────────────────────────────────────────────────────
# Monte-Carlo robustness utilities
# ───────────────────────────────────────────────────────────────────

# ---------- helpers ------------------------------------------------
@njit(fastmath=True)
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


@njit(fastmath=True)
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


@njit(fastmath=True)
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

@njit(fastmath=True)
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

@njit(parallel=True, fastmath=True)
def _simulate_batch(close, l_all, m_all, block_size, direction):
    n_sim, n = l_all.shape
    scores = np.full(n_sim, -1.0)
    for i in prange(n_sim):
        c_n, l_n, m_n = _make_noisy_signals(close, l_all[i], m_all[i])
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
    parser=skl2onnx_parser_castboost_classifier,
    options={"nocl": [True, False], "zipmap": [True, False]}
)
# 1) ───────── cache global de sesiones ───────────────────────────
_ONNX_CACHE: dict[int, rt.InferenceSession] = {}

def _predict_batch(model, X_base, noise_levels):
    sid = id(model)
    sess = _ONNX_CACHE.get(sid)
    if sess is None:
        onnx_model = convert_sklearn(
            model,
            initial_types=[('x', FloatTensorType([None, X_base.shape[1]]))],
            target_opset={"": 18, "ai.onnx.ml": 2},
            options={id(model): {'zipmap': False}}
        )
        providers = ['CPUExecutionProvider']
        sess = rt.InferenceSession(
            onnx_model.SerializeToString(),
            providers=providers
        )
        _ONNX_CACHE[sid] = sess

    iname = sess.get_inputs()[0].name
    onnx_run = sess.run

    # 1. Preparar los datos base
    n_sim, n_samples, n_feat = len(noise_levels), *X_base.shape
    X_big = np.repeat(X_base[None, :, :], n_sim, axis=0)
    
    # 2. Calcular sensibilidad de cada feature
    feature_importance = model.get_feature_importance()
    sensitivity = feature_importance / feature_importance.max()
    
    # 3. Generar y aplicar el ruido ajustado
    std = X_base.std(axis=0, keepdims=True)
    eps = np.random.normal(0.0, std, size=X_big.shape)
    adjusted_noise = noise_levels[:, None, None] * sensitivity[None, None, :]
    X_big += eps * adjusted_noise

    # 4. Realizar la predicción
    proba_flat = onnx_run(None, {iname: X_big.reshape(-1, n_feat).astype(np.float32)})[0]
    proba = proba_flat.reshape(n_sim, n_samples)
    return proba

def monte_carlo_full(
    model_main: object,
    model_meta: object,
    close: np.ndarray,
    X_main: np.ndarray,
    X_meta: np.ndarray,
    direction: str,
    n_sim: int = 100,
    block_size: int = 20
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
        # ── predicción batch + cast a float64 (necesario para _make_noisy_signals) ──
        noise_levels = np.random.uniform(0.005, 0.02, n_sim)
        l_all = _predict_batch(model_main, X_main, noise_levels)
        m_all = _predict_batch(model_meta, X_meta, noise_levels)

        scores = _simulate_batch(close, l_all, m_all, block_size, direction)

        valid = scores[scores > 0.0]
        return {
            "scores": scores,
            "p_positive": np.mean(scores > 0),
            "quantiles": np.quantile(valid, [0.05, 0.5, 0.95]) if valid.size else [-1,-1,-1],
        }
    except Exception as e:
        import traceback
        print(f"\nError en monte_carlo_full:")
        print(f"Error: {str(e)}")
        print("Traceback:")
        print(traceback.format_exc())
        return {"scores": np.array([-1.0]), "p_positive": 0.0, "quantiles": np.array([-1.0, -1.0, -1.0])}

def robust_oos_score_one_direction(dataset: pd.DataFrame,
                                   model_main: object,
                                   model_meta: object,
                                   direction: str,
                                   n_sim: int = 100,
                                   block_size: int = 20,
                                   agg: str = "q05") -> float:
    """
    Devuelve un score robusto (float) listo para Optuna.
    Solo prepara los datos y pasa los modelos y features a monte_carlo_full,
    que se encargará de aplicar ruido y hacer predicciones en cada simulación.
    """
    # 1) Copiar dataset --------------------------------------------
    ext_ds = dataset.copy()
    # 2) Validar dataset --------------------------------------------
    if ext_ds.empty:
        return -1.0

    # 3) preparar datos --------------------------------------------
    # Validar que las características de test coincidan con las de train
    test_main_cols = ext_ds.columns[ext_ds.columns.str.contains('_feature') & ~ext_ds.columns.str.contains('_meta_feature')] 
    test_meta_cols = ext_ds.columns[ext_ds.columns.str.contains('_meta_feature')]
    
    # Convertir a numpy arrays directamente para evitar problemas con pandas
    X_main = ext_ds[test_main_cols].to_numpy()
    X_meta = ext_ds[test_meta_cols].to_numpy()
    close_arr = ext_ds['close'].to_numpy(copy=True)
    
    if X_main.size == 0 or X_meta.size == 0:
        return -1.0

    # 3) Monte Carlo robusto -------------------------------------
    try:
        mc = monte_carlo_full(
            close=close_arr,
            X_main=X_main,
            X_meta=X_meta,
            model_main=model_main,
            model_meta=model_meta,
            direction=direction,
            n_sim=n_sim,
            block_size=block_size,
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
        print(f"\nError en robust_oos_score_one_direction: {e}")
        return -1.0