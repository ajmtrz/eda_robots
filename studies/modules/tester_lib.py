from numba import jit, njit
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


@jit(fastmath=True, cache=True)
def process_data(close, labels, metalabels, forward, backward):
    last_deal  = 2          # 2 = flat, 0 = short, 1 = long
    last_price = 0.0
    report, chart = [0.0], [0.0]
    line_f = line_b = 0

    for i in range(len(close)):
        line_f = len(report) if i <= forward  else line_f
        line_b = len(report) if i <= backward else line_b

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

    return np.array(report), np.array(chart), line_f, line_b

@jit(fastmath=True, cache=True)
def process_data_one_direction(close, labels, metalabels,
                             forward_idx, backward_idx, direction):
    last_deal = 2           # 2 = flat, 1 = in‑market (única dirección)
    last_price = 0.0
    report, chart = [0.0], [0.0]
    line_f = line_b = 0
    long_side = (direction == 'buy')

    for i in range(len(close)):
        # Marcar líneas de corte para gráficos
        line_f = len(report) if i <= forward_idx else line_f
        line_b = len(report) if i <= backward_idx else line_b

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

    return np.array(report), np.array(chart), line_f, line_b

# ───────────────────────────────────────────────────────────────────
# 2)  Wrappers del tester
# ───────────────────────────────────────────────────────────────────
def tester(dataset, forward, backward, plot=False):
    # Ya que el dataset está filtrado entre backward y forward,
    # obtenemos los índices correspondientes dentro de este dataset filtrado
    forw = dataset.index.get_indexer([forward], method='nearest')[0]
    back = dataset.index.get_indexer([backward], method='nearest')[0]

    close = dataset['close'].to_numpy()
    lab   = dataset['labels'].to_numpy()
    meta  = dataset['meta_labels'].to_numpy()

    # Pasamos los índices relativos al dataset filtrado
    rpt, ch, lf, lb = process_data(close, lab, meta, forw, back)

    # regresión lineal sobre el equity
    y = rpt.reshape(-1, 1)
    X = np.ascontiguousarray(np.arange(len(rpt))).reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    sign = 1 if lr.coef_[0][0] >= 0 else -1

    if plot:
        plt.plot(rpt)
        plt.plot(ch)
        plt.axvline(lf, color='purple', ls=':')
        plt.axvline(lb, color='red',    ls=':')
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
    
    # Necesitamos mínimo 5 operaciones para considerar la estrategia válida
    if num_trades < 5:
        return -1.0

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
    return_dd_ratio = total_return / max_dd if max_dd > 0 else 0.0

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

def tester_one_direction(dataset, forw, back,
                        direction='buy', plot=False):
    # Convertir fechas a índices numéricos
    forward_idx = dataset.index.get_indexer([forw])[0]
    backward_idx = dataset.index.get_indexer([back])[0]

    # Extraer datos necesarios
    close = np.ascontiguousarray(dataset['close'].values)
    lab = np.ascontiguousarray(dataset['labels'].values)
    meta = np.ascontiguousarray(dataset['meta_labels'].values)

    # Pasamos los índices numéricos en lugar de fechas
    rpt, ch, lf, lb = process_data_one_direction(
        close, lab, meta, forward_idx, backward_idx, direction)
    
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
        plt.axvline(lf, color='purple', ls=':', label='Forward Test')
        plt.axvline(lb, color='red', ls=':', label='Backward Test')
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
def tester_slow(dataset, forward, markup, plot=False):
    last_deal = 2
    last_price = 0.0
    report, chart = [0.0], [0.0]
    line_f = line_b = 0

    idx_fwd = dataset.index.get_indexer([forward], method='nearest')[0]

    close = dataset['close'].to_numpy()
    labels = dataset['labels'].to_numpy()
    metalabels = dataset['meta_labels'].to_numpy()

    for i in range(dataset.shape[0]):
        line_f = len(report) if i <= idx_fwd else line_f

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
        plt.axvline(x=line_f, color='purple', ls=':', lw=1, label='OOS')
        plt.axvline(x=line_b, color='red', ls=':', lw=1, label='OOS2')
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
               forward: datetime,
               backward: datetime,
               plt=False):

    ext_dataset = dataset.copy()
    mask = (ext_dataset.index > backward) & (ext_dataset.index < forward)
    ext_dataset = ext_dataset[mask]
    X = ext_dataset.iloc[:, 1:]

    ext_dataset['labels']      = result[0].predict_proba(X)[:, 1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X)[:, 1]
    ext_dataset[['labels', 'meta_labels']] = ext_dataset[['labels', 'meta_labels']].gt(0.5).astype(float)

    return tester(ext_dataset, forward, backward, plot=plt)


def test_model_one_direction(dataset: pd.DataFrame,
                           result: list,
                           forward: datetime,
                           backward: datetime,
                           direction: str = 'buy',
                           plt=False):
    # Copiar dataset para no modificar el original
    ext_dataset = dataset.copy()
    
    # Filtrado directo por fechas usando el índice
    mask = (ext_dataset.index > backward) & (ext_dataset.index < forward)
    ext_dataset = ext_dataset[mask]
    #print(f"Testing period: {ext_dataset.index[0]} to {ext_dataset.index[-1]}")
    
    # Extraer características regulares y meta-features
    X_main = ext_dataset.loc[:, ext_dataset.columns.str.contains('_feature') & ~ext_dataset.columns.str.contains('_meta_feature')]
    X_meta = ext_dataset.loc[:, ext_dataset.columns.str.contains('_meta_feature')]

    # Calcular probabilidades usando ambos modelos
    ext_dataset['labels'] = result[0].predict_proba(X_main)[:, 1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X_meta)[:, 1]
    
    # Convertir probabilidades a señales binarias
    ext_dataset[['labels', 'meta_labels']] = ext_dataset[['labels', 'meta_labels']].gt(0.5).astype(float)

    return tester_one_direction(ext_dataset, forward, backward, direction, plot=plt)