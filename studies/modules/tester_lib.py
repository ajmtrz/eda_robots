from numba import jit
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from typing import Any, Tuple, List


@jit(nopython=True)
def process_data(close, labels, metalabels, markup, forward, backward):
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

    return np.array(report), np.array(chart), line_f, line_b

@jit(nopython=True)
def process_data_one_direction(close, labels, metalabels,
                               markup, forward, backward, direction):
    last_deal = 2           # 2 = flat, 1 = in‑market (única dirección)
    last_price = 0.0
    report, chart = [0.0], [0.0]
    line_f = line_b = 0
    long_side = (direction == 'buy')

    for i in range(len(close)):
        line_f = len(report) if i <= forward  else line_f
        line_b = len(report) if i <= backward else line_b

        pred, pr, pred_meta = labels[i], close[i], metalabels[i]

        # ── abrir
        if last_deal == 2 and pred_meta == 1 and pred > 0.5:
            last_deal = 1
            last_price = pr
            continue

        # ── cerrar al flip de la señal
        if last_deal == 1 and pred < 0.5:
            last_deal = 2
            profit = (-markup + (pr - last_price)) if long_side \
                     else (-markup + (last_price - pr))
            report.append(report[-1] + profit)
            chart.append(chart[-1] + profit)
            continue

    return np.array(report), np.array(chart), line_f, line_b

# ───────────────────────────────────────────────────────────────────
# 2)  Wrappers del tester
# ───────────────────────────────────────────────────────────────────
def tester(dataset, forward, backward, markup, plot=False):
    forw = dataset.index.get_indexer([forward],  method='nearest')[0]
    back = dataset.index.get_indexer([backward], method='nearest')[0]

    close = dataset['close'].to_numpy()
    lab   = dataset['labels'].to_numpy()
    meta  = dataset['meta_labels'].to_numpy()

    rpt, ch, lf, lb = process_data(close, lab, meta, markup, forw, back)

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

import numpy as np

def evaluate_report(report: np.ndarray, r2_raw: float) -> float:
    if len(report) < 2:
        return -1.0

    returns = np.diff(report)
    num_trades = len(returns)
    if num_trades < 5:
        return -1.0

    # ────────────────────────
    # MÉTRICAS BASE
    gains = returns[returns > 0]
    losses = -returns[returns < 0]
    profit_factor = np.sum(gains) / np.sum(losses) if np.sum(losses) > 0 else 0.0

    equity_curve = report
    peak = equity_curve[0]
    max_dd = 0.0
    for x in equity_curve:
        peak = max(peak, x)
        max_dd = max(max_dd, peak - x)
    total_return = equity_curve[-1] - equity_curve[0]
    return_dd_ratio = total_return / max_dd if max_dd > 0 else 0.0

    # ────────────────────────
    # PUNTAJE COMPUESTO BASE
    base_score = (
        (profit_factor * 0.4) +  # Menor peso al profit factor
        (return_dd_ratio * 0.6)  # Mayor peso al return/DD
    )

    # Penalizaciones por métricas débiles
    penalization = 1.0
    if profit_factor < 2.0: penalization *= 0.8  # Umbral más alto
    if return_dd_ratio < 2.0: penalization *= 0.8  # Umbral más alto

    # ────────────────────────
    # AJUSTE POR TRADES
    min_trades = 200  # Mínimo de trades deseado
    trade_weight = min(1.0, num_trades / min_trades)  # Penalización lineal
    if num_trades > 50:
        trade_weight = min(2.0, 1.0 + (num_trades - 50) / 100)  # Bonus gradual
    
    base_score *= penalization * trade_weight

    # ────────────────────────
    # R² y pendiente
    if r2_raw > 0:
        r2_raw *= 1.0
    else:
        return -1.0

    # ────────────────────────
    # Score final
    final_score = 0.5 * base_score + 0.5 * r2_raw
    
    return final_score

def tester_one_direction(dataset, forward, backward,
                         markup, direction='buy', plot=False):
    forw = dataset.index.get_indexer([forward],  method='nearest')[0]
    back = dataset.index.get_indexer([backward], method='nearest')[0]

    close = dataset['close'].to_numpy()
    lab   = dataset['labels'].to_numpy()
    meta  = dataset['meta_labels'].to_numpy()

    rpt, ch, lf, lb = process_data_one_direction(
        close, lab, meta, markup, forw, back, direction)

    y = rpt.reshape(-1, 1)
    X = np.ascontiguousarray(np.arange(len(rpt))).reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    sign = 1 if lr.coef_[0][0] >= 0 else -1

    if plot:
        plt.plot(rpt)
        plt.axvline(lf, color='purple', ls=':')
        plt.axvline(lb, color='red',    ls=':')
        plt.plot(lr.predict(X))
        plt.title(f"R² {lr.score(X, y) * sign:.2f}")
        plt.show()

    r2_raw = lr.score(X, y) * sign
    #return lr.score(X, y) * sign
    return evaluate_report(rpt, r2_raw)

def test_model_one_direction_clustering(
        dataset: pd.DataFrame,
        result:  list,
        forward: datetime,
        backward: datetime,
        markup:  float,
        direction: str,
        plt: bool = False):

    pr_tst = dataset.copy()
    X = pr_tst.drop(columns=['close'])
    X_meta = X.loc[:,  X.columns.str.contains('meta_feature')]
    X      = X.loc[:, ~X.columns.str.contains('meta_feature')]

    pr_tst['labels']      = result[0].predict_proba(X)[:,1]
    pr_tst['meta_labels'] = result[1].predict_proba(X_meta)[:,1]

    # Corrección aquí:
    pr_tst[['labels', 'meta_labels']] = (pr_tst[['labels', 'meta_labels']] > 0.5).astype(float)

    return tester_one_direction(pr_tst, forward, backward, markup, direction, plt)

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
               markup: float,
               plt=False):

    ext_dataset = dataset.copy()
    X = ext_dataset.iloc[:, 1:]

    ext_dataset['labels']      = result[0].predict_proba(X)[:, 1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X)[:, 1]
    ext_dataset[['labels', 'meta_labels']] = ext_dataset[['labels', 'meta_labels']].gt(0.5).astype(float)

    return tester(ext_dataset, forward, backward, markup, plot=plt)


def test_model_one_direction(dataset: pd.DataFrame,
                             result: list,
                             forward: datetime,
                             backward: datetime,
                             markup: float,
                             direction: str = 'buy',
                             plt=False):

    ext_dataset = dataset.copy()
    # Extraer características en el mismo orden que se usaron en el entrenamiento
    feature_cols = sorted(ext_dataset.filter(regex='_feature$').columns)
    X = ext_dataset[feature_cols]

    ext_dataset['labels'] = result[0].predict_proba(X)[:, 1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X)[:, 1]
    ext_dataset[['labels', 'meta_labels']] = ext_dataset[['labels', 'meta_labels']].gt(0.5).astype(float)

    return tester_one_direction(ext_dataset, forward, backward, markup,
                               direction=direction, plot=plt)