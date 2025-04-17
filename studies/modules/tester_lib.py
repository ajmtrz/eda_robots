from numba import jit
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


@jit(nopython=True)
def process_data(close, labels, metalabels, stop, take, markup, forward, backward):
    last_deal = 2
    last_price = 0.0
    report = [0.0]
    chart = [0.0]
    line_f = 0
    line_b = 0

    for i in range(len(close)):
        line_f = len(report) if i <= forward else line_f
        line_b = len(report) if i <= backward else line_b
        
        pred = labels[i]
        pr = close[i]
        pred_meta = metalabels[i]  # 1 = allow trades

        if last_deal == 2 and pred_meta == 1:
            last_price = pr
            last_deal = 0 if pred < 0.5 else 1
            continue
        
        if last_deal == 0:
            if (-markup + (pr - last_price) >= take) or (-markup + (last_price - pr) >= stop):
                last_deal = 2
                profit = -markup + (pr - last_price)
                report.append(report[-1] + profit)
                chart.append(chart[-1] + profit)
                continue

        if last_deal == 1:
            if (-markup + (pr - last_price) >= stop) or (-markup + (last_price - pr) >= take):
                last_deal = 2
                profit = -markup + (last_price - pr)
                report.append(report[-1] + profit)
                chart.append(chart[-1] + (pr - last_price))
                continue
        
        # close deals by signals
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
def process_data_one_direction(close, labels, metalabels, stop, take, markup, forward, backward, direction):
    last_deal = 2
    last_price = 0.0
    report = [0.0]
    chart = [0.0]
    line_f = 0
    line_b = 0

    for i in range(len(close)):
        line_f = len(report) if i <= forward else line_f
        line_b = len(report) if i <= backward else line_b
        
        pred = labels[i]
        pr = close[i]
        pred_meta = metalabels[i]  # 1 = allow trades

        if last_deal == 2 and pred_meta == 1:
            last_price = pr
            last_deal = 2 if pred < 0.5 else 1
            continue
        
        if last_deal == 1 and direction == 'buy':
            if (-markup + (pr - last_price) >= take) or (-markup + (last_price - pr) >= stop):
                last_deal = 2
                profit = -markup + (pr - last_price)
                report.append(report[-1] + profit)
                chart.append(chart[-1] + profit)
                continue

        if last_deal == 1 and direction == 'sell':
            if (-markup + (pr - last_price) >= stop) or (-markup + (last_price - pr) >= take):
                last_deal = 2
                profit = -markup + (last_price - pr)
                report.append(report[-1] + profit)
                chart.append(chart[-1] + (pr - last_price))
                continue
        
        # close deals by signals
        if last_deal == 1 and pred < 0.5 and direction == 'buy':
            last_deal = 2
            profit = -markup + (pr - last_price)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + profit)
            continue

        if last_deal == 1 and pred < 0.5 and direction == 'sell':
            last_deal = 2
            profit = -markup + (last_price - pr)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + (pr - last_price))
            continue

    return np.array(report), np.array(chart), line_f, line_b

def tester(*args):
    '''
    This is a fast strategy tester based on numba
    List of parameters:

    dataset: must contain first column as 'close' and last columns with "labels" and "meta_labels"

    stop: stop loss value

    take: take profit value

    forward: forward time interval

    backward: backward time interval

    markup: markup value

    plot: false/true
    '''
    dataset, stop, take, forward, backward, markup, plot = args

    forw = dataset.index.get_indexer([forward], method='nearest')[0]
    backw = dataset.index.get_indexer([backward], method='nearest')[0]

    close = dataset['close'].to_numpy()
    labels = dataset['labels'].to_numpy()
    metalabels = dataset['meta_labels'].to_numpy()
    
    report, chart, line_f, line_b = process_data(close, labels, metalabels, stop, take, markup, forw, backw)

    y = report.reshape(-1, 1)
    X = np.arange(len(report)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)

    l = 1 if lr.coef_[0][0] >= 0 else -1

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

def tester_one_direction(*args):
    '''
    This is a fast strategy tester based on numba
    List of parameters:

    dataset: must contain first column as 'close' and last columns with "labels" and "meta_labels"

    stop: stop loss value

    take: take profit value

    forward: forward time interval

    backward: backward time interval

    markup: markup value

    direction: buy/sell

    plot: false/true
    '''
    dataset, stop, take, forward, backward, markup, direction, plot = args

    forw = dataset.index.get_indexer([forward], method='nearest')[0]
    backw = dataset.index.get_indexer([backward], method='nearest')[0]

    close = dataset['close'].to_numpy()
    labels = dataset['labels'].to_numpy()
    metalabels = dataset['meta_labels'].to_numpy()
    
    report, chart, line_f, line_b = process_data_one_direction(close, labels, metalabels, stop, take, markup, forw, backw, direction)

    y = report.reshape(-1, 1)
    X = np.arange(len(report)).reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(X, y)

    l = 1 if lr.coef_[0][0] >= 0 else -1

    if plot:
        plt.plot(report)
        plt.axvline(x=line_f, color='purple', ls=':', lw=1, label='OOS')
        plt.axvline(x=line_b, color='red', ls=':', lw=1, label='OOS2')
        plt.plot(lr.predict(X))
        plt.title("Strategy performance R^2 " + str(format(lr.score(X, y) * l, ".2f")))
        plt.xlabel("the number of trades")
        plt.ylabel("cumulative profit in pips")
        plt.show()

    return lr.score(X, y) * l

def tester_slow(dataset, stop, take, markup, forward, plot=False):
    last_deal = 2
    last_price = 0.0
    report = [0.0]
    chart = [0.0]
    line_f = 0
    line_b = 0

    # Assuming variables are defined elsewhere
    forw = dataset.index.get_indexer([forward], method='nearest')
    backw = dataset.index.get_indexer([forward], method='nearest')

    close = dataset['close'].to_numpy()
    labels = dataset['labels'].to_numpy()
    metalabels = dataset['meta_labels'].to_numpy()
    
    for i in range(dataset.shape[0]):
        line_f = len(report) if i <= forw else line_f
        line_b = len(report) if i <= backw else line_b
        
        pred = labels[i]
        pr = close[i]
        pred_meta = metalabels[i]  # 1 = allow trades

        if last_deal == 2 and pred_meta == 1:
            last_price = pr
            last_deal = 0 if pred < 0.5 else 1
            continue
        
        if last_deal == 0:
            if (-markup + (pr - last_price) >= take) or (-markup + (last_price - pr) >= stop):
                last_deal = 2
                profit = -markup + (pr - last_price)
                report.append(report[-1] + profit)
                chart.append(chart[-1] + profit)
                continue

        if last_deal == 1:
            if (-markup + (pr - last_price) >= stop) or (-markup + (last_price - pr) >= take):
                last_deal = 2
                profit = -markup + (last_price - pr)
                report.append(report[-1] + profit)
                chart.append(chart[-1] + (pr - last_price))
                continue
        
        # close deals by signals
        if last_deal == 0 and pred > 0.5 and pred_meta == 1:
            last_deal = 2
            profit = -markup + (pr - last_price)
            report.append(report[-1] + profit)
            chart.append(chart[-1] + profit)
            continue

        if last_deal == 1 and pred < 0.5 and pred_meta == 1:
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

def test_model(dataset: pd.DataFrame, 
               result: list, 
               stop: float, 
               take: float, 
               forward: float, 
               backward: float, 
               markup: float, 
               plt = False):
    
    ext_dataset = dataset.copy()
    X = ext_dataset[ext_dataset.columns[1:]]

    ext_dataset['labels'] = result[0].predict_proba(X)[:,1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X)[:,1]
    ext_dataset['labels'] = ext_dataset['labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    ext_dataset['meta_labels'] = ext_dataset['meta_labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    return tester(ext_dataset, stop, take, forward, backward, markup, plt)

def test_model_one_direction(dataset: pd.DataFrame, 
               result: list, 
               stop: float, 
               take: float, 
               forward: float, 
               backward: float, 
               markup: float,
               direction: str, 
               plt = False):
    
    ext_dataset = dataset.copy()
    X = ext_dataset[ext_dataset.columns[1:]]

    ext_dataset['labels'] = result[0].predict_proba(X)[:,1]
    ext_dataset['meta_labels'] = result[1].predict_proba(X)[:,1]
    ext_dataset['labels'] = ext_dataset['labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    ext_dataset['meta_labels'] = ext_dataset['meta_labels'].apply(lambda x: 0.0 if x < 0.5 else 1.0)
    return tester_one_direction(ext_dataset, stop, take, forward, backward, markup, direction, plt)
