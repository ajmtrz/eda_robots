import pandas as pd
import math
from datetime import datetime
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from bots.botlibs.labeling_lib import *
from bots.botlibs.tester_lib import test_model_one_direction

def get_prices() -> pd.DataFrame:
    p = pd.read_csv('files/'+hyper_params['symbol']+'.csv', sep='\s+')
    pFixed = pd.DataFrame(columns=['time', 'close'])
    pFixed['time'] = p['<DATE>'] + ' ' + p['<TIME>']
    pFixed['time'] = pd.to_datetime(pFixed['time'], format='mixed')
    pFixed['close'] = p['<CLOSE>']
    pFixed.set_index('time', inplace=True)
    pFixed.index = pd.to_datetime(pFixed.index, unit='s')
    return pFixed.dropna()

def get_features(data: pd.DataFrame) -> pd.DataFrame:
    pFixed = data.copy()
    pFixedC = data.copy()
    count = 0

    for i in hyper_params['periods']:
        pFixed[str(count)] = pFixedC.rolling(i).std()
        count += 1
    return pFixed.dropna()

def meta_learner(folds_number: int, iter: int, depth: int, l_rate: float) -> pd.DataFrame:
    dataset = get_labels_one_direction(get_features(get_prices()), 
                                       markup=hyper_params['markup'], 
                                       min=1,
                                       max=15,
                                       direction=hyper_params['direction'])
    data = dataset[(dataset.index < hyper_params['forward']) 
                   & (dataset.index > hyper_params['backward'])].copy()

    X = data[data.columns[1:-2]]
    y = data['labels']

    B_S_B = pd.DatetimeIndex([])

    # learn meta model with CV method
    meta_model = CatBoostClassifier(iterations = iter,
                                max_depth = depth,
                                learning_rate=l_rate,
                                verbose = False)
    cv = StratifiedKFold(n_splits=folds_number, shuffle=False)
    predicted = cross_val_predict(meta_model, X, y, method='predict_proba', cv=cv)
    
    coreset = X.copy()
    coreset['labels'] = y
    coreset['labels_pred'] = [x[0] < 0.5 for x in predicted]
    coreset['labels_pred'] = coreset['labels_pred'].apply(lambda x: 0 if x < 0.5 else 1)
    
    # select bad samples (bad labels indices)
    diff_negatives = coreset['labels'] != coreset['labels_pred']
    B_S_B = B_S_B.append(diff_negatives[diff_negatives == True].index)
    to_mark = B_S_B.value_counts()
    marked_idx = to_mark.index
    data['meta_labels'] = 1.0
    data.loc[data.index.isin(marked_idx), 'meta_labels'] = 0.0
    # data.loc[data.index.isin(marked_idx), 'labels'] = 0.0

    return data[data.columns[:]]

def meta_learners(models_number: int, iterations: int, depth: int, bad_samples_fraction: float):
    dataset = get_labels_one_direction(get_features(get_prices()),
                                       markup=hyper_params['markup'],
                                       min=1,
                                       max=15,
                                       direction=hyper_params['direction'])
    data = dataset[(dataset.index < hyper_params['forward']) & (dataset.index > hyper_params['backward'])].copy()

    X = data[data.columns[1:-2]]
    y = data['labels']

    BAD_WAIT = pd.DatetimeIndex([])
    BAD_TRADE = pd.DatetimeIndex([])

    for i in range(models_number):
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, train_size = 0.5, test_size = 0.5, shuffle = True)
        
        # learn debias model with train and validation subsets
        meta_m = CatBoostClassifier(iterations = iterations,
                                depth = depth,
                                custom_loss = ['Accuracy'],
                                eval_metric = 'Accuracy',
                                verbose = False,
                                use_best_model = True)
        
        meta_m.fit(X_train, y_train, eval_set = (X_val, y_val), plot = False)
        
        coreset = X.copy()
        coreset['labels'] = y
        coreset['labels_pred'] = meta_m.predict_proba(X)[:, 1]
        coreset['labels_pred'] = coreset['labels_pred'].apply(lambda x: 0 if x < 0.5 else 1)
        
        # add bad samples of this iteration (bad labels indices)
        coreset_w = coreset[coreset['labels']==0]
        coreset_t = coreset[coreset['labels']==1]

        diff_negatives_w = coreset_w['labels'] != coreset_w['labels_pred']
        diff_negatives_t = coreset_t['labels'] != coreset_t['labels_pred']
        BAD_WAIT = BAD_WAIT.append(diff_negatives_w[diff_negatives_w == True].index)
        BAD_TRADE = BAD_TRADE.append(diff_negatives_t[diff_negatives_t == True].index)

    to_mark_w = BAD_WAIT.value_counts()
    to_mark_t = BAD_TRADE.value_counts()
    marked_idx_w = to_mark_w[to_mark_w > to_mark_w.mean() * bad_samples_fraction].index
    marked_idx_t = to_mark_t[to_mark_t > to_mark_t.mean() * bad_samples_fraction].index

    data['meta_labels'] = 1.0
    data.loc[data.index.isin(marked_idx_w), 'meta_labels'] = 0.0
    data.loc[data.index.isin(marked_idx_t), 'meta_labels'] = 0.0
    # data.loc[data.index.isin(marked_idx_t), 'labels'] = 0.0
    # 👇 Añade visualización aquí
    print("Dataset para meta_learners():")
    print(dataset.head())
    print("\nDescripción del dataset:")
    print(dataset.describe())
    # 👆
    return data[data.columns[:]]

def fit_final_models(dataset) -> list:
    # features for model\meta models. We learn main model only on filtered labels 
    X, X_meta = dataset[dataset['meta_labels']==1], dataset[dataset.columns[1:-2]]
    X = X[X.columns[1:-2]]
    
    # labels for model\meta models
    y, y_meta = dataset[dataset['meta_labels']==1], dataset[dataset.columns[-1]]
    y = y[y.columns[-2]]
    
    y = y.astype('int16')
    y_meta = y_meta.astype('int16')

    # train\test split
    train_X, test_X, train_y, test_y = train_test_split(
        X, y, train_size=0.8, test_size=0.2, shuffle=True)
    
    train_X_m, test_X_m, train_y_m, test_y_m = train_test_split(
        X_meta, y_meta, train_size=0.8, test_size=0.2, shuffle=True)

    # learn main model with train and validation subsets
    model = CatBoostClassifier(iterations=300,
                               custom_loss=['Accuracy'],
                               eval_metric='Accuracy',
                               verbose=False,
                               use_best_model=True,
                               task_type='CPU')
    model.fit(train_X, train_y, eval_set=(test_X, test_y),
              early_stopping_rounds=15, plot=False)
    
    # learn meta model with train and validation subsets
    meta_model = CatBoostClassifier(iterations=300,
                                    custom_loss=['Accuracy'],
                                    eval_metric='Accuracy',
                                    verbose=False,
                                    use_best_model=True,
                                    task_type='CPU')
    meta_model.fit(train_X_m, train_y_m, eval_set=(test_X_m, test_y_m),
              early_stopping_rounds=15, plot=False)
    data = get_features(get_prices())
    R2 = test_model_one_direction(data, 
                    [model, meta_model], 
                    hyper_params['stop_loss'], 
                    hyper_params['take_profit'],
                    hyper_params['full forward'],
                    hyper_params['backward'],
                    hyper_params['markup'],
                    hyper_params['direction'],
                    plt=False)
    
    if math.isnan(R2):
        R2 = -1.0
        print('R2 is fixed to -1.0')
    print('R2: ' + str(R2))
    return [R2, model, meta_model]

def export_model_to_ONNX(model, model_number):
    model[1].save_model(
    hyper_params['export_path'] +'catmodel' + str(model_number) +'.onnx',
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'test model for BinaryClassification',
        'onnx_graph_name': 'CatBoostModel_for_BinaryClassification'
    },
    pool=None)

    model[2].save_model(
    hyper_params['export_path'] + 'catmodel_m' + str(model_number) +'.onnx',
    format="onnx",
    export_parameters={
        'onnx_domain': 'ai.catboost',
        'onnx_model_version': 1,
        'onnx_doc_string': 'test model for BinaryClassification',
        'onnx_graph_name': 'CatBoostModel_for_BinaryClassification'
    },
    pool=None)
    
    code = '#include <Math\Stat\Math.mqh>'
    code += '\n'
    code += '#resource "catmodel'+str(model_number)+'.onnx" as uchar ExtModel[]'
    code += '\n'
    code += '#resource "catmodel_m'+str(model_number)+'.onnx" as uchar ExtModel2[]'
    code += '\n'
    code += 'int Periods' + '[' + str(len(hyper_params['periods'])) + \
        '] = {' + ','.join(map(str, hyper_params['periods'])) + '};'
    code += '\n\n'

    # get features
    code += 'void fill_arays' + '( double &features[]) {\n'
    code += '   double pr[], ret[];\n'
    code += '   ArrayResize(ret, 1);\n'
    code += '   for(int i=ArraySize(Periods'')-1; i>=0; i--) {\n'
    code += '       CopyClose(NULL,PERIOD_H1,1,Periods''[i],pr);\n'
    code += '       ret[0] = MathStandardDeviation(pr);\n'
    code += '       ArrayInsert(features, ret, ArraySize(features), 0, WHOLE_ARRAY); }\n'
    code += '   ArraySetAsSeries(features, true);\n'
    code += '}\n\n'

    file = open(hyper_params['export_path'] + str(hyper_params['symbol']) + ' ONNX include' + str(model_number) + '.mqh', "w")
    file.write(code)

    file.close()
    print('The file ' + 'ONNX include' + '.mqh ' + 'has been written to disk')


hyper_params = {
    'symbol': 'XAUUSD_H1',
    'export_path': '/Users/dmitrievsky/Library/Containers/com.isaacmarovitz.Whisky/Bottles/54CFA88F-36A3-47F7-915A-D09B24E89192/drive_c/Program Files/MetaTrader 5/MQL5/Include/Mean reversion/',
    'model_number': 0,
    'markup': 0.20,
    'stop_loss':  10.0000,
    'take_profit': 5.0000,
    'direction': 'buy',
    'periods': [i for i in range(5, 300, 30)],
    'backward': datetime(2020, 1, 1),
    'forward': datetime(2024, 1, 1),
    'full forward': datetime(2026, 1, 1),
}

models = []
for i in range(5):
    print('Learn ' + str(i) + ' model')
    # models.append(fit_final_models(meta_learner(5, 15, 3, 0.1)))
    models.append(fit_final_models(meta_learners(25, 15, 3, 0.8)))


models.sort(key=lambda x: x[0])
data = get_features(get_prices())
test_model_one_direction(data,
        models[-1][1:],
        hyper_params['stop_loss'],
        hyper_params['take_profit'],
        hyper_params['forward'],
        hyper_params['backward'],
        hyper_params['markup'],
        hyper_params['direction'],
        plt=True)

export_model_to_ONNX(models[-1], 0)

models[-1][1].get_best_score()['validation']
