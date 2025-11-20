# %%
import optuna
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from modules.StrategySearcher import StrategySearcher
import warnings
import os
warnings.filterwarnings("ignore")

configs = [
    dict(
        symbol='XAUUSD', timeframe='H1', direction='both', pruner_type='successive',
        train_start=datetime(2018,4,1), train_end=datetime(2025,8,1),
        test_start=datetime(2023,8,1),  test_end=datetime(2024,4,1),
        label_method='random', search_type='clusters', search_subtype='kmeans',
        n_models=1, debug=True,
    ),
    # dict(
    #     symbol='GDAXI', timeframe='H1', direction='both', pruner_type='successive',
    #     train_start=datetime(2018,4,1), train_end=datetime(2026,4,1),
    #     test_start=datetime(2022,4,1),  test_end=datetime(2023,4,1),
    #     label_method='random', search_type='clusters', search_subtype='kmeans',
    #     n_models=1, debug=False,
    # ),
    # dict(
    #     symbol='NDX', timeframe='H1', direction='both', pruner_type='successive',
    #     train_start=datetime(2018,4,1), train_end=datetime(2026,4,1),
    #     test_start=datetime(2022,4,1),  test_end=datetime(2023,4,1),
    #     label_method='random', search_type='clusters', search_subtype='kmeans',
    #     n_models=1, debug=False,
    # ),
]

# Crear tag para cada configuración
for cfg in configs:
    # Construir el tag asegurando que no haya dobles guiones bajos por campos vacíos
    tag_parts = [
        cfg['symbol'],
        cfg['timeframe'],
        cfg['direction'],
        cfg['label_method'][:2],
        cfg['search_type'][:3],
        (cfg.get('search_subtype') or '')[:2],
    ]
    # Filtrar partes vacías y unir con "_"
    cfg["tag"] = "_".join([part for part in tag_parts if part]).strip("_")

DB_FILE = f"optuna_dbs/{cfg['tag']}.db"
DB_PATH = f"sqlite:///{DB_FILE}"
STUDY_NAME = f"{cfg['tag']}"

study = None
if not os.path.exists(DB_FILE):
    study = None
else:
    try:
        study = optuna.load_study(study_name=STUDY_NAME, storage=DB_PATH)
    except Exception:
        study = None

n_trials = 5000
if study:
    n_trials = len(study.trials) + n_trials

for cfg in configs:
    cfg['n_trials'] = n_trials

def launch(cfg):
    s = StrategySearcher(**cfg)
    s.run_search()

with ProcessPoolExecutor(max_workers=len(configs)) as pool:
    futures = {pool.submit(launch, c): c["tag"] for c in configs}
    for f in as_completed(futures):
        tag = futures[f]
        try:
            print(f"[{tag}] terminado")
        except Exception as e:
            print(f"[{tag}] falló: {e}")


