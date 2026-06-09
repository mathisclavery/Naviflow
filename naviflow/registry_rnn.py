import pandas as pd


from naviflow.config import PROJECT_ROOT

MODELS_STORE = PROJECT_ROOT / "models_store"


def _horizon_tag(horizon):
    """Etiquette horizon : None -> 'j0', N -> 'jN'."""
    return "j0" if horizon is None else f"j{horizon}"


def run_dir(grain="station", horizon=None, train_from=None):
    """Dossier du run pour une combinaison grain / horizon / periode.

    Nom : {grain}_{horizon}_{train_from}  (ex. station_j7_20220701, cluster_j0_all).
    """
    h = _horizon_tag(horizon)
    tf = (train_from or "all").replace("-", "")
    return MODELS_STORE / f"{grain}_{h}_{tf}"


def model_path(group_id, grain="station", horizon=None, train_from=None):
    """Chemin du fichier modele d'un groupe, dans le sous-dossier de son run."""
    return run_dir(grain, horizon, train_from) / f"xgb_{group_id}.json"


def results_path(grain="station", horizon=None, train_from=None):
    """Chemin du results.csv du run."""
    return run_dir(grain, horizon, train_from) / "results.csv"


def save_model(model, group_id, grain="station", horizon=None, train_from=None):
    """Sauvegarde un XGBRegressor au format natif, dans le dossier du run."""
    d = run_dir(grain, horizon, train_from)
    d.mkdir(parents=True, exist_ok=True)
    path = model_path(group_id, grain, horizon, train_from)
    model.save_model(path)
    return path


def load_model(group_id, grain="station", horizon=None, train_from=None):
    """Recharge un XGBRegressor depuis le dossier du run (un seul modele)."""
    path = model_path(group_id, grain, horizon, train_from)
    if not path.exists():
        raise FileNotFoundError(f"Aucun modele : {path}")
    model = XGBRegressor()
    model.load_model(path)
    return model


def save_results(results, grain="station", horizon=None, train_from=None):
    """Ecrit le results.csv dans le sous-dossier du run."""
    d = run_dir(grain, horizon, train_from)
    d.mkdir(parents=True, exist_ok=True)

    if isinstance(results, dict):
        rows = [{"group_id": gid, **metrics} for gid, metrics in results.items()]
    else:
        rows = list(results)

    df = pd.DataFrame(rows)
    df.insert(0, "grain", grain)
    df.insert(1, "horizon", _horizon_tag(horizon))
    df.insert(2, "train_from", train_from or "all")

    path = results_path(grain, horizon, train_from)
    df.to_csv(path, index=False)
    return path
