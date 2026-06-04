"""XGBoost avec RandomizedSearchCV pour optimisation des hyperparamètres."""

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
import pandas as pd

"""Création d'une target décalée pour la prédiction multi-horizon (J+N)."""

def add_target_horizon(df, horizon=7, target='NB_VALD_TOTAL',
                       group='ID_LIEU', date='JOUR'):
    """
    Crée une target décalée de `horizon` jours dans le futur (approche directe).

    Au lieu de prédire l'affluence du jour courant, le modèle apprend à
    prédire l'affluence dans N jours à partir des features/lags d'aujourd'hui.

    Exemple : pour J+7, la ligne du 1er janvier aura comme target
    l'affluence réelle du 8 janvier. Le modèle apprend donc
    "features du 1er janvier -> affluence du 8 janvier".

    La colonne créée s'appelle 'target_jN' (ex: 'target_j7' pour horizon=7).

    Args:
        df: DataFrame contenant déjà les lags (issu de add_lags)
        horizon: nombre de jours à prédire en avance (défaut 7)
        target: colonne cible
        group: colonne de regroupement (station)
        date: colonne de date pour le tri

    Returns:
        DataFrame avec la colonne target_jN ajoutée et les NaN de fin supprimés
    """
    df = df.copy()
    df[date] = pd.to_datetime(df[date])
    df = df.sort_values([group, date])

    target_col = f'target_j{horizon}'
    df[target_col] = df.groupby(group)[target].shift(-horizon)

    df = df.dropna(subset=[target_col])

    return df



DEFAULT_PARAM_GRID = {
    'n_estimators':     [300, 500, 700, 900, 1100, 1300, 1500],
    'max_depth':        [3, 4, 5, 6, 7, 8],
    'learning_rate':    [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
    'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 2, 3, 5, 7],
    'gamma':            [0, 0.1, 0.3, 0.5],
    'reg_alpha':        [0, 0.1, 0.5, 1.0],
    'reg_lambda':       [0.5, 1.0, 1.5, 2.0],
}


def run_xgboost(X, y, test_size=0.2, random_state=67, cv=5,
                n_iter=50, param_grid=None):
    """
    Entraîne un XGBoost avec RandomizedSearchCV pour optimiser les hyperparamètres.

    Args:
        X: features preprocessées (DataFrame issu de preprocess_to_dataframe)
        y: target (Series NB_VALD_TOTAL)
        test_size: proportion du test set (défaut 0.2)
        random_state: seed pour la reproductibilité
        cv: nombre de folds pour la cross-validation (défaut 5)
        n_iter: nombre d'itérations du RandomizedSearch (défaut 50).
                Avec une grille plus large, augmenter n_iter (ex: 80-100)
                pour mieux explorer l'espace des hyperparamètres.
        param_grid: dict des hyperparamètres à tester. Si None, utilise DEFAULT_PARAM_GRID.

    Returns:
        dict avec model, mae, r2, mae_cv, best_params, y_pred, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    search = RandomizedSearchCV(
        XGBRegressor(),
        grid,
        n_iter=n_iter,
        cv=cv,
        scoring='neg_mean_absolute_error',
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    model  = search.best_estimator_
    y_pred = model.predict(X_test)

    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    mae_cv = -search.best_score_

    print("=" * 50)
    print("XGBOOST")
    print("=" * 50)
    print(f"Meilleurs params : {search.best_params_}")
    print(f"MAE CV           : {mae_cv:.0f}")
    print(f"MAE test         : {mae:.0f}")
    print(f"R2               : {r2:.3f}")
    print(f"Erreur relative  : {mae / y_test.mean() * 100:.1f}%")

    return {
        'model': model,
        'mae': mae,
        'r2': r2,
        'mae_cv': mae_cv,
        'best_params': search.best_params_,
        'y_pred': y_pred,
        'y_test': y_test,
    }