"""XGBoost multi-sortie avec RandomizedSearchCV pour optimisation des hyperparamètres."""

from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
import pandas as pd
import numpy as np

# Grille élargie + régularisation fine
DEFAULT_PARAM_GRID = {
    'n_estimators':     [500, 800, 1200, 1600, 2000, 3000],
    'max_depth':        [3, 4, 5, 6, 7, 8, 10],
    'learning_rate':    [0.005, 0.01, 0.02, 0.03, 0.05, 0.08],
    'subsample':        [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel':[0.6, 0.8, 1.0],
    'min_child_weight': [1, 2, 3, 5, 7, 10],
    'gamma':            [0, 0.1, 0.3, 0.5, 1.0],
    'reg_alpha':        [0, 0.1, 0.5, 1.0, 2.0],
    'reg_lambda':       [0.5, 1.0, 1.5, 2.0, 3.0],
}


def run_linear_regression(X, y, test_size=0.2, random_state=67, cv=5):
    """
    Entraîne une régression linéaire avec cross-validation.

    Args:
        X: features preprocessées (DataFrame issu de preprocess_to_dataframe)
        y: target (Series NB_VALD_TOTAL)
        test_size: proportion du test set (défaut 0.2)
        random_state: seed pour la reproductibilité
        cv: nombre de folds pour la cross-validation (défaut 5)

    Returns:
        dict avec model, mae, r2, mae_cv, mae_cv_std, y_pred, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LinearRegression()

    # Cross-validation sur le train set
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv, scoring='neg_mean_absolute_error'
    )
    mae_cv     = -cv_scores.mean()
    mae_cv_std = cv_scores.std()

    # Fit final + évaluation test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("=" * 50)
    print("LINEAR REGRESSION")
    print("=" * 50)
    print(f"MAE CV ({cv} folds): {mae_cv:.0f} (+/- {mae_cv_std:.0f})")
    print(f"MAE test          : {mae:.0f}")
    print(f"R²                : {r2:.3f}")
    print(f"Erreur relative   : {mae / y_test.mean() * 100:.1f}%")

    return {
        'model': model,
        'mae': mae,
        'r2': r2,
        'mae_cv': mae_cv,
        'mae_cv_std': mae_cv_std,
        'y_pred': y_pred,
        'y_test': y_test,
    }


def run_xgboost(X, Y, dates, test_size=0.2, random_state=67, cv=3,
                n_iter=40, param_grid=None, use_gpu=False):
    """
    Entraîne un XGBoost multi-sortie optimisé avec RandomizedSearchCV.

    Predit le vecteur [J+1, ..., J+H] en une fois grace a
    multi_strategy='multi_output_tree'.

    Args:
        X: features preprocessées (2D)
        Y: targets multi-horizon (2D, shape [n, H]) issues de prepare_xgb
        test_size: proportion du test set
        random_state: seed
        cv: folds de cross-validation (defaut 3, multi-output plus lourd)
        n_iter: itérations du RandomizedSearch (défaut 40)
        param_grid: grille custom. Si None, utilise DEFAULT_PARAM_GRID.
        use_gpu: si True, device='cuda'

    Returns:
        dict avec model, mae, r2, mae_cv, mae_per_h, r2_per_h, best_params,
        y_pred, y_test
    """

    dates = np.asarray(dates, dtype='datetime64[ns]')

    unique_days = np.unique(dates)
    cutoff = unique_days[int(len(unique_days) * (1 - test_size))]

    train_mask = dates < cutoff
    test_mask  = ~train_mask

    X = np.asarray(X)
    Y = np.asarray(Y)
    train_order = np.argsort(dates[train_mask], kind='stable')

    X_train = X[train_mask][train_order]
    y_train = Y[train_mask][train_order]
    X_test  = X[test_mask]
    y_test  = Y[test_mask]

    grid = param_grid if param_grid is not None else DEFAULT_PARAM_GRID

    base_kwargs = dict(
        objective='reg:squarederror',         # MAE non supporte en multi-output
        tree_method='hist',                    # rapide
        multi_strategy='multi_output_tree',    # une structure d'arbre -> vecteur
        eval_metric='mae',
        random_state=random_state,
    )
    if use_gpu:
        base_kwargs['device'] = 'cuda'

    tscv = TimeSeriesSplit(n_splits=cv)

    search = RandomizedSearchCV(
        XGBRegressor(**base_kwargs),
        grid,
        n_iter=n_iter,
        cv=tscv,
        scoring='neg_mean_absolute_error',     # moyenne sur les sorties
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )

    search.fit(X_train, y_train)

    model  = search.best_estimator_
    y_pred = model.predict(X_test)

    # Metriques globales (moyenne sur tous les horizons)
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    mae_cv = -search.best_score_

    # Metriques par horizon (colonne par colonne)
    mae_raw = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2_raw  = r2_score(y_test, y_pred, multioutput='raw_values')
    horizons = list(range(1, Y.shape[1] + 1))
    mae_per_h = {h: float(m) for h, m in zip(horizons, mae_raw)}
    r2_per_h  = {h: float(r) for h, r in zip(horizons, r2_raw)}

    # print("=" * 50)
    # print("XGBOOST MULTI-SORTIE")
    # print("=" * 50)
    # print(f"Meilleurs params : {search.best_params_}")
    # print(f"MAE CV           : {mae_cv:.0f}")
    # print(f"MAE test (moy)   : {mae:.0f}")
    # print(f"R2 (moy)         : {r2:.3f}")
    # print(f"MAE par horizon  : {mae_per_h}")

    return {
        'model': model,
        'mae': mae,
        'r2': r2,
        'mae_cv': mae_cv,
        'mae_per_h': mae_per_h,
        'r2_per_h': r2_per_h,
        'best_params': search.best_params_,
        'y_pred': y_pred,
        'y_test': y_test,
    }
