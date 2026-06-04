"""XGBoost avec RandomizedSearchCV pour optimisation des hyperparamètres."""

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBRegressor
import pandas as pd

"""Création d'une target décalée pour la prédiction multi-horizon (J+N)."""

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
