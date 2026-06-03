"""Régression linéaire avec cross-validation."""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score


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