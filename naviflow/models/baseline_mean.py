"""Baseline naive : prédire toujours la moyenne du train set."""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def run_baseline_mean(X, y, test_size=0.2, random_state=67):
    """
    Baseline naive : prédit toujours la moyenne de y_train.
    Sert de point de comparaison minimum pour les autres modèles.

    Args:
        X: features preprocessées (DataFrame issu de preprocess_to_dataframe)
        y: target (Series NB_VALD_TOTAL)
        test_size: proportion du test set (défaut 0.2)
        random_state: seed pour la reproductibilité

    Returns:
        dict avec mae, r2, y_pred, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    y_pred = [y_train.mean()] * len(y_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("=" * 50)
    print("BASELINE — Prédiction de la moyenne")
    print("=" * 50)
    print(f"MAE test        : {mae:.0f}")
    print(f"R²              : {r2:.3f}")
    print(f"Erreur relative : {mae / y_test.mean() * 100:.1f}%")

    return {
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test,
    }