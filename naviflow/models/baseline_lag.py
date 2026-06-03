"""Baseline naïve de persistance : prédire l'affluence du même jour la semaine dernière (lag_7)."""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def run_baseline_lag(X, y, test_size=0.2, random_state=67, lag_col='lag_7'):
    """
    Baseline de persistance : prédit l'affluence d'un lag passé.

    Par défaut utilise lag_7 (même jour la semaine dernière) — c'est la
    meilleure baseline naïve pour des données journalières avec saisonnalité
    hebdomadaire, car elle capture automatiquement le pattern lundi/dimanche.

    C'est une baseline plus exigeante que la simple moyenne : elle force le
    modèle ML à prouver qu'il apporte mieux que "demain ressemble à la
    semaine dernière".

    Args:
        X: features preprocessées contenant la colonne de lag
        y: target (Series NB_VALD_TOTAL)
        test_size: proportion du test set (défaut 0.2)
        random_state: seed pour la reproductibilité
        lag_col: colonne de lag à utiliser comme prédiction (défaut 'lag_7')

    Returns:
        dict avec mae, r2, y_pred, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Prédiction = valeur du lag (ex: affluence du même jour semaine dernière)
    y_pred = X_test[lag_col]

    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)

    print("=" * 50)
    print(f"BASELINE — Persistance ({lag_col})")
    print("=" * 50)
    print(f"MAE test        : {mae:.0f}")
    print(f"R2              : {r2:.3f}")
    print(f"Erreur relative : {mae / y_test.mean() * 100:.1f}%")

    return {
        'mae': mae,
        'r2': r2,
        'y_pred': y_pred,
        'y_test': y_test,
    }