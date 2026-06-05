"""Chargement d'un calendrier categorise pour le projet d'affluence.

Produit un DataFrame avec une ligne par jour entre START_YEAR et END_YEAR,
indiquant pour chaque date si c'est un weekend, un jour ferie, un jour de
vacances scolaires (zone C - Paris) et/ou un pont.
"""

from datetime import date, timedelta

import holidays
import pandas as pd
from vacances_scolaires_france import SchoolHolidayDates
from naviflow.config import START_YEAR, END_YEAR, ZONE


def _is_pont(d, feries):
    """Un jour est un 'pont' si c'est un lundi ou un vendredi entoure
    d'un jour ferie et d'un weekend.

    - Vendredi : pont si le jeudi est ferie (week-end prolonge avant)
    - Lundi    : pont si le mardi est ferie (week-end prolonge apres)
    """
    if d.weekday() == 4:  # vendredi
        return (d - timedelta(days=1)) in feries
    if d.weekday() == 0:  # lundi
        return (d + timedelta(days=1)) in feries
    return False


def load_calendrier(start_year=START_YEAR, end_year=END_YEAR):
    """Charge un calendrier categorise jour par jour.

    Parametres
    ----------
    start_year, end_year : bornes inclusives sur l'annee (defaut : 2015-2026).

    Renvoie
    -------
    pd.DataFrame
        Colonnes : JOUR (datetime), IS_WEEKEND, IS_FERIE, IS_VACANCES, IS_PONT.
        Une ligne par jour de la periode. Les quatre indicateurs sont des
        booleens independants : un jour peut etre a la fois ferie et en
        vacances (ex. 14 juillet en ete), ou ferie et weekend, etc.
    """
    # 1. Generer toutes les dates de la periode
    jours = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="D",
    )
    df = pd.DataFrame({"JOUR": jours})

    # 2. Preparer les references feries et vacances une seule fois
    feries = holidays.France(years=range(start_year, end_year + 1))
    sh = SchoolHolidayDates()

    # 3. Calculer les 4 indicateurs (on convertit JOUR en date pour comparer)
    dates = df["JOUR"].dt.date
    df["IS_WEEKEND"] = df["JOUR"].dt.weekday >= 5
    df["IS_FERIE"] = dates.map(lambda d: d in feries)
    df["IS_VACANCES"] = dates.map(lambda d: sh.is_holiday_for_zone(d, ZONE))
    df["IS_PONT"] = dates.map(lambda d: _is_pont(d, feries))

    return df
