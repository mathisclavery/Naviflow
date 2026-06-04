"""Lecture des fichiers de validations IDFM.

Detection automatique de l'encodage et du separateur, puis lecture validee.
Ces fonctions transforment des fichiers sur disque en DataFrames pandas bruts ;
le nettoyage vient apres (module `validations`).
"""

from pathlib import Path

import pandas as pd

from naviflow.config import ENCODINGS, FILE_KIND, SEPARATORS, YEAR_CONFIG


def build_path(data_dir, year, period, ext):
    """Construit le chemin d'un fichier de validations.

    Convention IDFM : data-{year}-validations/{year}-{period}-validations.{ext}
    """
    year_dir = Path(data_dir) / f"{FILE_KIND}" / f"data-{year}-validations"
    return year_dir / f"{year}-{period}-{FILE_KIND}.{ext}"


def _try_read(file_path, encoding, sep, expected_cols, nrows=None):
    """Tente une lecture ; renvoie le df si les colonnes attendues sont la, sinon None."""
    try:
        df = pd.read_csv(file_path, sep=sep, encoding=encoding,
                         encoding_errors="replace", nrows=nrows, low_memory=False)
    except Exception:
        return None
    return df if expected_cols.issubset(df.columns) else None


def detect_formats(data_dir, expected_cols, year_config=YEAR_CONFIG, verbose=False):
    """Detecte (encoding, sep) de chaque fichier en testant les combinaisons.

    Renvoie {(year, period): (encoding, sep)}. Le scan brute-force ne tourne
    qu'ICI, une seule fois (sur 5 lignes), et son resultat est explicite.
    """
    formats = {}
    if verbose:
        print("=== Detection des formats ===")
    for year, periods in year_config.items():
        for period, ext in periods:
            file_path = build_path(data_dir, year, period, ext)
            found = None
            for enc in ENCODINGS:
                for sep in SEPARATORS:
                    if _try_read(file_path, enc, sep, expected_cols, nrows=5) is not None:
                        found = (enc, sep)
                        break
                if found:
                    break
            if found is None:
                raise ValueError(f"Aucun format valide pour : {file_path}")
            formats[(year, period)] = found
            if verbose:
                sep_name = {"\t": "TAB", ";": ";"}[found[1]]
                print(f"  {year}-{period:<3}: encoding={found[0]:<11} sep={sep_name}")
    return formats


def read_file(file_path, encoding, sep, expected_cols):
    """Lit un fichier avec encodage/separateur connus.

    `encoding_errors="replace"` evite de planter sur un octet isole incompatible
    (ex. un accent mal encode dans un nom de station) plus loin dans le fichier,
    la ou la detection sur 5 lignes ne l'avait pas vu. Valide ensuite la presence
    des colonnes attendues pour ecarter une lecture silencieusement fausse.
    """
    df = pd.read_csv(file_path, sep=sep, encoding=encoding,
                     encoding_errors="replace", low_memory=False)
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Colonnes inattendues dans {file_path} : {list(df.columns)}")
    return df


def load_source(data_dir, formats, expected_cols, year_config=YEAR_CONFIG):
    """Charge tous les fichiers dans un dict {annee: {periode: df}}.

    `formats` provient de detect_formats() : aucun scan ici.
    """
    data = {}
    for year, periods in year_config.items():
        year_data = {}
        for period, ext in periods:
            file_path = build_path(data_dir, year, period, ext)
            enc, sep = formats[(year, period)]
            year_data[period] = read_file(file_path, enc, sep, expected_cols)
        data[str(year)] = year_data
    return data


def diagnose_files(data_dir, year_config=YEAR_CONFIG):
    """Verifie l'existence physique de chaque fichier attendu (sans l'ouvrir).

    Renvoie la liste des (year, period, path) manquants.
    """
    print("=== Diagnostic : presence des fichiers ===")
    missing = []
    for year, periods in year_config.items():
        for period, ext in periods:
            p = build_path(data_dir, year, period, ext)
            ok = p.exists()
            if not ok:
                missing.append((year, period, p))
            print(f"  [{'OK ' if ok else 'MANQUANT'}] {year}-{period:<3} -> {p}")
    print(f"\n{len(missing)} fichier(s) manquant(s).")
    return missing
