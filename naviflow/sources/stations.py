"""Detection et fusion des stations portant plusieurs ID_LIEU.

Une meme station physique peut apparaitre sous plusieurs ID_LIEU au fil du
temps (changement de referentiel STIF -> IDFM) ou simultanement (double
comptage). Ce module :

  1. detecte automatiquement ces groupes par cle de libelle normalisee ;
  2. regroupe les cles qui partagent des IDs (ex. 'havr caumartin' et
     'havre caumartin' designent la meme station) ;
  3. classe chaque groupe (SUCCESSION / DOUBLON / COMPLEMENTAIRE) selon le
     chevauchement temporel et le ratio de volumes ;
  4. en deduit une action de fusion (merge_sum / merge_max) ;
  5. applique la fusion, puis regroupe les poles d'echange (Chatelet...).

La logique de classification reproduit celle du notebook d'analyse, mais ici
elle est recalculee a chaque load() : aucune table figee a maintenir.

Parametres ajustables dans config.py :
  - THRESHOLD_DAYS    : en deca, deux IDs sont consideres successifs.
  - RATIO_DOUBLON     : seuil du ratio de volumes (doublon vs complementaire).
  - STATION_OVERRIDES : forcage manuel de l'action pour certaines cles.
  - POLES_DEFINITION  : regroupement des stations co-localisees.
  - POLE_ID_BASE      : base des ID synthetiques negatifs des poles.
"""

import re
import unicodedata

import pandas as pd

from naviflow.config import (
    THRESHOLD_DAYS,
    RATIO_DOUBLON,
    STATION_OVERRIDES,
    POLES_DEFINITION,
    POLE_ID_BASE,
)


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def label_key(label):
    """Cle de comparaison normalisee d'un libelle de station.

    Minuscules, sans accents, sans suffixes de ligne (T12, RER E...), sans
    ponctuation. Sert au regroupement interne, pas a l'affichage.
    Ex : 'Chatelet-Les Halles' -> 'chatelet les halles'.
    """
    if pd.isna(label):
        return ""
    s = str(label)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"\b(t\d{1,2}|rer\s*[a-e])\b", " ", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


# --------------------------------------------------------------------------- #
# Analyse temporelle et de volume
# --------------------------------------------------------------------------- #
def _periodes(df, ids):
    """min / max / count de JOUR pour chaque ID, trie par date de debut."""
    sub = df[df["ID_LIEU"].isin(ids)]
    return (
        sub.groupby("ID_LIEU")["JOUR"]
           .agg(["min", "max", "count"])
           .reset_index()
           .sort_values("min")
    )


def _max_overlap_days(periodes):
    """Plus grand chevauchement temporel deux a deux (en jours), 0 si aucun."""
    rows = periodes.to_dict("records")
    best = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            ov = (min(a["max"], b["max"]) - max(a["min"], b["min"])).days
            best = max(best, ov)
    return best


def _means_on_common_period(df, ids):
    """Moyenne journaliere de chaque ID sur la periode commune a TOUS les IDs.

    Renvoie None si les IDs n'ont aucune fenetre temporelle commune.
    """
    sub = df[df["ID_LIEU"].isin(ids)]
    bornes = sub.groupby("ID_LIEU")["JOUR"].agg(["min", "max"])
    start = bornes["min"].max()
    end = bornes["max"].min()
    if start > end:
        return None
    common = sub[(sub["JOUR"] >= start) & (sub["JOUR"] <= end)]
    return common.groupby("ID_LIEU")["NB_VALD_TOTAL"].mean()


# --------------------------------------------------------------------------- #
# Union-Find pour regrouper les cles qui partagent des IDs
# --------------------------------------------------------------------------- #
def _group_keys_by_shared_ids(key_to_ids):
    """Regroupe les cles qui partagent au moins un ID_LIEU.

    Un meme ID peut porter plusieurs libelles au fil du temps (ex. ID 73688 =
    'Havr Caumartin' puis 'Havre Caumartin'), generant des cles differentes
    qui designent en realite la meme station physique. On fusionne ces cles
    en composantes connexes (Union-Find sur le graphe cle-ID-cle).

    Renvoie une liste de groupes : [{'keys': [...], 'ids': [...]}, ...].
    Chaque groupe est une station physique (au sens : meme ensemble d'IDs).
    """
    parent = {k: k for k in key_to_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # union des cles qui partagent un meme ID
    id_to_keys = {}
    for key, ids in key_to_ids.items():
        for i in ids:
            id_to_keys.setdefault(i, []).append(key)
    for keys_sharing_id in id_to_keys.values():
        for k in keys_sharing_id[1:]:
            union(keys_sharing_id[0], k)

    # collecte des composantes
    groups = {}
    for key, ids in key_to_ids.items():
        root = find(key)
        g = groups.setdefault(root, {"keys": set(), "ids": set()})
        g["keys"].add(key)
        g["ids"].update(ids)

    return [{"keys": sorted(g["keys"]), "ids": sorted(g["ids"])}
            for g in groups.values()]


# --------------------------------------------------------------------------- #
# Detection + classification automatique
# --------------------------------------------------------------------------- #
def detect_station_actions(df, threshold_days=THRESHOLD_DAYS,
                           ratio_doublon=RATIO_DOUBLON,
                           overrides=STATION_OVERRIDES):
    """Calcule automatiquement les actions de fusion par station physique.

    Renvoie un dict : { cle_canonique: (liste_ids, id_cible, action) }
    ou cle_canonique est la cle la plus 'longue' du groupe (typiquement la
    forme la moins abregee : 'havre caumartin' plutot que 'havr caumartin').

    Classification :
      - SUCCESSION    (overlap < threshold_days)         -> merge_sum
      - DOUBLON       (overlap, ratio >= ratio_doublon)  -> merge_max
      - COMPLEMENTAIRE(overlap, ratio <  ratio_doublon)  -> merge_sum
    Un override (cle -> action) a la priorite ; 'skip' = ignorer le groupe.
    Les overrides matchent n'importe quelle cle du groupe.
    """
    df = df.copy()
    df["_key"] = df["LIBELLE_ARRET"].apply(label_key)
    key_to_ids = df.groupby("_key")["ID_LIEU"].apply(lambda s: sorted(s.unique())).to_dict()

    groups = _group_keys_by_shared_ids(key_to_ids)

    actions = {}
    for group in groups:
        ids = group["ids"]
        keys = group["keys"]
        if len(ids) < 2:
            continue

        # cle canonique : la plus longue (= forme la moins abregee)
        canonical_key = max(keys, key=len)

        periodes = _periodes(df, ids)
        id_cible = int(periodes.sort_values("max").iloc[-1]["ID_LIEU"])
        overlap = _max_overlap_days(periodes)

        # action automatique
        if overlap < threshold_days:
            action = "merge_sum"               # succession
        else:
            means = _means_on_common_period(df, ids)
            if means is None or means.max() == 0:
                action = "merge_max"           # indetermine -> garde-un par defaut
            else:
                ratio = means.min() / means.max()
                action = "merge_max" if ratio >= ratio_doublon else "merge_sum"

        # override manuel : applique si l'une des cles du groupe est listee
        for k in keys:
            if k in overrides:
                action = overrides[k]
                break

        if action == "skip":
            continue
        actions[canonical_key] = ([int(i) for i in ids], id_cible, action)

    return actions


# --------------------------------------------------------------------------- #
# Application de la fusion
# --------------------------------------------------------------------------- #
def _apply_station_actions(df, station_actions):
    """Niveaux 1 & 2 : remappe les IDs vers leur cible et re-agrege.

    merge_max -> un seul volume par jour (max), evite le double-comptage.
    merge_sum -> somme classique.
    """
    df = df.copy()
    id_to_action, id_to_target = {}, {}
    for ids, id_cible, action in station_actions.values():
        for i in ids:
            id_to_action[i] = action
            id_to_target[i] = id_cible

    df["_action"] = df["ID_LIEU"].map(id_to_action).fillna("keep")
    df["ID_LIEU"] = df["ID_LIEU"].map(id_to_target).fillna(df["ID_LIEU"]).astype("Int64")

    mask_max = df["_action"] == "merge_max"
    part_max = (
        df[mask_max]
        .groupby(["JOUR", "ID_LIEU"], as_index=False)
        .agg(NB_VALD_TOTAL=("NB_VALD_TOTAL", "max"),
             LIBELLE_ARRET=("LIBELLE_ARRET", "last"))
    )
    part_rest = df[~mask_max][["JOUR", "ID_LIEU", "NB_VALD_TOTAL", "LIBELLE_ARRET"]]
    return pd.concat([part_max, part_rest], ignore_index=True)


def _apply_poles(df, poles_definition, pole_id_base):
    """Niveau 3 : regroupe les stations co-localisees sous un ID synthetique negatif."""
    df = df.copy()
    df["_key"] = df["LIBELLE_ARRET"].apply(label_key)

    pole_id_map, pole_labels = {}, {}
    for offset, (pole, cles) in enumerate(poles_definition.items()):
        synth_id = pole_id_base - offset
        pole_labels[synth_id] = pole
        for i in df.loc[df["_key"].isin(cles), "ID_LIEU"].unique():
            pole_id_map[i] = synth_id

    df["ID_LIEU"] = df["ID_LIEU"].replace(pole_id_map)
    for synth_id, name in pole_labels.items():
        df.loc[df["ID_LIEU"] == synth_id, "LIBELLE_ARRET"] = name
    return df


def merge_stations(df, threshold_days=THRESHOLD_DAYS, ratio_doublon=RATIO_DOUBLON,
                   overrides=STATION_OVERRIDES, poles_definition=POLES_DEFINITION,
                   pole_id_base=POLE_ID_BASE):
    """[Etape 4] Detecte et fusionne les stations multi-ID, puis les poles.

    Pipeline complet :
      1. detect_station_actions : classe chaque groupe et choisit l'action.
      2. _apply_station_actions : applique la fusion (niveaux 1 & 2).
      3. _apply_poles           : regroupe les poles d'echange (niveau 3).
      4. re-agregation finale   : unicite (JOUR, ID_LIEU).

    Les actions sont recalculees a chaque appel : rien n'est fige en dur.
    """
    station_actions = detect_station_actions(df, threshold_days, ratio_doublon, overrides)
    df = _apply_station_actions(df, station_actions)
    df = _apply_poles(df, poles_definition, pole_id_base)
    return (
        df.groupby(["JOUR", "ID_LIEU"], as_index=False)
          .agg(NB_VALD_TOTAL=("NB_VALD_TOTAL", "sum"),
               LIBELLE_ARRET=("LIBELLE_ARRET", "last"))
    )
