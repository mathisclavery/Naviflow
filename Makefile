# ============================================================
#  Entrainement XGBoost
# ============================================================
# Parametres (surchargeables en ligne de commande) :
#   GRAIN      : station (defaut) ou cluster
#   N_ITER     : iterations du RandomizedSearch (defaut 50)
#   HORIZON    : horizon de prediction J+N (vide = jour courant = j0)
#   TRAIN_FROM : date de debut des donnees (defaut = tout l'historique)
#   FORCE      : 1 pour reentrainer et ecraser les modeles existants
#
# Exemples :
#   make train_xgb                                   # par cluster, j0, tout l'historique
#   make train_xgb GRAIN=station                     # par station
#   make train_xgb GRAIN=station N_ITER=80           # par station, recherche plus large
#   make train_xgb GRAIN=cluster HORIZON=7           # par cluster, prediction J+7
#   make train_xgb TRAIN_FROM=2024-01-01 N_ITER=5   # test rapide
#   make train_xgb TRAIN_FROM=2021-01-01            # post-COVID seulement
#   make train_xgb FORCE=1                           # ecrase les modeles existants

GRAIN      ?= cluster
N_ITER     ?= 50
HORIZON    ?=
TRAIN_FROM ?= 2015-01-01
FORCE      ?= 0

train_xgb:
	@GRAIN=$(GRAIN) N_ITER=$(N_ITER) HORIZON=$(HORIZON) \
	 TRAIN_FROM=$(TRAIN_FROM) FORCE=$(FORCE) \
		python -m naviflow.interface.main_xgb

download:
	python -m naviflow.gcp.main_gcp download

baseline:
	python -m naviflow.gcp.main_gcp baseline
