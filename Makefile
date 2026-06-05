# ============================================================
#  Entrainement XGBoost
# ============================================================
# Parametres (surchargeables en ligne de commande) :
#   GRAIN    : station (defaut) ou cluster
#   N_ITER   : iterations du RandomizedSearch (defaut 50)
#   HORIZON  : horizon de prediction J+N (vide = jour courant)
#
# Exemples :
#   make train_xgb                          # par station, 50 iters
#   make train_xgb GRAIN=cluster            # par cluster
#   make train_xgb GRAIN=station N_ITER=80  # par station, recherche plus large
#   make train_xgb GRAIN=cluster HORIZON=7  # par cluster, prediction J+7

GRAIN   ?= cluster
N_ITER  ?= 50
HORIZON ?=
FORCE ?= 0

train_xgb:
	@GRAIN=$(GRAIN) N_ITER=$(N_ITER) HORIZON=$(HORIZON) FORCE=$(FORCE) \
		python -m naviflow.interface.main_xgb

download:
	python -m naviflow.gcp.main_gcp download

baseline:
	python -m naviflow.gcp.main_gcp baseline
