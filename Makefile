# ============================================================
#  Modele GLOBAL poole (production)
# ============================================================
# Pipeline de prod en deux temps :
#   make train_xgb_global        # entraine LE modele global (2015+, COVID exclu)
#   make save_features_global    # pre-calcule les features de service pour l'API
#
# Parametres (surchargeables) :
#   TRAIN_FROM : date de debut des donnees (defaut 2015-01-01)
#   DAYS       : nb de jours recents servables par l'API (defaut 120)

GRAIN      ?= station
HORIZON    ?=
TRAIN_FROM ?= 2015-01-01
LABEL      ?= reference
NOTES      ?=
DAYS       ?= 120
N_ITER     ?= 30

train_xgb_global:
	@python -m naviflow.interface.main_xgb_global

save_features_global:
	@DAYS=$(DAYS) python -m naviflow.interface.main_save_features_global

baseline_xgb:
	@GRAIN=$(GRAIN) HORIZON=$(HORIZON) TRAIN_FROM=$(TRAIN_FROM) \
		python -m naviflow.interface.main_baselines

eval:
	@LABEL="$(LABEL)" NOTES="$(NOTES)" LAGS="$(LAGS)" ROLLS="$(ROLLS)" LOG_TARGET="$(LOG_TARGET)" GLOBAL="$(GLOBAL)" ALL="$(ALL)" \
		python -m naviflow.interface.main_eval

tune_global:
	@N_ITER="$(N_ITER)" LAGS="$(LAGS)" ROLLS="$(ROLLS)" \
		python -m naviflow.interface.main_tune_global

download:
	python -m naviflow.gcp.main_gcp download

baseline:
	python -m naviflow.gcp.main_gcp baseline
