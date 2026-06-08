# =============================================================================
# upload_raw_data.py
# Script d'upload des données brutes vers Google Cloud Storage
# =============================================================================
import os
import time

from google.cloud import storage

from naviflow.config import (
    GCP_PROJECT,
    BUCKET_NAME
)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
LOCAL_FOLDER  = "raw_data"             # Dossier local à uploader
BUCKET_FOLDER = "raw"                  # Dossier cible dans le bucket
TIMEOUT     = 300                      # Timeout par fichier en secondes
MAX_RETRIES = 3                        # Nombre de tentatives en cas d'échec

# -----------------------------------------------------------------------------
# CONNEXION À GCP
# Les credentials sont lus automatiquement depuis la variable d'environnement
# GOOGLE_APPLICATION_CREDENTIALS
# -----------------------------------------------------------------------------
client = storage.Client(project=GCP_PROJECT)
bucket = client.bucket(BUCKET_NAME)

# -----------------------------------------------------------------------------
# FONCTIONS
# -----------------------------------------------------------------------------
def get_existing_blobs():
    """
    Récupère la liste des fichiers déjà présents dans le bucket.
    Permet de reprendre un upload interrompu sans re-uploader les fichiers déjà envoyés.
    Returns:
        set: ensemble des chemins de fichiers déjà présents dans le bucket
    """
    blobs = client.list_blobs(BUCKET_NAME, prefix=f"{BUCKET_FOLDER}/")
    return {blob.name for blob in blobs}


def upload_file(local_path, blob_path):
    """
    Upload un fichier local vers le bucket avec gestion des erreurs et retry.
    Args:
        local_path (str): chemin du fichier local
        blob_path  (str): chemin cible dans le bucket
    Returns:
        bool: True si l'upload a réussi, False sinon
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(local_path, timeout=TIMEOUT)
            print(f"  ⬆️  Uploadé : {blob_path}")
            return True

        except Exception as e:
            print(f"  ⚠️  Tentative {attempt}/{MAX_RETRIES} échouée : {e}")
            time.sleep(5)  # Pause avant de réessayer

    # Toutes les tentatives ont échoué
    print(f"  ❌ Échec définitif : {blob_path}")
    return False


def upload_folder(local_folder, bucket_folder):
    """
    Upload récursif d'un dossier local vers le bucket GCS.
    Les fichiers déjà présents dans le bucket sont ignorés.
    Args:
        local_folder  (str): chemin du dossier local à uploader
        bucket_folder (str): dossier cible dans le bucket
    """
    # Récupération des fichiers déjà uploadés
    already_uploaded = get_existing_blobs()
    print(f"📋 {len(already_uploaded)} fichiers déjà présents dans le bucket\n")

    success_count = 0
    failure_count = 0

    # Parcours récursif du dossier local
    for root, dirs, files in os.walk(local_folder):
        for filename in files:

            # Chemin complet du fichier local
            local_path = os.path.join(root, filename)

            # Chemin relatif par rapport au dossier racine
            # ex: "meteo/data-2025/fichier.csv"
            relative_path = os.path.relpath(local_path, local_folder)

            # Chemin complet dans le bucket
            # ex: "raw/meteo/data-2025/fichier.csv"
            blob_path = os.path.join(bucket_folder, relative_path)

            # On ignore les fichiers déjà uploadés
            if blob_path in already_uploaded:
                print(f"  ⏭️  Déjà uploadé : {blob_path}")
                continue

            # Upload du fichier
            if upload_file(local_path, blob_path):
                success_count += 1
            else:
                failure_count += 1

    # Résumé final
    print(f"\n{'='*50}")
    print(f"✅ Upload terminé : {success_count} fichiers uploadés")
    if failure_count > 0:
        print(f"❌ {failure_count} fichiers en échec — relance le script pour réessayer")
    print(f"{'='*50}")


# -----------------------------------------------------------------------------
# POINT D'ENTRÉE
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    upload_folder(LOCAL_FOLDER, BUCKET_FOLDER)
