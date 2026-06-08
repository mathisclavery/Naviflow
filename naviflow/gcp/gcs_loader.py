"""Téléchargement des données brutes depuis GCS vers raw_data/ local.

Usage :
    python -m naviflow.gcp.gcs_loader

Télécharge uniquement les fichiers absents en local (reprise possible).
"""

import os
from pathlib import Path

from google.cloud import storage

from naviflow.config import (
    GCP_PROJECT,
    BUCKET_NAME
)

BUCKET_FOLDER = "raw"
LOCAL_FOLDER  = Path(__file__).resolve().parent.parent.parent / "raw_data"


def download_raw_data():
    """Télécharge raw/ depuis GCS vers raw_data/ local.

    Ignore les fichiers déjà présents en local.
    """
    client = storage.Client(project=GCP_PROJECT)
    blobs  = list(client.list_blobs(BUCKET_NAME, prefix=f"{BUCKET_FOLDER}/"))

    print(f"📋 {len(blobs)} fichiers trouvés dans gs://{BUCKET_NAME}/{BUCKET_FOLDER}/\n")

    downloaded = 0
    skipped    = 0

    for blob in blobs:
        # Chemin relatif depuis raw/ -> ex: "validations/data-2015-validations/..."
        relative_path = Path(blob.name).relative_to(BUCKET_FOLDER)
        local_path    = LOCAL_FOLDER / relative_path

        # Dossier vide (dossier GCS fictif) -> skip
        if blob.name.endswith("/"):
            continue

        # Fichier déjà présent -> skip
        if local_path.exists():
            print(f"  ⏭️  Déjà présent : {relative_path}")
            skipped += 1
            continue

        # Téléchargement
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"  ⬇️  Téléchargé  : {relative_path}")
        downloaded += 1

    print(f"\n{'='*50}")
    print(f"✅ {downloaded} fichiers téléchargés, {skipped} déjà présents.")
    print(f"{'='*50}")


if __name__ == "__main__":
    download_raw_data()
