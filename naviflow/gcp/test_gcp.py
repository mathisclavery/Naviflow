from google.cloud import storage

from naviflow.config import (
    GCP_PROJECT,
    BUCKET_NAME
)

# Connexion au projet
client = storage.Client(project=GCP_PROJECT)

# Accès au bucket
bucket = client.bucket(BUCKET_NAME)

# Test 1 — Vérifier que le bucket est accessible
print("✅ Bucket accessible :", bucket.exists())

# Test 2 — Lister les dossiers existants
print("\n📁 Contenu du bucket :")
blobs = client.list_blobs(BUCKET_NAME)
for blob in blobs:
    print(" -", blob.name)

# Test 3 — Uploader un fichier test
blob = bucket.blob("raw/test_connexion.txt")
blob.upload_from_string("Connexion NaviFlow OK !")
print("\n⬆️  Fichier test uploadé dans raw/")

# Test 4 — Relire le fichier uploadé
contenu = blob.download_as_text()
print("⬇️  Contenu lu :", contenu)

# Test 5 — Supprimer le fichier test
blob.delete()
print("🗑️  Fichier test supprimé")
