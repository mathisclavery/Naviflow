def upload_raw():
    """
    Upload les données brutes vers GCS.
    """
    import upload_raw_data
    upload_raw_data.upload_folder(
        upload_raw_data.LOCAL_FOLDER,
        upload_raw_data.BUCKET_FOLDER
    )
    
run_upload_raw:
    python -m naviflow.interface.main upload_raw
