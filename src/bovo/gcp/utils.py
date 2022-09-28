# more infos at whttps://console.cloud.google.com/storage/browser?project=bovo-predict&prefix=

from pathlib import Path

from google.cloud import storage

DATA_PATH = Path("data/")

_client = storage.Client()


def upload(path: Path | str):
    path = Path(path)
    check_path(path)

    in_data_path = path.relative_to(DATA_PATH)
    bucket_name = in_data_path.parts[0]
    bucket = _client.bucket(bucket_name)

    if path.is_file():
        in_bucket_path = path.relative_to(DATA_PATH / bucket_name)
        blob = bucket.blob(str(in_bucket_path))
        upload_blob(blob, path)
    else:
        for file_path in path.glob("*"):
            if file_path.is_file():
                in_bucket_path = file_path.relative_to(DATA_PATH / bucket_name)
                blob = bucket.blob(str(in_bucket_path))
                upload_blob(blob, file_path)


def upload_blob(blob: storage.Blob, save_path: Path):
    with open(save_path, "rb") as f:
        blob.upload_from_file(f)


def download(path: Path | str):
    path = Path(path)
    check_path(path)

    in_data_path = path.relative_to(DATA_PATH)
    bucket_name = in_data_path.parts[0]
    in_bucket_path = path.relative_to(DATA_PATH / bucket_name)

    for blob in _client.list_blobs(bucket_name, prefix=str(in_bucket_path)):
        file_save_path = DATA_PATH / bucket_name / blob.name
        download_blob(blob, file_save_path)


def download_blob(blob_or_uri, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "wb") as f:
        _client.download_blob_to_file(blob_or_uri, f)


def check_path(path: Path):
    if DATA_PATH not in path.parents:
        raise ValueError(
            f"Path not supported. Supposed to be in the form {DATA_PATH}/bucket_name/folder/blob"
        )
