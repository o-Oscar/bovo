from pathlib import Path

from google.cloud import storage

DATA_PATH = Path("data/")

_client = storage.Client()


def upload(path: Path | str):
    save_path = Path(path)
    print(check_path(save_path))


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
