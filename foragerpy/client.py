import asyncio
import json
import os
import pickle
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np
from PIL import Image
from tqdm import tqdm

from foragerpy.utils import make_identifier, parse_gcs_path, resize_image

GET_DATASET_ENDPOINT = "api/get_dataset_info"
CREATE_DATASET_ENDPOINT = "api/create_dataset"
DELETE_DATASET_ENDPOINT = "api/delete_dataset"
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")

INDEX_UPLOAD_GCS_PATH = "gs://foragerml/indexes/"  # trailing slash = directory
AUX_LABELS_UPLOAD_GCS_PATH = "gs://foragerml/aux_labels/"  # trailing slash = directory
THUMBNAIL_UPLOAD_GCS_PATH = "gs://foragerml/thumbnails/"  # trailing slash = directory
RESIZE_MAX_HEIGHT = 200


class Client(object):
    def __init__(self, uri: Optional[str] = None, use_proxy: bool = False) -> None:
        if not use_proxy and (
            "http_proxy" in os.environ or "https_proxy" in os.environ
        ):
            print(
                "WARNING: http_proxy/https_proxy env variables set, but "
                "--use_proxy flag not specified. Will not use proxy."
            )

        self.uri = uri if uri else self._default_uri()
        self.use_proxy = use_proxy

    def _default_uri(self):
        return "http://localhost:8000"

    async def add_dataset(
        self, name: str, train_images_directory: str, val_images_directory: str
    ):
        # Make sure that a dataset with this name doesn't already exist
        async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
            async with session.get(
                os.path.join(self.uri, GET_DATASET_ENDPOINT, name)
            ) as response:
                print(response)
                assert response.status == 404, f"Dataset {name} already exists"

        # Add to database
        params = {
            "dataset": name,
            "train_images_directory": train_images_directory,
            "val_images_directory": val_images_directory,
        }
        add_db_start = time.time()
        if True:
            async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
                async with session.post(
                    os.path.join(self.uri, CREATE_DATASET_ENDPOINT), json=params
                ) as response:
                    j = await response.json()
                    assert j["status"] == "success", j
        add_db_end = time.time()
        print(f"Time: {add_db_end - add_db_start}")

    async def delete_dataset(self, name: str):
        params = {
            "dataset": name,
        }
        async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
            async with session.post(
                os.path.join(self.uri, DELETE_DATASET_ENDPOINT), json=params
            ) as response:
                j = await response.json()
                assert j["status"] == "success", j

    async def import_labels(self, dataset_name: str):
        pass

    async def export_labels(self, dataset_name: str):
        pass

    async def import_embeddings(self, dataset_name: str):
        pass

    async def import_scores(self, dataset_name: str):
        pass
