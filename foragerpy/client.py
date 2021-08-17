import asyncio
import json
import os
import pickle
import time
import uuid
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, TypedDict

import aiohttp
import numpy as np
from PIL import Image
from tqdm import tqdm

from foragerpy.utils import make_identifier, parse_gcs_path, resize_image

FORAGER_EMBEDDINGS_DIR = Path("~/.forager/embeddings").expanduser()
FORAGER_SCORES_DIR = Path("~/.forager/scores").expanduser()
FORAGER_IMAGE_LISTS_DIR = Path("~/.forager/image_lists").expanduser()

GET_DATASET_ENDPOINT = "api/get_dataset_info"
CREATE_DATASET_ENDPOINT = "api/create_dataset"
DELETE_DATASET_ENDPOINT = "api/delete_dataset"
IMPORT_LABELS_ENDPOINT = "api/add_annotations_multi"
EXPORT_LABELS_ENDPOINT = "api/get_annotations"
IMPORT_EMBEDDINGS_ENDPOINT = "api/add_model_output"
IMPORT_SCORES_ENDPOINT = "api/add_model_output"
IMAGE_EXTENSIONS = ("jpg", "jpeg", "png")

INDEX_UPLOAD_GCS_PATH = "gs://foragerml/indexes/"  # trailing slash = directory
AUX_LABELS_UPLOAD_GCS_PATH = "gs://foragerml/aux_labels/"  # trailing slash = directory
THUMBNAIL_UPLOAD_GCS_PATH = "gs://foragerml/thumbnails/"  # trailing slash = directory
RESIZE_MAX_HEIGHT = 200


class Annotation(TypedDict):
    path: str
    category: str
    mode: str


class Client(object):
    def __init__(
        self, user_email: str, uri: Optional[str] = None, use_proxy: bool = False
    ) -> None:
        if not use_proxy and (
            "http_proxy" in os.environ or "https_proxy" in os.environ
        ):
            print(
                "WARNING: http_proxy/https_proxy env variables set, but "
                "--use_proxy flag not specified. Will not use proxy."
            )

        self.user_email = user_email
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
                assert response.status == 404, f"Dataset {name} already exists"

        # Add to database
        params = {
            "dataset": name,
            "train_images_directory": train_images_directory,
            "val_images_directory": val_images_directory,
        }
        async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
            async with session.post(
                os.path.join(self.uri, CREATE_DATASET_ENDPOINT), json=params
            ) as response:
                j = await response.json()
                assert j["status"] == "success", j

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

    async def import_annotations(
        self,
        dataset_name: str,
        annotations: List[Annotation],
    ):
        params = {
            "dataset": dataset_name,
            "user": self.user_email,
            "annotations": [
                {
                    "category": label["category"],
                    "mode": label["mode"],
                    "path": label["path"],
                }
                for label in annotations
            ],
        }
        async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
            async with session.post(
                os.path.join(self.uri, IMPORT_LABELS_ENDPOINT),
                json=params,
            ) as response:
                j = await response.json()
                assert j["created"] == len(annotations), j

    async def export_annotations(
        self, dataset_name: str, categories: List[str] = [], modes: List[str] = []
    ) -> List[Annotation]:
        params = {
            "dataset_name": dataset_name,
            "by_path": True,
        }
        if categories:
            params["tags"] = categories

        if modes:
            params["modes"] = modes

        async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
            async with session.post(
                os.path.join(self.uri, EXPORT_LABELS_ENDPOINT),
                json=params,
            ) as response:
                j = await response.json()
        return j

    async def import_embeddings(
        self,
        dataset_name: str,
        name: str,
        paths: List[str],
        embeddings: np.ndarray,
    ):
        FORAGER_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        model_uuid = str(uuid.uuid4())
        embeddings_path = FORAGER_EMBEDDINGS_DIR / model_uuid

        embeddings_mm = np.memmap(
            embeddings_path,
            dtype="float32",
            mode="w+",
            shape=embeddings.shape,
        )
        embeddings_mm[:] = embeddings[:]
        embeddings_mm.flush()

        params = {"name": name, "paths": paths, "embeddings_path": str(embeddings_path)}
        async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
            async with session.post(
                os.path.join(self.uri, IMPORT_EMBEDDINGS_ENDPOINT, dataset_name),
                json=params,
            ) as response:
                j = await response.json()
                assert j["status"] == "success", j

    async def import_scores(
        self,
        dataset_name: str,
        name: str,
        paths: List[str],
        scores: np.ndarray,
    ):
        FORAGER_SCORES_DIR.mkdir(parents=True, exist_ok=True)

        model_uuid = str(uuid.uuid4())
        scores_path = FORAGER_SCORES_DIR / model_uuid
        np.save(scores_path, scores)

        params = {"name": name, "paths": paths, "scores_path": scores_path}
        async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
            async with session.post(
                os.path.join(self.uri, IMPORT_SCORES_ENDPOINT, dataset_name),
                json=params,
            ) as response:
                j = await response.json()
                assert j["status"] == "success", j
