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

SERVER_URL = os.environ["SERVER_URL"]
GET_DATASET_ENDPOINT = "api/get_dataset_info_v2"
CREATE_DATASET_ENDPOINT = "api/create_dataset_v2"
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
        return "localhost:8000"

    async def add_dataset(
        self, name: str, train_images_directory: str, val_images_directory: str
    ):
        # Make sure that a dataset with this name doesn't already exist
        async with aiohttp.ClientSession(trust_env=self.use_proxy) as session:
            async with session.get(
                os.path.join(SERVER_URL, GET_DATASET_ENDPOINT, name)
            ) as response:
                assert response.status == 404, f"Dataset {name} already exists"

        index_id = str(uuid.uuid4())

        parent_dir = Path() / name

        train_dir = parent_dir / "train"
        val_dir = parent_dir / "val"
        thumbnails_dir = parent_dir / "thumbnails"
        index_dir = parent_dir / "index"
        aux_labels_dir = parent_dir / "aux_labels"
        for d in (train_dir, val_dir, thumbnails_dir, index_dir, aux_labels_dir):
            d.mkdir(parents=True, exist_ok=True)

        # Download train images
        download_start = time.time()
        if train_images_directory.startswith("gs://"):
            print("Downloading training images...")
            proc = await asyncio.create_subprocess_exec(
                "gsutil",
                "-m",
                "cp",
                "-r",
                "-n",
                os.path.join(train_images_directory, "*"),
                str(train_dir),
            )
            await proc.wait()
            train_paths = [
                p for e in IMAGE_EXTENSIONS for p in train_dir.glob(f"**/*.{e}")
            ]
        else:
            train_paths = [
                p
                for e in IMAGE_EXTENSIONS
                for p in train_images_directory.glob(f"**/*.{e}")
            ]

        # Download val images
        if val_images_directory.startswith("gs://"):
            print("Downloading validation images...")
            proc = await asyncio.create_subprocess_exec(
                "gsutil",
                "-m",
                "cp",
                "-r",
                "-n",
                os.path.join(val_images_directory, "*"),
                str(val_dir),
            )
            await proc.wait()
            val_paths = [p for e in IMAGE_EXTENSIONS for p in val_dir.glob(f"**/*.{e}")]
        else:
            val_paths = [
                p
                for e in IMAGE_EXTENSIONS
                for p in val_images_directory.glob(f"**/*.{e}")
            ]

        download_end = time.time()

        # Create identifier files
        _, train_gcs_relative_path = parse_gcs_path(train_gcs_path)
        _, val_gcs_relative_path = parse_gcs_path(val_gcs_path)

        train_labels = [
            os.path.join(train_gcs_relative_path, p.relative_to(train_dir))
            for p in train_paths
        ]
        val_labels = [
            os.path.join(val_gcs_relative_path, p.relative_to(val_dir))
            for p in val_paths
        ]

        labels = train_labels + val_labels
        json.dump(labels, Path(index_dir / "labels.json").open("w"))

        train_identifiers = {make_identifier(l): i for i, l in enumerate(train_labels)}
        json.dump(train_identifiers, Path(index_dir / "identifiers.json").open("w"))

        val_identifiers = {
            make_identifier(l): i + len(train_identifiers)
            for i, l in enumerate(val_labels)
        }
        json.dump(val_identifiers, Path(index_dir / "val_identifiers.json").open("w"))

        # Create embeddings
        res4_path = index_dir / "local" / "imagenet_early"
        res5_path = index_dir / "local" / "imagenet"
        linear_path = index_dir / "local" / "imagenet_linear"

        res4_full_path = index_dir / "local" / "imagenet_full_early"
        res5_full_path = index_dir / "local" / "imagenet_full"

        clip_path = index_dir / "local" / "clip"
        for d in (
            res4_path,
            res5_path,
            clip_path,
            linear_path,
            res4_full_path,
            res5_full_path,
        ):
            d.mkdir(parents=True, exist_ok=True)

        image_paths = train_paths + val_paths

        # Create thumbnails
        thumbnail_start = time.time()
        print("Creating thumbnails...")
        for path in tqdm(image_paths):
            resize_image(path, thumbnails_dir, RESIZE_MAX_HEIGHT)
        thumbnail_end = time.time()

        # Upload index to Cloud Storage
        upload_start = time.time()
        if True:
            proc = await asyncio.create_subprocess_exec(
                "gsutil",
                "-m",
                "cp",
                "-r",
                str(index_dir),
                os.path.join(INDEX_UPLOAD_GCS_PATH, index_id),
            )
            await proc.wait()

            proc = await asyncio.create_subprocess_exec(
                "gsutil",
                "-m",
                "cp",
                "-r",
                str(aux_labels_dir),
                os.path.join(AUX_LABELS_UPLOAD_GCS_PATH, index_id),
            )
            await proc.wait()

            # Upload thumbnails to Cloud Storage
            proc = await asyncio.create_subprocess_exec(
                "gsutil",
                "-m",
                "cp",
                "-r",
                str(thumbnails_dir),
                os.path.join(THUMBNAIL_UPLOAD_GCS_PATH, index_id),
            )
            await proc.wait()
        upload_end = time.time()

        # Add to database
        params = {
            "dataset": name,
            "train_path": train_gcs_path,
            "val_path": val_gcs_path,
            "index_id": index_id,
        }
        add_db_start = time.time()
        if True:
            async with aiohttp.ClientSession(trust_env=use_proxy) as session:
                async with session.post(
                    os.path.join(SERVER_URL, CREATE_DATASET_ENDPOINT), json=params
                ) as response:
                    j = await response.json()
                    assert j["status"] == "success", j
        add_db_end = time.time()

        print("Timing")
        for k, t in [
            ("Download", download_end - download_start),
            ("Resnet", resnet_end - resnet_start),
            ("Clip", clip_end - clip_start),
            ("Thumbnail", thumbnail_end - thumbnail_start),
            ("Upload", upload_end - upload_start),
            ("Add db", add_db_end - add_db_start),
        ]:
            print("{:15} {}".format(k, str(timedelta(seconds=t))))

    async def delete_dataset(self, name: str):
        pass

    async def import_labels(self, dataset_name: str):
        pass

    async def export_labels(self, dataset_name: str):
        pass
