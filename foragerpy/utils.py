import os.path

from PIL import Image

RESIZE_MAX_HEIGHT = 200


def parse_gcs_path(path):
    assert path.startswith("gs://")
    path = path[len("gs://") :]
    bucket_end = path.find("/")
    bucket = path[:bucket_end]
    relative_path = path[bucket_end:].strip("/")
    return bucket, relative_path


def make_identifier(path):
    return os.path.splitext(os.path.basename(path))[0]


def resize_image(input_path, output_dir, max_height):
    image = Image.open(input_path)
    image = image.convert("RGB")
    image.thumbnail((image.width, max_height))
    image.save(output_dir / f"{make_identifier(input_path)}.jpg")
