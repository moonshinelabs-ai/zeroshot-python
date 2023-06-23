import hashlib
import os

import requests


def fetch_model(url: str, cache_location: str = "/tmp/zeroshot/cache"):
    """Fetch a model from either a URL or the cache if the file doesn't exist."""
    # We'll hash the file based on the URL
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
    file_path = os.path.join(cache_location, url_hash)

    # If the file isn't in the cache, download it
    if not os.path.exists(file_path):
        # Make sure the cache directory exists
        os.makedirs(cache_location, exist_ok=True)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Write the file to the cache
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

    return file_path
