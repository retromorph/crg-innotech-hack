import requests
from PIL import Image


def load_image_from_url(url):
    response = requests.get(url, stream=True).raw
    return Image.open(response)
