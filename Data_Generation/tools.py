import requests
import matplotlib.pyplot as plt
from PIL import Image


def read_image_from_url(url):
    img = Image.open(requests.get(url, stream=True).raw)
    return img


def plot_image(img):
    plt.imshow(img)
    plt.axis("off")
    plt.show()
