import requests
import matplotlib.pyplot as plt

# from PIL import Image
from skimage import io


# def read_image_from_url(url):
#     img = Image.open(requests.get(url, stream=True).raw)
#     return img


def read_image_url(url):
    """Read an image from a URL using skimage.
    
    Args:
        url (str): URL of the file.
    
    Returns:
        np.array: An image.
    
    Examples:
        >>> img = read_image_url('https://raw.githubusercontent.com/miguelgfierro/codebase/master/share/Lenna.png')
        >>> img.shape
        (512, 512, 3)
    """
    return io.imread(url)


# def plot_image(img):
#     plt.imshow(img)
#     plt.axis("off")
#     plt.show()


def plot_image(img):
    """Plot an image.
    
    Args:
        img (np.array): An image.
    
    **Examples**::
    
        >> img = io.imread('share/Lenna.png')
        >> plot_image(img)
    """
    io.imshow(img)
    io.show()
    