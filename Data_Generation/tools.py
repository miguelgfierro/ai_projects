import requests
import matplotlib.pyplot as plt
from skimage import io


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


def plot_image(img):
    """Plot an image.
    
    Args:
        img (np.array): An image.
    
    **Examples**::
    
        >> img = io.imread('share/Lenna.png')
        >> plot_image(img)
    """
    io.imshow(img)
    plt.axis("off")
    plt.show()
    