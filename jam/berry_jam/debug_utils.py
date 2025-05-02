from calendar import c
from typing import Any
import jax
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

mpl.rcParams['savefig.transparent'] = True
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False

# mpl.rcParams["font.sans-serif"] = ["Fira Sans Regular", "Candara",
#                                    "Optima", "Arial"]
mpl.rcParams["font.sans-serif"] = ["Fira Sans"]
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.weight"] = "regular"
mpl.rcParams['axes.titleweight'] = "regular"
# mpl.rcParams['figure.titleweight'] = "medium"
mpl.rcParams["axes.labelweight"] = "regular"
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Fira Sans'
mpl.rcParams['mathtext.sf'] = 'Fira Sans'
mpl.rcParams['mathtext.cal'] = 'Fira Sans'
mpl.rcParams['mathtext.it'] = 'Fira Sans:italic'
mpl.rcParams['mathtext.bf'] = 'Fira Sans:bold'
mpl.rcParams['mathtext.tt'] = 'Fira Code:medium'

plt.rc('grid', linestyle="--", color='black', alpha=0.1)


def plot_image(image: np.ndarray, title: str = "Image", save_path: str = "./tmp.jpg") -> None:
    """
    Plot a single image with a title and optional save path.

    Args:
        image (np.ndarray): The image to plot.
        title (str): The title of the plot.
        save_path (str): Optional path to save the plot.
    """
    fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='viridis')
    ax.set_title(title)
    ax.axis('off')

    cbar = fig.colorbar(im, ax=ax)
    cbar_outline: Any = cbar.outline
    cbar_outline.set_edgecolor('gray')  
    cbar_outline.set_linewidth(0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()