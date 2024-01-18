from typing import Optional, Tuple, List, Callable, Union
from jax.typing import ArrayLike

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D

AxesSubplot = Axes

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
mpl.rcParams['mathtext.it'] = 'Fira Sans:normalitalic'
mpl.rcParams['mathtext.bf'] = 'Fira Sans:bold'
mpl.rcParams['mathtext.tt'] = 'Fira Code:medium'

plt.rc('grid', linestyle="--", color='black', alpha=0.1)



def visualize_matrix(mat: ArrayLike,
                     dpi: int = 300) -> Tuple[Figure, AxesSubplot, AxesImage]:
    f, _ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    _ax.spines['left'].set_visible(False)
    _ax.spines['bottom'].set_visible(False)
    _ax.spines['right'].set_visible(False)
    _ax.spines['top'].set_visible(False)
    pt = _ax.matshow(mat, cmap="Greys")
    cax = f.colorbar(pt, shrink=0.8, drawedges=False)
    cax.outline.set_visible(False)
    return f, _ax, pt


def bounded_f_line(f: Callable[[float], float],
                   inv_f: Callable[[float], float],
                   _ax: AxesSubplot, **kwargs) -> Optional[Line2D]:
    e = 0.000001
    x_min, x_max = _ax.get_xbound()
    y_min, y_max = _ax.get_ybound()
    fx_min = f(x_min)
    fx_max = f(x_max)
    fy_min = inv_f(y_min)
    fy_max = inv_f(y_max)
    x_range = [x_min, x_max]
    y_range = [fx_min, fx_max]
    in_range = True
    if not (y_min - e <= fx_min <= y_max + e):
        if x_max + e >= fy_max >= fy_min >= x_min - e:
            x_range[0], y_range[0] = fy_min, y_min
        elif x_max + e >= fy_min >= fy_max >= x_min - e:
            x_range[0], y_range[0] = fy_max, y_max
        else:
            in_range = False
    if not (y_min - e <= fx_max <= y_max + e):
        if x_max + e >= fy_max >= fy_min >= x_min - e:
            x_range[1], y_range[1] = fy_max, y_max
        elif x_max + e >= fy_min >= fy_max >= x_min - e:
            x_range[1], y_range[1] = fy_min, y_min
        else:
            in_range = False

    if in_range:
        line = Line2D(x_range, y_range, **kwargs)
        _ax.add_line(line)
        return line
    else:
        return None


def bounded_line(p1: ArrayLike, p2: ArrayLike,
                 _ax: AxesSubplot,
                 **kwargs) -> Optional[Line2D]:
    if p2[0] == p1[0]:
        return _ax.axvline(x=p1[0])
    elif p2[1] == p1[1]:
        return _ax.axhline(y=p1[1])
    else:
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])

        def f(x):
            return p1[1] + k * (x - p1[0])

        def inv_f(y):
            return p1[0] + 1 / k * (y - p1[1])

        return bounded_f_line(f, inv_f, _ax, **kwargs)
