from jax.typing import ArrayLike
import bokeh as bk
from typing import NamedTuple
from bokeh.plotting import figure


class PlotSize(NamedTuple):
    width: int
    height: int


def plot_xys(
    x: ArrayLike,
    ys: list[ArrayLike],
    title: str = "",
    size: PlotSize = PlotSize(900, 300),
    labels: list[str] = None,
):
    # plot multiple series using bokeh
    plot = figure(title=title, width=size.width, height=size.height)
    colors = bk.palettes.Category20[20]
    for i, y in enumerate(ys):
        color = colors[i % len(colors)]
        if labels is None:
            label = f"color {i}"
        elif len(labels) != len(ys):
            raise ValueError(f"labels must be same length as ys")
        else:
            label = labels[i]
        plot.line(x, y, legend_label=label, line_width=2, color=color)
    plot.legend.location = "top_left"

    return plot
