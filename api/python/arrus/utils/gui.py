import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import time
from typing import List
from matplotlib.animation import FuncAnimation
from collections.abc import Iterable


@dataclasses.dataclass(frozen=True)
class Layer2D:
    """
    :param input: which input array should be used as an input to the layer
      (ordinal number), if None (default) the ordinal number of the layer in the
      input list of layers will be used.
    """
    metadata: object
    value_range: tuple
    cmap: str
    input: int = None
    value_func: object = None


@dataclasses.dataclass(frozen=True)
class View2D:
    layers: List[Layer2D]
    xlabel: str = None
    ylabel: str = None
    extent: tuple = None


class Display2D:
    """
    A very simple implementation of the 2D display.

    The 2D Display is intended to be used.

    Currently, implemented using matplotlib package.
    """

    def __init__(self, window_size=None, title=None, xlabel=None,
                 ylabel=None, interval=10, input_timeout=2, extent=None,
                 show_colorbar=False, **kwargs):
        """
        2D display constructor.

        :param value_range: range of values to display, (vmin, vmax)
        :param window_size: size of the window
        :param title: window title
        :param xlabel: x label
        :param ylabel: y label
        :param cmap: color map to apply
        :param interval: number of milliseconds between successive img updates
        :param extent: OX/OZ extent: a tuple of (ox_min, ox_max, oz_max, oz_min)
        """
        accepted_params = [
            {"metadata", "value_range", "cmap"},
            {"layers"},
            {"views"}
        ]
        kwargs_params = set(kwargs.keys())
        actual_params_set = [s for s in accepted_params if s == kwargs_params]
        if len(actual_params_set) != 1:
            raise ValueError("Exactly one of the following parameter "
                             "combinations should be provided: "
                             f"{accepted_params}")

        if kwargs_params == accepted_params[0]:
            views = [View2D(layers=[Layer2D(**kwargs)],
                           xlabel=xlabel, ylabel=ylabel,
                           extent=extent)]
        elif kwargs_params == accepted_params[1]:
            views = [View2D(layers=kwargs["layers"],
                           xlabel=xlabel, ylabel=ylabel,
                           extent=extent)]
        else:
            views = kwargs["views"]
        self.views = views
        self.window_size = window_size
        self.title = title
        self.input_timeout = input_timeout
        self.interval = interval
        self.show_colorbar = show_colorbar
        self._prepare(self.views)
        self._current_queue = None
        self._anim = None

    def _prepare(self, views):
        self._fig, self._axes = plt.subplots(1, len(views))
        if len(views) == 1:
            self._axes = [self._axes]
        if self.window_size is not None:
            self._fig.set_size_inches(self.window_size)

        self.all_canvases = []
        self.all_layers = []

        for view_id, view in enumerate(self.views):
            if view.xlabel is not None:
                self._axes[view_id].set_xlabel(view.xlabel)
            if view.ylabel is not None:
                self._axes[view_id].set_ylabel(view.ylabel)
            layers = view.layers
            for i, layer in enumerate(layers):
                metadata = layer.metadata
                value_range = layer.value_range
                cmap = layer.cmap
                input_shape = metadata.input_shape[-2:]
                datatype = metadata.dtype
                empty = np.zeros(input_shape, dtype=datatype)
                if value_range:
                    vmin, vmax = value_range
                else:
                    # determine min max based on min max value of the input dtype
                    if np.issubdtype(empty.dtype, np.floating):
                        finfo = np.finfo(empty.dtype)
                        vmin, vmax = finfo.min, finfo.max
                    elif np.issubdtype(empty.dtype, np.integer):
                        iinfo = np.iinfo(empty.dtype)
                        vmin, vmax = iinfo.min, iinfo.max
                    else:
                        raise ValueError(f"Unsupported data type: {empty.dtype}")
                img = self._axes[view_id].imshow(empty, cmap=cmap, vmin=vmin, vmax=vmax,
                                                 extent=view.extent)
                self.all_canvases.append(img)
                if layer.input is None:
                    layer = dataclasses.replace(layer, input=i)
                self.all_layers.append(layer)

        if self.show_colorbar:
            if len(self.all_layers) > 1:
                raise ValueError("Colorbar for display with multiple layers "
                                 "is currently not supported.")
            self._fig.colorbar(self.all_canvases[0])

    def start(self, queue):
        self._current_queue = queue
        self._anim = FuncAnimation(self._fig, self._update,
                                   interval=self.interval)
        plt.show()

    def _update(self, frame):
        datas = self._current_queue.get(timeout=self.input_timeout)
        for c, l in zip(self.all_canvases, self.all_layers):
            data = datas[l.input]
            if l.value_func is not None:
                data = l.value_func(data)
            c.set_data(data)
