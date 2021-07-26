import dataclasses

import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation
from collections.abc import Iterable


@dataclasses.dataclass(frozen=True)
class Layer2D:
    """

    :param clip_mode: whether to clip data before displaying it, available
      values: "off" - no clipping will be performed, "transparent" - set values
      outside given value_range to be transparent
    """
    metadata: object
    value_range: tuple
    cmap: str
    clip: str = "off"


class Display2D:
    """
    A very simple implementation of the 2D display.

    Currently, implemented using matplotlib FuncAnimation.

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
        :param extent: OX/OZ extent: a list of [ox_min, ox_max, oz_max, oz_min]
        """
        accepted_params = [{"metadata", "value_range", "cmap"}, {"layers"}]
        kwargs_params = set(kwargs.keys())
        actual_params_set = [s for s in accepted_params if s == kwargs_params]
        if len(actual_params_set) != 1:
            raise ValueError("Exactly one of the following parameter "
                             f"combinations should "
                             f"be provided: {accepted_params}")
        if kwargs_params == accepted_params[0]:
            layers = [Layer2D(**kwargs)]
        else:
            layers = kwargs["layers"]
        self.layers = layers
        self.window_size = window_size
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.input_timeout = input_timeout
        self.interval = interval
        self.extent = extent
        self.show_colorbar = show_colorbar
        self._prepare(self.layers)
        self._current_queue = None
        self._anim = None

    def _prepare(self, layers):
        self._fig, self._ax = plt.subplots()
        if self.window_size is not None:
            self._fig.set_size_inches(self.window_size)
        if self.xlabel is not None:
            self._ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            self._ax.set_ylabel(self.ylabel)
        if self.title is not None:
            self._fig.canvas.set_window_title(self.title)

        self.imgs = [None]*len(layers)
        self.clip_modes = [None]*len(layers)
        self.value_ranges = [None]*len(layers)
        clip_mode_map = {
            "off": 0,
            "transparent": 1
        }

        for i, layer in enumerate(self.layers):
            metadata = layer.metadata
            value_range = layer.value_range
            cmap = layer.cmap
            input_shape = metadata.input_shape
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
            img = self._ax.imshow(empty, cmap=cmap, vmin=vmin, vmax=vmax,
                                  extent=self.extent)
            self.imgs[i] = img
            self.clip_modes[i] = clip_mode_map[layer.clip]
            self.value_ranges[i] = value_range

        if self.show_colorbar:
            if len(layers) > 1:
                raise ValueError("Colorbar for display with multiple layers "
                                 "is currently not supported.")
            self._fig.colorbar(self.imgs[0])

    def start(self, queue):
        self._current_queue = queue
        self._anim = FuncAnimation(self._fig, self._update,
                                   interval=self.interval)
        plt.show()

    def _update(self, frame):
        datas = self._current_queue.get(timeout=self.input_timeout)
        for data, img, clip_mode, value_range in zip(datas, self.imgs, self.clip_modes, self.value_ranges):
            if clip_mode == 1:
                vmin, vmax = value_range
                data[data > vmax] = None
                data[data < vmin] = None
            img.set_data(data)
