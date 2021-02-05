import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.animation import FuncAnimation


class Display2D:
    """
    A very simple implementation of the 2D display.

    Currently, implemented using matplotlib FuncAnimation.

    """
    def __init__(self, metadata, value_range=None,
                 window_size=None, title=None, xlabel=None,
                 ylabel=None, cmap=None, interval=10,
                 input_timeout=2):
        """
        2D display constructor.

        :param value_range: range of values to display, (vmin, vmax)
        :param window_size: size of the window
        :param title: window title
        :param xlabel: x label
        :param ylabel: y label
        :param cmap: color map to apply
        :param interval: number of milliseconds between successive img updates
        """
        self.metadata = metadata
        self.window_size = window_size
        self.value_range = value_range
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.input_timeout = input_timeout
        self.interval = interval
        self._prepare(metadata)
        self._current_queue = None
        self._anim = None

    def _prepare(self, metadata):
        self._fig, self._ax = plt.subplots()
        if self.window_size is not None:
            self._fig.set_size_inches(self.window_size)
        if self.xlabel is not None:
            self._ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            self._ax.set_ylabel(self.ylabel)
        if self.title is not None:
            self._fig.canvas.set_window_title(self.title)

        # TODO compute extent based on sampling frequency and

        input_shape = metadata.input_shape
        datatype = metadata.dtype
        empty = np.zeros(input_shape, dtype=datatype)
        if self.value_range:
            vmin, vmax = self.value_range
            self._img = self._ax.imshow(empty, cmap=self.cmap,
                                        vmin=vmin, vmax=vmax)
        else:
            # determine min max based on min max value of the input data type
            if np.issubdtype(empty.dtype, np.floating):
                finfo = np.finfo(empty.dtype)
                vmin, vmax = finfo.min, finfo.max
            elif np.issubdtype(empty.dtype, np.integer):
                iinfo = np.iinfo(empty.dtype)
                vmin, vmax = iinfo.min, iinfo.max
            else:
                raise ValueError(f"Unsupported data type: {empty.dtype}")
            self._img = self._ax.imshow(empty, cmap=self.cmap,
                                        vmin=vmin, vmax=vmax)
            pass

    def start(self, queue):
        self._current_queue = queue
        self._anim = FuncAnimation(self._fig, self._update,
                                   interval=self.interval)
        plt.show()

    def _update(self, frame):
        data = self._current_queue.get(timeout=self.input_timeout)
        self._img.set_data(data)
