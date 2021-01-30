import matplotlib.pyplot as plt
import numpy as np
import threading
import time


class Display2D:
    def __init__(self, metadata, value_range=None,
                 window_size=None, title=None, xlabel=None,
                 ylabel=None, cmap=None):
        """
        2D display constructor.

        :param value_range: range of values to display, (vmin, vmax)
        :param window_size: size of the window
        :param title: windo title
        :param xlabel: x label
        :param ylabel: y label
        :param cmap: color map to apply
        """
        self.metadata = metadata
        self.window_size = window_size
        self.value_range = value_range
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.cmap = cmap
        self.stop_event = threading.Event()
        self._prepare(metadata)

    def _prepare(self, metadata):
        self._fig, self._ax = plt.subplots()
        self._fig.canvas.mpl_connect("close_event", self.stop)
        if self.window_size is not None:
            self._fig.set_size_inches(self.window_size)
        if self.xlabel is not None:
            self._ax.set_xlabel(self.xlabel)
        if self.ylabel is not None:
            self._ax.set_ylabel(self.ylabel)
        if self.title is not None:
            self._fig.canvas.set_window_title(self.title)

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
        self._fig.show()

    def start(self, queue):
        while not self.stop_event.is_set():
            data = queue.get()
            self._img.set_data(data[100:, :])
            # self._ax.set_aspect("auto")
            self._fig.canvas.flush_events()
            plt.draw()

    def stop(self):
        self.stop_event.set()
