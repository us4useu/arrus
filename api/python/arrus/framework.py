import arrus.metadata
import ctypes
import numpy as np
import arrus.core
import traceback


class OnNewDataCallback(arrus.core.OnNewDataCallbackWrapper):

    def __init__(self, callback_fn):
        super(OnNewDataCallback, self).__init__()
        self._callback_fn = callback_fn

    def run(self, element):
        try:
            self._callback_fn(element)
        except Exception as e:
            print(f"Exception in the callback function")
            traceback.print_exc()
        except:
            print("Unknown exception in the callback function.")
            traceback.print_exc()


class DataBuffer:

    def __init__(self, buffer_handle):
        self._buffer_handle = buffer_handle
        self._user_defined_callback = None

    def register_on_new_data_callback(self, callback):
        self._user_defined_callback = callback
        self._callback_wrapper = OnNewDataCallback(self._user_defined_callback)
        arrus.core.registerOnNewDataCallbackWrapper(self._buffer_handle, self._callback_wrapper)

    def _callback(self, element):
        # TODO wrap into numpy array
        self._user_defined_callback()


class LegacyBuffer:
    """
    Buffer storing data that comes from the us4r device.

    The buffer is implemented as a circular queue. The consumer gets data from
    the queue's tails (end), the producer puts new data at the queue's head
    (firt element of the queue).

    This class provides an access to the queue's tail only. The user
    can access the latest data produced by the device by accessing `tail()`
    function. To release the tail data that is not needed anymore the user
    can call `release_tail()` function.
    """

    def __init__(self, buffer_handle,
                 fac: arrus.metadata.FrameAcquisitionContext,
                 data_description: arrus.metadata.EchoDataDescription,
                 frame_shape: tuple,
                 rx_batch_size: int):
        self.buffer_handle = buffer_handle
        self.data_description = data_description
        self.frame_shape = frame_shape
        self.buffer_cache = {}
        self.frame_metadata_cache = {}
        # Required to determine time step between frame metadata positions.
        self.n_samples = fac.raw_sequence.get_n_samples()
        if len(self.n_samples) > 1:
            raise RuntimeError
        self.n_samples = next(iter(self.n_samples))
        # FIXME This won't work when the the rx aperture has to be splitted to multiple operations
        # Currently works for rx aperture <= 64 elements
        self.n_triggers = self.data_description.custom["frame_channel_mapping"].frames.shape[0]
        self.rx_batch_size = rx_batch_size

    def register_on_new_data_callback(self):
        pass

    def tail(self, timeout=None):
        """
        Returns data available at the tail of the buffer.

        :param timeout: timeout in milliseconds, None means infinite timeout
        :return: a pair: RF data, metadata
        """
        data_addr = self.buffer_handle.tailAddress(
            -1 if timeout is None else timeout)
        if data_addr not in self.buffer_cache:
            array = self._create_array(data_addr)
            frame_metadata_view = array[:self.n_samples*self.n_triggers*self.rx_batch_size:self.n_samples]
            self.buffer_cache[data_addr] = array
            self.frame_metadata_cache[data_addr] = frame_metadata_view
        else:
            array = self.buffer_cache[data_addr]
            frame_metadata_view = self.frame_metadata_cache[data_addr]
        return array

    def release_tail(self, timeout=None):
        """
        Marks the tail data as no longer needed.

        :param timeout: timeout in milliseconds, None means infinite timeout
        """
        self.buffer_handle.releaseTail(-1 if timeout is None else timeout)

    def _create_array(self, addr):
        ctypes_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int16))
        arr = np.ctypeslib.as_array(ctypes_ptr, shape=self.frame_shape)
        return arr

    def get_n_elements(self):
        return self.buffer_handle.getNumberOfElements()

    def get_element(self, i):
        return self.buffer_handle.getElementAddress(i)

    def get_element_size(self):
        return self.buffer_handle.getElementSize()