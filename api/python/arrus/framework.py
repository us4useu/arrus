import arrus.metadata
import ctypes
import numpy as np
import arrus.core
import traceback


class OnNewDataCallback(arrus.core.OnNewDataCallbackWrapper):

    def __init__(self, callback_fn):
        super().__init__()
        self._callback_fn = callback_fn

    def run(self, element):
        try:
            self._callback_fn(element)
        except Exception as e:
            print(e)
            traceback.print_exc()


class OnBufferOverflowCallback(arrus.core.OnBufferOverflowCallbackWrapper):

    def __init__(self, callback_fn):
        super().__init__()
        self._callback_fn = callback_fn

    def run(self):
        try:
            self._callback_fn()
        except Exception as e:
            print(e)
            traceback.print_exc()


class DataBufferElement:
    """
    Data buffer element. Allows to access the space of the acquired data.
    """

    def __init__(self, element_handle):
        self._element_handle = element_handle
        self._size = element_handle.getSize()
        self._numpy_array_wrapping = self._create_np_array(element_handle)

    @property
    def data(self):
        """
        The data wrapped into a numpy array.
        """
        return self._numpy_array_wrapping

    @property
    def size(self):
        return self._size

    def release(self):
        self._element_handle.release()

    def _create_np_array(self, element):
        ndarray = element.getData()
        if ndarray.getDataType() != arrus.core.NdArray.DataType_INT16:
            raise ValueError("Currently output data type int16 is supported only.")
        addr = arrus.core.castToInt(ndarray.getInt16())
        ctypes_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int16))
        shape = arrus.utils.core.convert_from_tuple(ndarray.getShape())
        arr = np.ctypeslib.as_array(ctypes_ptr, shape=shape)
        return arr


class DataBuffer:
    """
    The output data buffer.

    This buffer allows to register a callback, that should called when new
    data arrives.

    The buffer elements are automatically released after running all available callbacks.
    """
    def __init__(self, buffer_handle):
        self._buffer_handle = buffer_handle
        self._callbacks = []
        self._register_internal_callback()
        self._on_buffer_overflow_callbacks = []
        self._register_internal_buffer_overflow_callback()
        self.elements = self._wrap_elements()
        self.n_elements = len(self.elements)

    def append_on_new_data_callback(self, callback):
        """
        Append to the list of callbacks that should be run when new data
        arrives.

        Note: the callback function should explicitly release buffer element
        using

        :param callback: a callback function, should take one parameter -- DataBufferElement instance
        """
        self._callbacks.append(callback)

    def append_on_buffer_overflow_callback(self, callback):
        """
        Register callback that will be called when buffer overflow occurs.

        :param callback: callback function to register
        """
        self._on_buffer_overflow_callbacks.append(callback)

    def _register_internal_callback(self):
        self._callback_wrapper = OnNewDataCallback(self._callback)
        arrus.core.registerOnNewDataCallbackFifoLockFreeBuffer(
            self._buffer_handle, self._callback_wrapper)

    def _callback(self, element):
        pos = element.getPosition()
        py_element = self.elements[pos]
        for cbk in self._callbacks:
            cbk(py_element)

    def _on_buffer_overflow_callback(self):
        for cbk in self._on_buffer_overflow_callbacks:
            cbk()

    def _register_internal_buffer_overflow_callback(self):
        self._overflow_callback_wrapper = OnBufferOverflowCallback(
            self._on_buffer_overflow_callback)
        arrus.core.registerOnBufferOverflowCallback(
            self._buffer_handle, self._overflow_callback_wrapper)

    def _wrap_elements(self):
        return [DataBufferElement(self._buffer_handle.getElement(i))
                for i in range(self._buffer_handle.getNumberOfElements())]


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

    def get_n_elements(self):
        return self.buffer_handle.getNumberOfElements()

    def get_element(self, i):
        return self.buffer_handle.getElementAddress(i)

    def get_element_size(self):
        return self.buffer_handle.getElementSize()
