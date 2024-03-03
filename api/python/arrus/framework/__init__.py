import arrus.metadata
import ctypes
import numpy as np
import arrus.core
import traceback
from arrus.framework.constant import Constant


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
        # All data wrapped as a single uint8 1D array.
        self.array = self._create_element_array(element_handle)
        self._numpy_array_wrappings = self._create_np_arrays(element_handle)

    @property
    def data(self):
        """Deprecated, use arrays[0]. """
        return self._numpy_array_wrappings[0]

    @property
    def arrays(self):
        return self._numpy_array_wrappings

    @property
    def size(self):
        return self._size

    def release(self):
        self._element_handle.release()

    def _create_element_array(self, element):
        ndarray = element.getData(0)
        addr = arrus.core.castUint8ToInt(ndarray.getUint8())
        ctypes_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_uint8))
        shape = (element.getSize(), )  # Number of bytes
        return np.ctypeslib.as_array(ctypes_ptr, shape=shape)

    def _create_np_arrays(self, element):
        arrays = []
        for i in range(element.getNumberOfArrays()):
            ndarray = element.getData(i)
            if ndarray.getDataType() != arrus.core.NdArrayDef.DataType_INT16:
                raise ValueError("Currently output data type int16 is supported only.")
            addr = arrus.core.castToInt(ndarray.getInt16())
            ctypes_ptr = ctypes.cast(addr, ctypes.POINTER(ctypes.c_int16))
            shape = arrus.utils.core.convert_from_tuple(ndarray.getShape())
            arr = np.ctypeslib.as_array(ctypes_ptr, shape=shape)
            arrays.append(arr)
        return arrays


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
