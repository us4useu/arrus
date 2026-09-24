import abc
import queue
import copy
import sys
import re

import numpy as np
import importlib
import importlib.util
import dataclasses
from collections import defaultdict

import arrus.core
import arrus.exceptions
import arrus.devices.us4r
import arrus.devices.file
import arrus.medium
import arrus.metadata
import arrus.params
import arrus.devices.cpu
import arrus.devices.device
import arrus.devices.gpu
import arrus.ops.us4r
import arrus.ops.imaging
import arrus.ops.tgc
import arrus.kernels.tgc
import arrus.kernels.kernel
import arrus.utils
import arrus.utils.core
import arrus.framework
from typing import Sequence, Dict, Iterable, List, Optional
from numbers import Number

from arrus.devices.ultrasound import Ultrasound
from arrus.devices.us4r import Us4R


class AbstractSession(abc.ABC):
    """
    An abstract class of session.

    This class is not intended to be instantiated.
    """

    def __init__(self):
        pass

    @abc.abstractmethod
    def get_device(self, path: str):
        """
        Returns a device located at given path.

        :param path: a path to a device, for example '/Us4R:0'
        :return: a device located in a given path.
        """
        raise ValueError("Tried to access an abstract method.")


@dataclasses.dataclass(frozen=True)
class SessionContext:
    medium: arrus.medium.Medium


class _DictInterruptListener(arrus.core.Us4OEMInterruptListener):
    """
    Internal Us4OEMInterruptListener that dispatches every fired interrupt
    to a user-supplied callback looked up in a Us4OEMInterrupt -> callable map.
    """

    def __init__(self, callbacks):
        super().__init__()
        self._callbacks = dict(callbacks)

    def on_any_interrupt(self, interrupt, oem):
        cb = self._callbacks.get(interrupt)
        if cb is not None:
            cb(oem)


_SUPPORTED_PARAM_KEYS = frozenset({"us4r:0/system_callbacks"})


def create_session_settings_from(filepath, params=None):
    """
    Loads SessionSettings from a prototxt configuration file and (optionally)
    attaches per-device parameters that are not encoded in the prototxt.

    The ``params`` argument is a ``dict`` from string keys to arbitrary
    Python values. Currently a single key is supported:

    - ``"us4r:0/system_callbacks"``: a mapping from
      :class:`arrus.devices.us4oem.Us4OEMInterrupt` to a callable
      ``f(oem_ordinal)`` invoked when that interrupt fires on the
      corresponding us4OEM. Interrupts not present in the mapping are ignored.

    Example::

        from arrus.devices.us4oem import Us4OEMInterrupt

        def on_watchdog0(oem):
            print(f"WATCHDOG_IRQ0 on OEM {oem}")

        settings = arrus.create_session_settings_from(
            "us4r.prototxt",
            params={
                "us4r:0/system_callbacks": {
                    Us4OEMInterrupt.WATCHDOG_IRQ0: on_watchdog0,
                },
            },
        )
        sess = arrus.Session(session_settings=settings)

    :param filepath: path to the prototxt configuration file.
    :param params: optional mapping from string parameter keys to their values.
        Unknown keys raise ``ValueError``.
    :return: an opaque SessionSettings handle that can be passed as
        ``session_settings`` to :class:`Session`. The returned object also
        owns the underlying interrupt listener, so it must remain alive for
        as long as the Session built from it.
    """
    if params is None:
        params = {}
    unknown = set(params.keys()) - _SUPPORTED_PARAM_KEYS
    if unknown:
        raise ValueError(
            f"Unsupported keys in params: {sorted(unknown)}. "
            f"Supported keys: {sorted(_SUPPORTED_PARAM_KEYS)}.")

    system_callbacks = params.get("us4r:0/system_callbacks", {})
    listener = _DictInterruptListener(system_callbacks)
    settings = arrus.core.createSessionSettingsFrom(filepath, listener)
    # The C++ adapter holds a raw pointer to the listener; tie its lifetime
    # to the settings handle so callers don't need to keep it themselves.
    settings._arrus_interrupt_listener = listener
    return settings


class Session(AbstractSession):
    """
    A communication session with the ultrasound system.

    Currently, only localhost session is available.

    This class is a context manager. All the processing to be done on the device
    on the devices should be done withing the session context.
    """

    def __init__(self, cfg_path: str = None,
                 medium: arrus.medium.Medium = None,
                 session_settings=None):
        """
        Session constructor.

        :param cfg_path: a path to configuration file. Ignored when
            ``session_settings`` is provided. If neither ``cfg_path`` nor
            ``session_settings`` is given, ``"us4r.prototxt"`` is used.
        :param medium: medium description to set in context
        :param session_settings: an opaque SessionSettings handle obtained from
            ``arrus.core.createSessionSettingsFrom(cfg_path, listener)``. When
            given, the session is built from this handle (e.g. with a system
            interrupt listener attached) instead of loading ``cfg_path`` here.
        """
        super().__init__()
        import arrus.logging
        if session_settings is not None:
            if cfg_path is not None:
                raise ValueError(
                    "Provide either cfg_path or session_settings, not both.")
            self._session_handle = arrus.core.createSessionSharedHandleFromSettings(
                session_settings)
            # Keep the settings handle alive for the lifetime of the session
            # so that any owned objects (e.g. an interrupt listener attached
            # by create_session_settings_from) outlive the C++ Session.
            self._session_settings = session_settings
        else:
            if cfg_path is None:
                cfg_path = "us4r.prototxt"
            self._session_handle = arrus.core.createSessionSharedHandle(cfg_path)
            self._session_settings = None
        self._context = SessionContext(medium=medium)
        self._py_devices = self._create_py_devices()
        self._current_processing = None
        # The processing object provided by the user (see the _set_processing method).
        self._current_processing_spec = None
        # The sub-sequences prepared with prepare_subsequences, not triggered yet: (metadata, processing).
        self._prepared_subsequences = None
        # The processing updates that should be applied, before the given element is triggered:
        # [(element number, metadata, processing), ...] (see prepare_subsequences).
        self._pending_processing_updates = []
        # The number of the triggered elements (MANUAL mode), since the start of the scheme.
        self._n_runs = 0
        self._is_started = False
        self._current_scheme = None
        # Current metadata (for the full sequence)
        self.metadatas = None
        arrus.logging.log(arrus.logging.DEBUG, f"ARRUS Python API. Python version: {sys.version}")

    def upload(self, scheme: arrus.ops.us4r.Scheme):
        """
        Uploads a given sequence on devices.

        :param scheme: scheme to upload
        :raises: ValueError when some of the input parameters are invalid
        :return: a data buffer and constant metadata
        """
        # Verify the input parameters.
        # Prepare sequence to load
        us_device: Ultrasound = self.get_device("/Ultrasound:0")
        us_device_dto = us_device.get_dto()
        self._prepared_subsequences = None
        self._pending_processing_updates = []
        medium = self._context.medium
        sequences = scheme.tx_rx_sequence
        if not isinstance(sequences, Iterable):
            sequences = (sequences, )
        processing = scheme.processing
        constants = scheme.constants

        raw_seqs = []
        tx_delay_constants = []
        # TODO make sure all sequences have the same TGC (different TGCs are not supported)
        # Convert to raw sequences and upload.
        sequences = [dataclasses.replace(s, name=f"TxRxSequence:{i}")
                     if s.name is None else s
                     for i, s in enumerate(sequences)]
        constants_by_sequence_name = self._group_constants_by_sequence_name(sequences, constants)
        for i, sequence in enumerate(sequences):
            kernel_context = self._create_kernel_context(
                sequence,
                us_device_dto,
                medium,
                scheme.digital_down_conversion,
                constants_by_sequence_name.get(sequence.name, [])
            )
            conversion_results = arrus.kernels.get_kernel(type(sequence))(kernel_context)
            raw_seq = conversion_results.sequence
            raw_seqs.append(raw_seq)
            tx_delay_constants.extend(conversion_results.constants)

        actual_scheme = dataclasses.replace(
            scheme,
            tx_rx_sequence=raw_seqs,
            constants=tx_delay_constants
        )
        core_scheme = arrus.utils.core.convert_to_core_scheme(actual_scheme)
        upload_result = self._session_handle.upload(core_scheme)
        self._current_scheme = actual_scheme
        # Update the DTO with the new data sampling frequency (determined by the scheme).
        us_device_dto = dataclasses.replace(
            us_device_dto,
            data_sampling_frequency=us_device.current_sampling_frequency
        )
        # Output buffer
        buffer_handle = arrus.core.getFifoLockFreeBuffer(upload_result)
        self.buffer = arrus.framework.DataBuffer(buffer_handle)

        # Constant metadata
        # NOTE: the below should be called after session_handle.upload()
        us_device.set_tgc_and_context(sequences, self.medium)
        self.metadatas = []

        for i, (raw_seq, seq) in enumerate(zip(raw_seqs, sequences)):
            data_description = us_device.get_data_description(upload_result, raw_seq, array_id=i)
            # -- Constant metadata
            # --- Frame acquisition context
            fac = self._create_frame_acquisition_context(
                seq, raw_seq, us_device_dto, medium, tx_delay_constants)
            input_shape = self.buffer.elements[0].arrays[i].shape
            is_iq_data = scheme.digital_down_conversion is not None
            const_metadata = arrus.metadata.ConstMetadata(
                context=fac, data_desc=data_description,
                input_shape=input_shape, is_iq_data=is_iq_data, dtype="int16",
                version=arrus.__version__
            )
            self.metadatas.append(const_metadata)

        # numpy/cupy processing initialization
        return  self._set_processing(self.buffer, self.metadatas, processing, sequences)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_scheme()
        self.close()

    def start_scheme(self):
        """
        Starts the execution of the uploaded scheme.
        """
        # The prepared sub-sequences (if any) are used right from the start.
        self._apply_all_processing_updates()
        arrus.core.arrusSessionStartScheme(self._session_handle)
        self._is_started = True
        self._n_runs = 1

    def stop_scheme(self):
        """
        Stops execution of the scheme.
        """
        arrus.core.arrusSessionStopScheme(self._session_handle)
        self._is_started = False

    def run(self, sync: bool=False, timeout: int=None):
        """
        Runs the uploaded scheme.

        The behaviour of this method depends on the work mode:
        - MANUAL: triggers execution of batch of sequences only ONCE,
        - MANUAL_OP: triggers execution of a single TX/RX only ONCE,
        - HOST, ASYNC: triggers execution of batch of sequences IN A LOOP (Host: trigger is on buffer element release).
          The run function can be called only once (before the scheme is stopped).


        :param sync: whether this method should work in a synchronous or asynchronous; true means synchronous, i.e.
                     the caller will wait until the triggered TX/RX or sequence of TX/RXs has been done. This parameter only
                     matters when the work mode is set to MANUAL or MANUAL_OP. NOTE: For the US4R device, this method ONLY waits
                     for the completion of the TX/RX sequence. Currently, it DOES NOT WAIT for the data transfer to the host PC
                     or for the processing to finish — to wait for these two events, either wait for the final data using
                     buffer.get() / register your own callback function.
        :param timeout: timeout [ms]; std::nullopt means to wait infinitely. This parameter is only relevant when
                        sync = true; the value of this parameter only matters when work mode is set to MANUAL or MANUAL_OP.
        """
        if not self._is_started:
            # Starts the scheme: the prepared sub-sequences (if any) are used right from the start.
            self._apply_all_processing_updates()
            arrus.core.arrusSessionRun(self._session_handle, sync, timeout)
            self._is_started = True
            self._n_runs = 1
            return
        # The number of the element triggered now.
        element = self._n_runs
        # The data of this element are acquired with the sub-sequences prepared before the previous run.
        self._apply_processing_updates(until_element=element)
        prepared = self._prepared_subsequences
        if prepared is not None:
            # This run switches the us4R to the prepared sub-sequences; the element triggered now is still acquired
            # with the current ones, the next element is the first one acquired with the prepared ones.
            self._pending_processing_updates.append((element + 1, ) + tuple(prepared))
            self._prepared_subsequences = None
        try:
            arrus.core.arrusSessionRun(self._session_handle, sync, timeout)
        except Exception:
            if prepared is not None:
                # Not switched (e.g. the device was not waiting for the trigger).
                self._pending_processing_updates.pop()
                self._prepared_subsequences = prepared
            raise
        self._n_runs += 1

    def close(self):
        """
        Closes session.

        This method disconnects with all the devices available during this session.
        Sets the state of the session to closed, any subsequent call to the object
        methods (e.g. upload, startScheme..) will result in exception.
        """
        self.stop_scheme()
        if self._current_processing is not None:
            self._current_processing.close()
        arrus.core.arrusSessionClose(self._session_handle)

    def get_device(self, path: str):
        """
        Returns a device identified by a given id.

        The available devices are determined by the initial session settings.

        The handle to device is invalid after the session is closed
        (i.e. the session object is disposed).

        :param path: a path to the device
        :return: a handle to device
        """
        device_handle = self._session_handle.getDevice(path)

        device_id = device_handle.getDeviceId()
        device_type = device_id.getDeviceType()
        device_ordinal = device_id.getOrdinal()

        py_id = (device_type, device_ordinal)
        if py_id in self._py_devices:
            return self._py_devices[py_id]


        # Cast device to its type class.
        specific_device_cast = {
            arrus.core.DeviceType_Us4R:
                lambda handle: arrus.devices.us4r.Us4R(
                    arrus.core.castToUs4r(handle)),
            arrus.core.DeviceType_File:
                lambda handle: arrus.devices.file.File(
                    arrus.core.castToFile(handle)),
            arrus.core.DeviceType_GPU:
                lambda handle: arrus.devices.gpu.Gpu(
                    arrus.core.castToGpu(handle))
        }.get(device_type, None)
        if specific_device_cast is None:
            raise arrus.exceptions.DeviceNotFoundError(path)
        specific_device = specific_device_cast(device_handle)
        self._py_devices[py_id] = specific_device
        return specific_device

    def set_parameters(self, params):
        if self._contains_py_params(params):
            self._handle_py_params(params)
            params = self._remove_py_params(params)
        core_params = arrus.utils.core.convert_to_core_parameters(params)
        self._session_handle.setParameters(core_params)

    def set_parameter(self, key: str, value: Sequence[Number]):
        """
        Sets the value for parameter with the given name.
        TODO: note: this method currently is not thread-safe
        """
        if self._current_processing is not None:
            return self._current_processing.set_parameter(key, value)

    def get_parameter(self, key: str) -> Sequence[Number]:
        """
        Returns the current value for parameter with the given name.
        """
        if self._current_processing is not None:
            return self._current_processing.processing.get_parameter(key)

    def get_parameters(self) -> Dict[str, arrus.params.ParameterDef]:
        if self._current_processing is not None:
            return self._current_processing.get_parameters()

    def get_session_context(self):
        return self._context

    @property
    def medium(self):
        """
        Returns currently set Medium.
        NOTE: this method is not thread-safe!
        """
        return self._context.medium

    @medium.setter
    def medium(self, value):
        """
        Sets a new medium in the current session context.
        NOTE: this method is not thread-safe!
        """
        self._context = SessionContext(medium=value)

    def set_subsequences(self, subsequences, processing=None, sris: List[Optional[float]] = None):
        """
        Selects the TX/RXs to be executed, for each of the uploaded TX/RX sequences.

        The `subsequences` array should have exactly n elements, where n is the number of currently uploaded
        sequences. The element subsequences[i] determines the TX/RXs of the i-th sequence to run, and can be:

        - a slice, e.g. `slice(2, 8)`: the [start, end) range of the TX/RXs,
        - a list of the TX/RX ordinal numbers, e.g. `[2, 3, 5, 8, 13]`: exactly these TX/RXs will be executed,
          in that order. The numbers should be provided in the increasing order, without repetitions.

        As a shortcut, when a single TX/RX sequence is uploaded, the list of the TX/RX numbers can be provided
        directly, e.g. `session.set_subsequences([2, 3, 5, 8, 13])`.

        NOTE: in the Python API we operate on the logical TX/RXs; a single logical TX/RX can be translated to
        more than one physical TX/RX (e.g. when the RX aperture is larger than the number of the RX channels
        of a single us4OEM). Selecting non-consecutive TX/RXs is supported by the us4OEM+ devices only.

        The `sris` should have exactly n elements, or should be empty (which means that no additional sri should be
        applied).

        To turn off the given sequence, just provide an empty list of TX/RXs (or e.g. `slice(0, 0)`) for it.
        For such sequences, the metadata will describe only empty data.

        :param subsequences: the TX/RXs to run, for each Scheme sub-sequence
        :param sris: sris to apply to each Scheme sub-sequence
        :return returns: the buffer and metadata for the modified Scheme. The metadata array size is always equal to
           the number of sequences in the original Scheme
        """
        sris = [] if sris is None else sris
        subsequences = self._convert_to_subsequences(subsequences)

        arrus_ops = arrus.utils.core.convert_to_arrus_subsequences(subsequences)
        arrus_sris = arrus.utils.core.convert_to_optional_vector(sris)

        # Any previously prepared sub-sequences are overwritten.
        self._prepared_subsequences = None
        self._pending_processing_updates = []
        upload_result = self._session_handle.setSubsequences(arrus_ops, arrus_sris)

        buffer_handle = arrus.core.getFifoLockFreeBuffer(upload_result)
        self.buffer = arrus.framework.DataBuffer(buffer_handle)
        result_metadatas = self._create_subsequence_metadata(subsequences, upload_result)
        return self._set_processing(self.buffer, result_metadatas, processing, [])

    def prepare_subsequences(self, subsequences, processing=None, sris: List[Optional[float]] = None):
        """
        Prepares the TX/RXs to be executed, without stopping the scheme.

        The parameters are the same as for `set_subsequences`. In contrast to `set_subsequences`, the scheme
        can be running: the new sub-sequences are programmed in the part of the us4R sequencer memory, that is
        currently not in use (sequencer double-buffering), i.e. the next sub-sequence can be prepared while the
        current one is acquired and processed.

        NOTE: when the scheme is running, the new sub-sequences are used starting from the SECOND `run` after this
        call: the next `run` still acquires the data with the current sub-sequences (while waiting for the trigger,
        the us4R sequencer has already moved to the first TX/RX of the next acquisition). The processing is updated
        accordingly (right before the second `run`). When the scheme is started (start_scheme, or `run` of the
        stopped scheme), the prepared sub-sequences are used right from the start.

        Requirements (when the scheme is running):

        - the MANUAL work mode,
        - the new sub-sequences must produce the data of exactly the same shape as the current ones (e.g. the same
          number of TX/RXs), so that the output buffer and the processing can be reused,
        - the output buffer should have the same number of elements as the RX buffer,
        - `processing` should be None or the same object as the currently used one,
        - the next `run` should be called after the data of the previous `run` arrived (otherwise the data
          of the previous `run` may be processed with the metadata of the new sub-sequences).

        When the scheme is stopped, this method is equivalent to `set_subsequences`.

        :return: the metadata of the data acquired with the prepared sub-sequences (the input of the processing).
        """
        if not self._is_started:
            return self.set_subsequences(subsequences, processing=processing, sris=sris)
        if processing is not None and processing is not self._current_processing_spec:
            raise ValueError("Only the currently used processing can be updated while the scheme is running; "
                             "please stop the scheme and use set_subsequences instead.")
        sris = [] if sris is None else sris
        subsequences = self._convert_to_subsequences(subsequences)
        arrus_ops = arrus.utils.core.convert_to_arrus_subsequences(subsequences)
        arrus_sris = arrus.utils.core.convert_to_optional_vector(sris)
        upload_result = self._session_handle.prepareSubsequences(arrus_ops, arrus_sris)
        # NOTE: the output buffer stays the same (the same layout is required); self.buffer is kept, so the
        # processing does not have to re-bind to a new buffer wrapper.
        result_metadatas = self._create_subsequence_metadata(subsequences, upload_result)
        self._prepared_subsequences = (result_metadatas, self._current_processing_spec)
        return result_metadatas

    def _apply_processing_updates(self, until_element):
        """Applies the processing updates for the sub-sequences used starting from the given element."""
        while self._pending_processing_updates and self._pending_processing_updates[0][0] <= until_element:
            _, metadatas, processing = self._pending_processing_updates.pop(0)
            self._set_processing(self.buffer, metadatas, processing, [])

    def _apply_all_processing_updates(self):
        updates = [u[1:] for u in self._pending_processing_updates]
        if self._prepared_subsequences is not None:
            updates.append(self._prepared_subsequences)
        self._pending_processing_updates = []
        self._prepared_subsequences = None
        for metadatas, processing in updates:
            self._set_processing(self.buffer, metadatas, processing, [])

    def _create_subsequence_metadata(self, subsequences, upload_result):
        """Creates the metadata for the given sub-sequences (each of the uploaded sequences)."""
        us_device: Ultrasound = self.get_device("/Ultrasound:0")
        result_metadatas = []
        for array_id, (ops, array, metadata) in enumerate(
                zip(subsequences, self.buffer.elements[0].arrays, self.metadatas)):
            input_shape = array.shape
            sequence = metadata.context.sequence.get_subsequence(ops)
            raw_sequence = metadata.context.raw_sequence.get_subsequence(ops)
            data_description = us_device.get_data_description_updated_for_subsequence(array_id, upload_result, sequence, ops=ops)
            fac = dataclasses.replace(
                metadata.context,
                sequence=sequence,
                raw_sequence=raw_sequence
            )
            metadata = metadata.copy(
                input_shape=input_shape,
                data_desc=data_description,
                context=fac,
            )
            result_metadatas.append(metadata)
        return result_metadatas

    def _get_uploaded_sequences(self):
        if self._current_scheme is None:
            raise ValueError("Please upload the scheme first")
        sequences = self._current_scheme.tx_rx_sequence
        if not isinstance(sequences, Iterable):
            sequences = [sequences]
        return sequences

    def _convert_to_subsequences(self, subsequences):
        """
        Converts the input sub-sequence specification to the list of the TX/RX ordinal numbers,
        for each of the uploaded TX/RX sequences.
        """
        sequences = self._get_uploaded_sequences()
        n_sequences = len(sequences)

        if len(subsequences) > 0 and all(isinstance(s, (int, np.integer)) for s in subsequences):
            # A single, flat list of the TX/RX numbers was provided.
            if n_sequences != 1:
                raise ValueError("A flat list of the TX/RX numbers can be used only when a single TX/RX "
                                 f"sequence is uploaded (currently uploaded: {n_sequences}). "
                                 "Please provide a separate list of TX/RXs for each sequence.")
            subsequences = [subsequences]
        if len(subsequences) != n_sequences:
            raise ValueError(f"Exactly {n_sequences} sub-sequences should be provided "
                             f"(got: {len(subsequences)}).")
        result = []
        for sequence, ops in zip(sequences, subsequences):
            if isinstance(ops, slice):
                ops = range(*ops.indices(len(sequence.ops)))
            result.append([int(op) for op in ops])
        return result

    def set_subsequence(self, start=None, end=None, array_id=0, processing=None, sri=None):
        """
        Turns on the sequence with the arrayId and sets the TX/RXs that should be executed. This method turns off
        all the uploaded TX/RX sequences except the sequence pointed by `arrayId`.

        The TX/RXs to run can be provided either as the [start, end) range, or as an explicit list of the TX/RX
        ordinal numbers, e.g. `session.set_subsequence([2, 3, 5, 8, 13])`.

        This method requires that:

        - at least one TX/RX is selected,
        - the scheme was uploaded,
        - the TX/RX sequence length is greater than the `end` value (and than any of the provided TX/RX numbers),
        - the scheme is stopped.

        :param start: the TX/RX number which should now be the first TX/RX, or the list of the TX/RXs to run
        :param end: the TX/RX number which should now be the last TX/RX (exclusive)
        :param sri: the new SRI to apply
        :param array_id: id array to select, default: array with id 0
        :param processing: processing that should be used to process the output data for the given sub-sequence
        :return: the new data buffer and metadata
        """
        n_sequences = len(self._get_uploaded_sequences())
        if isinstance(start, Iterable):
            if end is not None:
                raise ValueError("The `end` parameter should not be used with the list of TX/RXs.")
            ops = [int(op) for op in start]
        else:
            if start is None or end is None:
                raise ValueError("Please provide the [start, end) range or the list of the TX/RXs to run.")
            if start >= end:
                raise ValueError("The `set_subsequence` method requires start < end.")
            ops = list(range(int(start), int(end)))
        if len(ops) == 0:
            raise ValueError("At least one TX/RX should be selected.")

        # Turn off all the other sequences.
        subsequences = [[] for _ in range(n_sequences)]
        sris = [None]*n_sequences

        subsequences[array_id] = ops
        sris[array_id] = sri

        return self.set_subsequences(subsequences=subsequences, sris=sris, processing=processing)

    def _set_processing(self, buffer, metadatas, processing, sequences):
        # Try to update the currently running processing first: when the user provides exactly the same
        # processing as the one that is currently in use (e.g. when selecting a new TX/RX sub-sequence),
        # there is no need to re-create the whole processing pipeline -- it's enough to update the
        # operations that depend on the acquisition parameters (e.g. RxBeamforming, ScanConversion).
        if (self._current_processing is not None
                and processing is not None
                and processing is self._current_processing_spec):
            try:
                return self._current_processing.update(buffer, metadatas)
            except ValueError as e:
                # The processing cannot be updated (e.g. the output data shape has changed)
                # -- re-create it from scratch.
                arrus.logging.log(
                    arrus.logging.DEBUG,
                    f"Re-creating the data processing pipeline, reason: {e}")
        # setup processing
        if self._current_processing is not None:
            self._current_processing.close()
            self._current_processing = None
        self._current_processing_spec = None

        if processing is not None:
            # setup processing
            import arrus.utils.imaging as _imaging
            # The object provided by the user -- kept in order to be able to determine whether the
            # currently running processing can be updated (see the beginning of this method).
            processing_spec = processing
            if not isinstance(processing, _imaging.Processing):
                # Wrap into the Processing object.
                processing = _imaging.Processing(
                    graph=processing,
                    callback=None,
                )

            # GPU settings
            gpu_memory_limit_percentage = None
            use_memory_pool = True  # Default to True
            # Whether to use P2P DMA is now derived from the output buffer placement
            # (set via Scheme.output_buffer.placement); P2P DMA is enabled when the
            # buffer is placed on the GPU.
            placement = getattr(self._current_scheme.output_buffer, "placement", None)
            use_p2p_dma = (
                placement is not None
                and arrus.devices.device.parse_device_id(placement).device_type
                    == arrus.devices.device.DeviceType("GPU")
            )
            if self._session_handle.hasDevice("/GPU:0"):
                # Read GPU specific settings
                gpu = self.get_device("/GPU:0")
                gpu_settings = gpu.get_settings()
                gpu_memory_limit_percentage = gpu_settings.memory_limit_percentage
                use_memory_pool = gpu_settings.use_memory_pool

            processing_runner = _imaging.ProcessingRunner(
                input_buffer=buffer, metadata=metadatas, processing=processing,
                use_memory_pool=use_memory_pool,
                gpu_memory_limit_percentage=gpu_memory_limit_percentage,
                use_p2p_dma=use_p2p_dma
            )
            outputs = processing_runner.outputs
            self._current_processing = processing_runner
            self._current_processing_spec = processing_spec
        else:
            # Device buffer and const_metadata
            outputs = buffer, metadatas
        return outputs

    def _contains_py_params(self, params):
        # Currently only start/stop params must by handled
        # by the Python layer, because os the self._buffer handle
        return Us4R.SEQUENCE_START_VAR in params or Us4R.SEQUENCE_END_VAR in params

    def _remove_py_params(self, params):
        params = params.copy()
        params.pop(Us4R.SEQUENCE_START_VAR, None)
        params.pop(Us4R.SEQUENCE_END_VAR, None)
        return params

    def _handle_py_params(self, params):
        # Currently only start/stop params must be handled in the Python layer.
        sequence_start = params.get(Us4R.SEQUENCE_START_VAR, None)
        sequence_end = params.get(Us4R.SEQUENCE_START_VAR, None)
        self.set_subsequence(sequence_start, sequence_end)

    # def set_current_medium(self, medium: arrus.medium.Medium):
    #     # TODO mutex, forbid when context is frozen (e.g. when us4r is running)
    #     raise RuntimeError("NYI")

    def _create_py_devices(self):
        devices = {
            (arrus.core.DeviceType_CPU, 0) : arrus.devices.cpu.CPU(0)
        }
        return devices

    def _create_kernel_context(self, seq, device, medium, hardware_ddc,
                               constants):
        return arrus.kernels.kernel.KernelExecutionContext(
            device=device, medium=medium, op=seq, custom={},
            hardware_ddc=hardware_ddc,
            constants=constants
        )

    def _create_frame_acquisition_context(self, seq, raw_seq, device, medium,
                                          constants):
        return arrus.metadata.FrameAcquisitionContext(
            device=device, sequence=seq, raw_sequence=raw_seq,
            medium=medium, custom_data={},
            constants=constants
        )

    def _group_constants_by_sequence_name(self, sequences, constants):
        # Validate constant names:
        #
        # - the sequence name should be in the collection of sequences,
        # - the constant name is expected to be /{SequenceName}/txFocus (currently only TX focus is supported)
        result = defaultdict(list)
        # TODO NOTE: assuming Us4R:0 device here
        pattern = re.compile(r"^/([A-Za-z][A-Za-z0-9_:]*)/([^/]+)$")
        sequence_names = {s.name.strip() for s in sequences}
        for constant in constants:
            r = pattern.match(constant.name)
            if r:
                sequence_name, parameter_name = r.groups()
                if sequence_name not in sequence_names:
                    raise ValueError(f"One of the constants is assigned to unknown sequence with name {sequence_name}")
                if not parameter_name.startswith("txFocus"):
                    raise ValueError(f"Currently only 'txFocus' parameter is supported (got: {parameter_name})")
                result[sequence_name].append(constant)
            else:
                raise ValueError(f"The Constant name should follow the following pattern: {pattern.pattern}")
        return result


