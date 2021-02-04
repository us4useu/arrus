import numpy as np
import math
import scipy
import scipy.signal as signal
import scipy.ndimage
import arrus.metadata
import arrus.devices.device
import arrus.devices.cpu
import arrus.devices.gpu
import arrus.kernels.imaging
import queue
import dataclasses


class Pipeline:
    """
    Imaging pipeline.

    Processes given data,metadata using a given sequence of steps.
    The processing will be performed on a given device ('placement').
    """
    def __init__(self, steps, placement=None):
        self.steps = steps
        self._host_registered = None
        self._placement = None
        if placement is not None:
            self.set_placement(placement)

    def __call__(self, data):
        """
        :param data: numpy array with data to process
        :return:
        """
        data = self.num_pkg.asarray(data)
        for step in self.steps:
            data = step._process(data)
        return data

    def _initialize(self, data):
        self.__call__(data)

    def initialize(self, const_metadata):
        input_shape = const_metadata.input_shape
        input_dtype = const_metadata.dtype
        for step in self.steps:
            const_metadata = step._prepare(const_metadata)
        # Force cupy to recompile kernels before running the pipeline.
        init_array = self.num_pkg.zeros(input_shape, dtype=input_dtype)+1000
        self._initialize(init_array)
        return const_metadata

    def set_placement(self, device):
        """
        Sets the pipeline to be executed on a particular device.

        :param device: device on which the pipeline should be executed
        """
        device_type = None
        if isinstance(device, str):
            # Device id
            device_type = arrus.devices.device.get_device_type_str(device)
        elif isinstance(device, arrus.devices.device.DeviceId):
            device_type = device.device_type.type
        elif isinstance(device, arrus.devices.device.Device):
            device_type = device.get_device_id().device_type.type
        self._placement = device_type
        # Initialize steps with a proper library.
        if self._placement == "GPU":
            import cupy as cp
            import cupyx.scipy.ndimage as cupy_scipy_ndimage
            pkgs = dict(num_pkg=cp, filter_pkg=cupy_scipy_ndimage)
        elif self._placement == "CPU":
            import scipy.ndimage
            pkgs = dict(num_pkg=np, filter_pkg=scipy.ndimage)
        else:
            raise ValueError(f"Unsupported device: {device}")
        for step in self.steps:
            step.set_pkgs(**pkgs)
        self.num_pkg = pkgs['num_pkg']
        self.filter_pkg = pkgs['filter_pkg']

    def register_host_buffer(self, buffer):
        if self._placement == "GPU":
            import cupy as cp
            for element in buffer.elements:
                cp.cuda.runtime.hostRegister(element.data.ctypes.data, element.size, 1)
            self._host_registered = buffer

    def stop(self):
        if self._host_registered is not None:
            import cupy as cp
            for element in self._host_registered.elements:
                cp.cuda.runtime.hostUnregister(element.data.ctypes.data)


class Operation:
    """
    An operation to perform in the imaging pipeline -- one data processing
    stage.
    """
    def _initialize(self, data):
        """
        Initialization function.

        This function will be called on a cupy initialization stage.

        By default, it runs `_process` function on a test cupy data.

        :return: the processed data.
        """
        return self._process(data)

    def set_pkgs(self, **kwargs):
        """
        Provides to possibility to gather python packages for numerical
        processing and filtering.

        The provided kwargs are:

        - `num_pkg`: numerical package: numpy for CPU, cupy for GPU
        - `filter_pkg`: scipy.ndimage for CPU, cupyx.scipy.ndimage for GPU
        """
        pass

    def _prepare(self, const_metadata):
        """
        Function that will called when the processing pipeline is prepared.

        :param const_metadata: const metadata describing output from the \
          previous Operation.
        :return: const metadata describing output of this Operation.
        """
        raise ValueError("Calling abstract method")

    def _process(self, data):
        """
        Function that will be called when new data arrives.

        :param data: input data
        :return: output data
        """
        raise ValueError("Calling abstract method")


class Lambda(Operation):
    """
    Custom function to perform on data from a given step.


    """

    def __init__(self, function):
        """
        Lambda op constructor.

        :param function: a function with a single input: (cupy or numpy array \
          with the data)
        """
        self.func = function
        pass

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def _initialize(self, data):
        pass

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def _process(self, data):
        return self.func(data)


class BandpassFilter:
    """
    Bandpass filtering to apply to signal data.

    A bandwidth [0.5, 1.5]*center_frequency is currently used.

    The filtering is performed along the last axis.

    Currently only FIR filter is available.

    NOTE: consider using Filter op, which provides more possibilities to
    define what kind of filter is used (e.g. by providing filter coefficients).
    """

    def __init__(self, numtaps=7, bounds=(0.5, 1.5), filter_type="butter",
                 num_pkg=None, filter_pkg=None):
        """
        Bandpass filter constructor.

        :param bounds: determines filter's frequency boundaries,
            e.g. setting 0.5 will give a bandpass filter
            [0.5*center_frequency, 1.5*center_frequency].
        """
        self.taps = None
        self.numtaps = numtaps
        self.bound_l, self.bound_r = bounds
        self.filter_type = filter_type
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        l, r = self.bound_l, self.bound_r
        center_frequency = const_metadata.context.sequence.pulse.center_frequency
        sampling_frequency = const_metadata.data_description.sampling_frequency
        # FIXME(pjarosik) implement iir filter
        taps, _ = scipy.signal.butter(
                2,
                [l * center_frequency, r * center_frequency],
                btype='bandpass', fs=sampling_frequency)
        self.taps = self.xp.asarray(taps).astype(self.xp.float32)
        return const_metadata

    def _process(self, data):
        result = self.filter_pkg.convolve1d(data, self.taps, axis=-1,
                                            mode='constant')
        return result


class Filter:
    """
    Filter data in one dimension along the last axis.

    The filtering is performed according to the given filter taps.

    Currently only FIR filter is available.
    """

    def __init__(self, taps, num_pkg=None, filter_pkg=None):
        """
        Bandpass filter constructor.

        :param taps: filter feedforward coefficients
        """
        self.taps = taps
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        self.taps = self.xp.asarray(self.taps).astype(self.xp.float32)
        return const_metadata

    def _process(self, data):
        result = self.filter_pkg.convolve1d(data, self.taps, axis=-1,
                                            mode='constant')
        return result


class QuadratureDemodulation:
    """
    Quadrature demodulation (I/Q decomposition).
    """
    def __init__(self, num_pkg=None):
        self.mod_factor = None
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _is_prepared(self):
        return self.mod_factor is not None

    def _prepare(self, const_metadata):
        xp = self.xp
        fs = const_metadata.data_description.sampling_frequency
        fc = const_metadata.context.sequence.pulse.center_frequency
        _, _, n_samples = const_metadata.input_shape
        t = (xp.arange(0, n_samples) / fs).reshape(1, 1, -1)
        self.mod_factor = (2 * xp.cos(-2 * xp.pi * fc * t)
                           + 2 * xp.sin(-2 * xp.pi * fc * t) * 1j)
        self.mod_factor = self.mod_factor.astype(xp.complex64)
        return const_metadata.copy(is_iq_data=True, dtype="complex64")

    def _process(self, data):
        return self.mod_factor * data


class Decimation:
    """
    Decimation + CIC (Cascade Integrator-Comb) filter.

    See: https://en.wikipedia.org/wiki/Cascaded_integrator%E2%80%93comb_filter
    """

    def __init__(self, decimation_factor, cic_order, num_pkg=None, impl="legacy"):
        """
        Decimation op constructor.

        :param decimation_factor: decimation factor to apply
        :param cic_order: CIC filter order
        """
        self.decimation_factor = decimation_factor
        self.cic_order = cic_order
        self.xp = num_pkg
        self.impl = impl
        if self.impl == "legacy":
            self._decimate = self._legacy_decimate
        elif self.impl == "fir":
            self._decimate = self._fir_decimate

    def set_pkgs(self, num_pkg, filter_pkg, **kwargs):
        self.xp = num_pkg
        self.filter_pkg = filter_pkg # not used by the GPU implementation (custom kernel for complex input data)

    def _prepare(self, const_metadata):
        new_fs = (const_metadata.data_description.sampling_frequency
                  / self.decimation_factor)
        new_signal_description = arrus.metadata.EchoDataDescription(
            sampling_frequency=new_fs, custom=
            const_metadata.data_description.custom)

        n_frames, n_channels, n_samples = const_metadata.input_shape
        total_n_samples = n_frames*n_channels*n_samples

        output_shape = n_frames, n_channels, n_samples//self.decimation_factor

        # CIC FIR coefficients
        if self.impl == "fir":
            cicFir = self.xp.array([1], dtype=self.xp.float32)
            cicFir1 = self.xp.ones(self.decimation_factor, dtype=self.xp.float32)
            for i in range(self.cic_order):
                cicFir = self.xp.convolve(cicFir, cicFir1, 'full')
            fir_taps = cicFir
            n_fir_taps = len(fir_taps)
            if self.xp == np:
                def _cpu_fir_filter(data):
                    return self.filter_pkg.convolve1d(
                        np.real(data), fir_taps,
                        axis=-1, mode='constant',
                        cval=0, origin=-1) \
                        + self.filter_pkg.convolve1d(np.imag(data),
                                          fir_taps, axis=-1,
                                          mode='constant', cval=0,
                                          origin=-1)*1j
                # CPU
                self._fir_filter = _cpu_fir_filter
            else:
                # GPU
                import cupy as cp
                _fir_output_buffer = cp.zeros(const_metadata.input_shape,
                                                dtype=cp.complex64)
                # Kernel settings
                from arrus.utils.fir import (
                    get_default_grid_block_size,
                    get_default_shared_mem_size,
                    run_fir)
                grid_size, block_size = get_default_grid_block_size(n_samples, total_n_samples)
                shared_memory_size = get_default_shared_mem_size(n_samples, n_fir_taps)

                def _gpu_fir_filter(data):
                    run_fir(grid_size, block_size,
                        (_fir_output_buffer, data, n_samples,
                         total_n_samples, fir_taps, n_fir_taps),
                            shared_memory_size)
                    return _fir_output_buffer

                self._fir_filter = _gpu_fir_filter
        return const_metadata.copy(data_desc=new_signal_description,
                                   input_shape=output_shape)

    def _process(self, data):
        return self._decimate(data)

    def _fir_decimate(self, data):
        fir_output = self._fir_filter(data)
        data_out = fir_output[:, :, 0::self.decimation_factor]
        return data_out

    def _legacy_decimate(self, data):
        data_out = data
        for i in range(self.cic_order):
            data_out = self.xp.cumsum(data_out, axis=-1)
        data_out = data_out[:, :, 0::self.decimation_factor]
        for i in range(self.cic_order):
            data_out[:, :, 1:] = self.xp.diff(data_out, axis=-1)
        return data_out


class RxBeamforming:
    """
    Classical rx beamforming (reconstructing scanline by scanline).

    Expected input data shape: n_emissions, n_rx, n_samples

    Currently the beamforming op works only for LIN sequence output data.
    """

    def __init__(self, num_pkg=None):
        self.delays = None
        self.buffer = None
        self.rx_apodization = None
        self.xp = num_pkg
        self.interp1d_func = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg
        if self.xp is np:
            import scipy.interpolate

            def numpy_interp1d(input, samples, output):
                n_samples = input.shape[-1]
                x = np.arange(0, n_samples)
                interpolator = scipy.interpolate.interp1d(
                    x, input, kind="linear", bounds_error=False,
                    fill_value=0.0)
                interp_values = interpolator(samples)
                n_scanlines, _, n_samples = interp_values.shape
                interp_values = np.reshape(interp_values, (n_scanlines, n_samples))
                output[:] = interp_values

            self.interp1d_func = numpy_interp1d
        else:
            import cupy as cp
            if self.xp != cp:
                raise ValueError(f"Unhandled numerical package: {self.xp}")
            import arrus.utils.interpolate
            self.interp1d_func = arrus.utils.interpolate.interp1d

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        # TODO verify that all angles, focal points are the same
        # TODO make sure start_sample is computed appropriately
        context = const_metadata.context
        probe_model = const_metadata.context.device.probe.model
        seq = const_metadata.context.sequence
        raw_seq = const_metadata.context.raw_sequence
        medium = const_metadata.context.medium
        rx_aperture_center_element = np.array(seq.rx_aperture_center_element)

        self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape
        self.is_iq = const_metadata.is_iq_data
        if self.is_iq:
            buffer_dtype = self.xp.complex64
        else:
            buffer_dtype = self.xp.float32

        # -- Output buffer
        self.buffer = self.xp.zeros((self.n_tx, self.n_rx * self.n_samples),
                                    dtype=buffer_dtype)

        # -- Delays
        acq_fs = (const_metadata.context.device.sampling_frequency
                  / seq.downsampling_factor)
        fs = const_metadata.data_description.sampling_frequency
        fc = seq.pulse.center_frequency
        n_periods = seq.pulse.n_periods
        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound
        tx_angle = 0  # TODO use appropriate tx angle
        start_sample = seq.rx_sample_range[0]
        rx_aperture_origin = _get_rx_aperture_origin(seq)

        _, _, tx_delay_center = arrus.kernels.imaging.compute_tx_parameters(
            seq, probe_model, c)

        burst_factor = n_periods / (2 * fc)
        # -start_sample compensates the fact, that the data indices always start from 0
        initial_delay = - start_sample / acq_fs
        if seq.init_delay == "tx_start":
            burst_factor = n_periods / (2 * fc)
            _, _, tx_delay_center = arrus.kernels.imaging.compute_tx_parameters(
                seq, probe_model, c)
            initial_delay += tx_delay_center + burst_factor
        elif not seq.init_delay == "tx_center":
            raise ValueError(f"Unrecognized init_delay value: {initial_delay}")

        radial_distance = (
                (start_sample / acq_fs + np.arange(0, self.n_samples) / fs)
                * c / 2
        )
        x_distance = (radial_distance * np.sin(tx_angle)).reshape(1, -1)
        z_distance = radial_distance * np.cos(tx_angle).reshape(1, -1)

        origin_offset = (rx_aperture_origin[0]
                         - (seq.rx_aperture_center_element[0]))
        # New coordinate system: origin: rx aperture center
        element_position = ((np.arange(0, self.n_rx) + origin_offset)
                            * probe_model.pitch)
        element_position = element_position.reshape((self.n_rx, 1))
        if not probe_model.is_convex_array():
            element_angle = np.zeros((self.n_rx, 1))
            element_x = element_position
            element_z = np.zeros((self.n_rx, 1))
        else:
            element_angle = element_position / probe_model.curvature_radius
            element_x = probe_model.curvature_radius * np.sin(element_angle)
            element_z = probe_model.curvature_radius * (
                    np.cos(element_angle) - 1)

        tx_distance = radial_distance
        rx_distance = np.sqrt(
            (x_distance - element_x) ** 2 + (z_distance - element_z) ** 2)

        self.t = (tx_distance + rx_distance) / c + initial_delay
        self.delays = self.t * fs # in number of samples
        total_n_samples = self.n_rx * self.n_samples
        # Move samples outside the available area
        self.delays[np.isclose(self.delays, self.n_samples-1)] = self.n_samples-1
        self.delays[self.delays > self.n_samples-1] = total_n_samples + 1
        # (RF data will also be unrolled to a vect. n_rx*n_samples elements,
        #  row-wise major order).
        self.delays = self.xp.asarray(self.delays)
        self.delays += self.xp.arange(0, self.n_rx).reshape(self.n_rx, 1) \
                       * self.n_samples
        self.delays = self.delays.reshape(-1, self.n_samples * self.n_rx) \
            .astype(self.xp.float32)
        # Apodization
        lambd = c / fc
        max_tang = math.tan(
            math.asin(min(1, 2 / 3 * lambd / probe_model.pitch)))
        rx_tang = np.abs(np.tan(np.arctan2(x_distance - element_x,
                                           z_distance - element_z) - element_angle))
        rx_apodization = (rx_tang < max_tang).astype(np.float32)
        rx_apod_sum = np.sum(rx_apodization, axis=0)
        rx_apod_sum[rx_apod_sum == 0] = 1
        rx_apodization = rx_apodization/(rx_apod_sum.reshape(1, self.n_samples))
        self.rx_apodization = self.xp.asarray(rx_apodization)
        # IQ correction
        self.t = self.xp.asarray(self.t)
        self.iq_correction = self.xp.exp(1j * 2 * np.pi * fc * self.t) \
            .astype(self.xp.complex64)
        # Create new output shape
        return const_metadata.copy(input_shape=(self.n_tx, self.n_samples))

    def _process(self, data):
        data = data.copy().reshape(self.n_tx, self.n_rx * self.n_samples)

        self.interp1d_func(data, self.delays, self.buffer)
        out = self.buffer.reshape((self.n_tx, self.n_rx, self.n_samples))
        if self.is_iq:
            out = out * self.iq_correction
        out = out * self.rx_apodization
        out = self.xp.sum(out, axis=1)
        return out.reshape((self.n_tx, self.n_samples))


class EnvelopeDetection:
    """
    Envelope detection (Hilbert transform).

    Currently this op works only for I/Q data (complex64).
    """

    def __init__(self, num_pkg=None):
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata.copy(is_iq_data=False, dtype="float32")

    def _process(self, data):
        if data.dtype != self.xp.complex64:
            raise ValueError(
                f"Data type {data.dtype} is currently not supported.")
        return self.xp.abs(data)


class Transpose:
    """
    Data transposition.
    """

    def __init__(self, axes=None):
        """
        :param axes: permutation of axes to apply
        """
        self.axes = axes
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata):
        input_shape = const_metadata.input_shape
        axes = list(range(len(input_shape)))[::-1] if self.axes is None else self.axes
        output_shape = tuple(input_shape[ax] for ax in axes)
        return const_metadata.copy(input_shape=output_shape)

    def _process(self, data):
        return self.xp.transpose(data, self.axes)


class ScanConversion:
    """
    Scan conversion (interpolation to target mesh).

    Currently linear interpolation is used by default, values outside
    the input mesh will be set to 0.0.

    Currently the op is implement for CPU only.

    Currently, the op is available only for convex probes.


    """

    def __init__(self, x_grid, z_grid):
        """
        Scan converter constructor.

        :param x_grid: a vector of grid points along OX axis [m]
        :param z_grid: a vector of grid points along OZ axis [m]
        """
        self.dst_points = None
        self.dst_shape = None
        self.x_grid = x_grid.reshape(1, -1)
        self.z_grid = z_grid.reshape(1, -1)
        self.is_gpu = False

    def set_pkgs(self, num_pkg, **kwargs):
        if num_pkg != np:
            self.is_gpu = True
        # Ignoring provided num. package - currently CPU implementation is
        # available only.

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        probe = const_metadata.context.device.probe.model
        medium = const_metadata.context.medium
        data_desc = const_metadata.data_description

        if not probe.is_convex_array():
            raise ValueError(
                "Scan conversion currently works for convex probes data only.")

        n_samples, _ = const_metadata.input_shape
        seq = const_metadata.context.sequence
        custom_data = const_metadata.context.custom_data

        acq_fs = (const_metadata.context.device.sampling_frequency
                  / seq.downsampling_factor)
        fs = data_desc.sampling_frequency

        start_sample = seq.rx_sample_range[0]

        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound

        tx_ap_cent_ang, _, _ = arrus.kernels.imaging.get_tx_aperture_center_coords(seq, probe)

        z_grid_moved = self.z_grid.T + probe.curvature_radius - np.max(
             probe.element_pos_z)

        self.radGridIn = (
                (start_sample / acq_fs + np.arange(0, n_samples) / fs)
                * c / 2)

        self.azimuthGridIn = tx_ap_cent_ang
        azimuthGridOut = np.arctan2(self.x_grid, z_grid_moved)
        radGridOut = (np.sqrt(self.x_grid ** 2 + z_grid_moved ** 2)
                      - probe.curvature_radius)

        dst_points = np.dstack((radGridOut, azimuthGridOut))
        w, h, d = dst_points.shape
        self.dst_points = dst_points.reshape((w * h, d))
        self.dst_shape = len(self.z_grid.squeeze()), len(self.x_grid.squeeze())
        return const_metadata.copy(input_shape=self.dst_shape)

    def _process(self, data):
        if self.is_gpu:
            data = data.get()
        data[np.isnan(data)] = 0.0
        self.interpolator = scipy.interpolate.RegularGridInterpolator(
            (self.radGridIn, self.azimuthGridIn), data, method="linear",
            bounds_error=False, fill_value=0)
        return self.interpolator(self.dst_points).reshape(self.dst_shape)


class LogCompression:
    """
    Converts data to decibel scale.
    """
    def __init__(self):
        pass

    def set_pkgs(self, **kwargs):
        # Intentionally ignoring num. package -
        # currently numpy is available only.
        pass

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def _process(self, data):
        if not isinstance(data, np.ndarray):
            data = data.get()
        data[data == 0] = 1e-9
        return 20 * np.log10(data)


class DynamicRangeAdjustment:
    """
    Clips data values to given range.
    """

    def __init__(self, min=20, max=80):
        """
        Constructor.

        :param min: minimum value to clamp
        :param max: maximum value to clamp
        """
        self.min = min
        self.max = max
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def _process(self, data):
        return self.xp.clip(data, a_min=self.min, a_max=self.max)


class ToGrayscaleImg:
    """
    Converts data to grayscale image (uint8).
    """

    def __init__(self):
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def _process(self, data):
        data = data - self.xp.min(data)
        data = data/self.xp.max(data)*255
        return data.astype(self.xp.uint8)


class Enqueue:
    """
    Copies the output data from the previous stage to the provided queue
    and passes the input data unmodified for further processing.

    Works on queue.Queue objects.

    :param queue: queue.Queue instance
    :param block: if true, will block the pipeline until a free slot is available,
      otherwise will raise queue.Full exception
    :param ignore_full: when true and block = False, this step will not throw
      queue.Full exception. Set it to true if it is ok to ommit some of the
      acquired frames.
    """

    def __init__(self, queue, block=True, ignore_full=False):
        self.queue = queue
        self.block = block
        self.ignore_full = ignore_full
        if self.block:
            self._process = self._put_block
        else:
            if self.ignore_full:
                self._process = self._put_ignore_full
            else:
                self._process = self._put_non_block
        self._copy_func = None

    def set_pkgs(self, num_pkg, **kwargs):
        if num_pkg == np:
            self._copy_func = np.copy
        else:
            import cupy as cp
            self._copy_func = cp.asnumpy

    def _prepare(self, const_metadata):
        return const_metadata

    def _initialize(self, data):
        pass

    def _put_block(self, data):
        self.queue.put(self._copy_func(data))
        return data

    def _put_non_block(self, data):
        self.queue.put_nowait(self._copy_func(data))
        return data

    def _put_ignore_full(self, data):
        try:
            self.queue.put_nowait(self._copy_func(data))
        except queue.Full:
            pass
        return data


class SelectFrames(Operation):
    """
    Selects frames for a given sequence for further processing.
    """

    def __init__(self, frames):
        """
        Constructor.

        :param frames: frames to select
        """
        self.frames = frames

    def set_pkgs(self, **kwargs):
        pass

    def _initialize(self, data):
        pass

    def _prepare(self, const_metadata):
        input_shape = const_metadata.input_shape
        context = const_metadata.context
        seq = context.sequence
        n_frames = len(self.frames)

        if len(input_shape) != 3:
            raise ValueError("The input should be 3-D "
                             "(frame number should be the first axis)")

        input_n_frames, d2, d3 = input_shape
        output_shape = (n_frames, d2, d3)
        # TODO make this op less prone to changes in op implementation
        if isinstance(seq, arrus.ops.imaging.PwiSequence):
            # select appropriate angles
            output_angles = seq.angles[self.frames]
            new_seq = dataclasses.replace(seq, angles=output_angles)
            new_context = const_metadata.context
            new_context = arrus.metadata.FrameAcquisitionContext(
                device=new_context.device, sequence=new_seq,
                raw_sequence=new_context.raw_sequence,
                medium=new_context.medium, custom_data=new_context.custom_data)
            return const_metadata.copy(input_shape=output_shape,
                                       context=new_context)
        else:
            return const_metadata.copy(input_shape=output_shape)

    def _process(self, data):
        return data[self.frames]


class Squeeze(Operation):
    """
    Squeezes input array (removes axes = 1).
    """
    def __init__(self):
        pass

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata):
        output_shape = tuple(i for i in const_metadata.input_shape if i != 1)
        return const_metadata.copy(input_shape=output_shape)

    def _process(self, data):
        return self.xp.squeeze(data)


class RxBeamformingImg:
    """
    Rx beamforming for synthetic aperture imaging.

    Expected input data shape: n_emissions, n_rx, n_samples

    Currently Plane Wave Imaging (Pwi) is supported only.
    """
    def __init__(self, x_grid, z_grid, num_pkg=None):
        self.x_grid = x_grid
        self.z_grid = z_grid
        self.delays = None
        self.buffer = None
        self.rx_apodization = None
        self.xp = num_pkg
        self.interp1d_func = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg
        if self.xp is np:
            import scipy.interpolate

            def numpy_interp1d(input, samples, output):
                n_samples = input.shape[-1]
                x = np.arange(0, n_samples)
                interpolator = scipy.interpolate.interp1d(
                    x, input, kind="linear", bounds_error=False,
                    fill_value=0.0)
                interpolated_values = interpolator(samples)
                output[:] = interpolated_values

            self.interp1d_func = numpy_interp1d
        else:
            import cupy as cp
            if self.xp != cp:
                raise ValueError(f"Unhandled numerical package: {self.xp}")
            import arrus.utils.interpolate
            self.interp1d_func = arrus.utils.interpolate.interp1d

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        probe_model = const_metadata.context.device.probe.model

        if probe_model.is_convex_array():
            raise ValueError("PWI reconstruction mode is available for "
                             "linear arrays only.")

        seq = const_metadata.context.sequence
        medium = const_metadata.context.medium

        self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape
        self.is_iq = const_metadata.is_iq_data
        if self.is_iq:
            self.buffer_dtype = self.xp.complex64
        else:
            self.buffer_dtype = self.xp.float32

        # -- Output buffer
        x_size = len(self.x_grid.flatten())
        z_size = len(self.z_grid.flatten())
        self.buffer_shape = (self.n_rx, x_size, z_size)
        self.buffer = self.xp.zeros(self.buffer_shape, dtype=self.buffer_dtype)\
            .flatten()
        self.buffer = self.xp.atleast_2d(self.buffer)
        self.lri_buffer = self.xp.zeros((self.n_tx, x_size, z_size),
                                        dtype=self.buffer_dtype)

        # -- Delays

        # --- Initial delay
        acq_fs = (const_metadata.context.device.sampling_frequency
                  / seq.downsampling_factor)
        fs = const_metadata.data_description.sampling_frequency
        fc = seq.pulse.center_frequency
        n_periods = seq.pulse.n_periods
        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound

        angles = np.atleast_1d(np.array(seq.angles))
        angles = np.expand_dims(angles, axis=(1, 2, 3))  # (ntx, 1, 1, 1)
        tx_delay_center = 0.5*(probe_model.n_elements-1)*probe_model.pitch*np.abs(np.tan(angles))/c
        tx_delay_center = np.squeeze(tx_delay_center)

        start_sample = seq.rx_sample_range[0]
        burst_factor = n_periods / (2 * fc)
        initial_delay = (- start_sample / acq_fs
                         + tx_delay_center
                         + burst_factor)
        initial_delay = np.array(initial_delay)
        initial_delay = initial_delay[..., np.newaxis, np.newaxis, np.newaxis]

        # --- Distances and apodizations
        lambd = c / fc
        max_tang = math.tan(
            math.asin(min(1, 2 / 3 * lambd / probe_model.pitch)))

        element_pos_x = probe_model.element_pos_x
        rx_aperture_origin = np.zeros(self.n_tx, dtype=np.int16)
        rx_aperture_origin = np.expand_dims(rx_aperture_origin, axis=(1, 2, 3))
        # (ntx, 1, 1, 1)
        # TODO parametrize rx aperture size
        rx_aperture_size = probe_model.n_elements
        irx = np.arange(0, probe_model.n_elements)
        irx = np.expand_dims(irx, axis=(0, 2, 3))  # (1, nrx, 1, 1)
        itx = np.expand_dims(np.arange(0, self.n_tx), axis=(1, 2, 3))
        rx_aperture_element_pos_x = (rx_aperture_origin+irx -
                                     (probe_model.n_elements-1)/2)*probe_model.pitch


        # Output delays/apodization
        x_grid = np.expand_dims(self.x_grid, axis=(0, 1, 3))
        # (1, 1, x_size, 1)
        z_grid = np.expand_dims(self.z_grid, axis=(0, 1, 2))

        # (1, 1, 1, z_size)

        tx_distance = x_grid*np.sin(angles) + z_grid*np.cos(angles)
        # (ntx, 1, x_size, z_size)
        r1 = (x_grid-element_pos_x[0])*np.cos(angles) - z_grid*np.sin(angles)
        r2 = (x_grid-element_pos_x[-1])*np.cos(angles) - z_grid*np.sin(angles)
        tx_apodization = np.logical_and(r1 >= 0, r2 <= 0).astype(np.int8)

        rx_distance = np.sqrt((x_grid-rx_aperture_element_pos_x)**2 + z_grid**2)
        # (ntx, nrx, x_size, z_size)
        # rx_distance = np.expand_dims(rx_distance, axis=(0, 1))
        rx_tangens = np.abs(x_grid - rx_aperture_element_pos_x)/(z_grid+1e-12)
        rx_apodization = (rx_tangens < max_tang).astype(np.int8)

        delay_total = (tx_distance + rx_distance)/c + initial_delay
        samples = delay_total * fs
        # samples outside should be neglected
        samples[np.logical_or(samples < 0, samples >= self.n_samples-1)] = np.Inf
        samples = samples + irx*self.n_samples
        samples[np.isinf(samples)] = -1
        samples = samples.astype(np.float32)
        rx_weights = tx_apodization*rx_apodization

        # IQ correction
        if self.is_iq:
            t = self.xp.asarray(delay_total)
            rx_weights = self.xp.asarray(rx_weights)
            self.iq_correction = self.xp.exp(1j*2*np.pi*fc*t)\
                .astype(self.xp.complex64)
            self.rx_weights = self.iq_correction * rx_weights
        else:
            self.rx_weights = self.xp.asarray(rx_weights)
        tx_weights = np.sum(rx_weights, axis=1)
        self.rx_weights = self.rx_weights.astype(self.xp.complex64)
        self.tx_weights = self.xp.asarray(tx_weights).astype(self.xp.float32)
        self.samples = self.xp.asarray(samples).astype(self.xp.float32)
        # Create new output shape
        return const_metadata.copy(input_shape=(len(self.x_grid),
                                                len(self.z_grid)))

    def _process(self, data):
        data = data.copy().reshape(self.n_tx, self.n_rx*self.n_samples)
        for i in range(self.n_tx):
            self.interp1d_func(data[i:(i+1)], self.samples[i:(i+1)].flatten(),
                               self.buffer)
            rf_rx = self.buffer.reshape(self.buffer_shape)
            rf_rx = rf_rx * self.rx_weights[i]
            rf_rx = np.sum(rf_rx, axis=0)
            self.lri_buffer[i] = rf_rx
        return np.sum(self.lri_buffer, axis=0)/np.sum(self.tx_weights, axis=0)


def _get_rx_aperture_origin(sequence):
    rx_aperture_size = sequence.rx_aperture_size
    rx_aperture_center_element = np.array(sequence.rx_aperture_center_element)
    rx_aperture_origin = np.round(rx_aperture_center_element -
                               (rx_aperture_size - 1) / 2 + 1e-9)
    return rx_aperture_origin


