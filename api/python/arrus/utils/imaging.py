import numpy as np
import math
import scipy
import scipy.signal as signal
import arrus.metadata
import arrus.devices.cpu
import arrus.devices.gpu
import arrus.kernels.imaging


class Pipeline:
    """
    Imaging pipeline.

    Processes given data,metadata using a given sequence of steps.
    The processing will be performed on a given device ('placement').
    """
    def __init__(self, steps, placement=None):
        self.steps = steps
        if placement is not None:
            self.set_placement(placement)

    def __call__(self, data):
        for step in self.steps:
            data = step(data)
        return data

    def initialize(self, const_metadata):
        input_shape = const_metadata.input_shape
        input_dtype = const_metadata.dtype
        for step in self.steps:
            const_metadata = step._prepare(const_metadata)
        # Force cupy to recompile kernels before running the pipeline.
        init_array = self.num_pkg.zeros(input_shape, dtype=input_dtype)
        self.__call__(init_array)
        return const_metadata

    def set_placement(self, device):
        """
        Sets the pipeline to be executed on a particular device.

        :param device: device on which the pipeline should be executed
        """
        self.placement = device
        # Initialize steps with a proper library.
        if isinstance(self.placement, arrus.devices.gpu.GPU):
            import cupy as cp
            import cupyx.scipy.ndimage as cupy_scipy_ndimage
            pkgs = dict(num_pkg=cp, filter_pkg=cupy_scipy_ndimage)
        elif isinstance(self.placement, arrus.devices.cpu.CPU):
            import scipy.ndimage
            pkgs = dict(num_pkg=np, filter_pkg=scipy.ndimage)
        else:
            raise ValueError(f"Unsupported device: {device}")
        for step in self.steps:
            step.set_pkgs(**pkgs)
        self.num_pkg = pkgs['num_pkg']
        self.filter_pkg = pkgs['filter_pkg']


class BandpassFilter:
    """
    Bandpass filtering to apply to signal data.

    A bandwidth [0.5, 1.5]*center_frequency is currently used.

    The filtering is performed along the last axis.

    Currently only FIR filter is available.
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
        self.taps = self.xp.asarray(taps)
        return const_metadata

    def __call__(self, data):
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

    def __call__(self, data):
        return self.mod_factor * data


class Decimation:
    """
    Decimation + CIC (Cascade Integrator-Comb) filter.

    See: https://en.wikipedia.org/wiki/Cascaded_integrator%E2%80%93comb_filter
    """

    def __init__(self, decimation_factor, cic_order, num_pkg=None):
        """
        Decimation op constructor.

        :param decimation_factor: decimation factor to apply
        :param cic_order: CIC filter order
        """
        self.decimation_factor = decimation_factor
        self.cic_order = cic_order
        self.xp = num_pkg

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata):
        new_fs = (const_metadata.data_description.sampling_frequency
                  / self.decimation_factor)
        new_signal_description = arrus.metadata.EchoDataDescription(
            sampling_frequency=new_fs, custom=
            const_metadata.data_description.custom)

        n_frames, n_channels, n_samples = const_metadata.input_shape

        output_shape = n_frames, n_channels, n_samples//self.decimation_factor
        return const_metadata.copy(data_desc=new_signal_description,
                                   input_shape=output_shape)

    def __call__(self, data):
        data_out = data
        for i in range(self.cic_order):
            data_out = self.xp.cumsum(data_out, axis=-1)

        data_out = data_out[:, :, 0:-1:self.decimation_factor]

        for i in range(self.cic_order):
            data_out[:, :, 1:] = self.xp.diff(data_out, axis=-1)
        return data_out


class RxBeamforming:
    """
    Rx beamforming.

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

        self.n_tx, self.n_rx, self.n_samples = const_metadata.input_shape

        # TODO store iq trait in the metadata.data_description
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
        # TODO keep cache data? Here all the tx/rx parameters are recomputed
        _, _, tx_delay_center = arrus.kernels.imaging.compute_tx_parameters(
            seq, probe_model, c)
        # Assuming, that all tx/rxs have the constant start sample value.
        if raw_seq is None:
            start_sample = context.custom_data["start_sample"] + 1
        else:
            start_sample = raw_seq.ops[0].rx.sample_range[0]
        rx_aperture_origin = _get_rx_aperture_origin(seq)

        burst_factor = n_periods / (2 * fc)
        initial_delay = (- start_sample / acq_fs
                         + tx_delay_center
                         + burst_factor)

        radial_distance = (
                (start_sample / acq_fs + np.arange(0, self.n_samples) / fs)
                * c / 2
        )
        x_distance = (radial_distance * np.sin(tx_angle)).reshape(1, -1)
        z_distance = radial_distance * np.cos(tx_angle).reshape(1, -1)

        origin_offset = (rx_aperture_origin[0]
                         - (seq.rx_aperture_center_element[0] + 1))

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
        self.delays = self.t * fs
        total_n_samples = self.n_rx * self.n_samples
        # Move samples outside the available area
        self.delays[self.delays > self.n_samples] = total_n_samples + 1
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

    def __call__(self, data):
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
        return const_metadata.copy(is_iq_data=False)

    def __call__(self, data):
        if data.dtype != self.xp.complex64:
            raise ValueError(
                f"Data type {data.dtype} is currently not supported.")
        return self.xp.abs(data)


class Transpose:
    """
    Data transposition.
    """

    def __init__(self, axes=None):
        self.axes = axes
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata):
        input_shape = const_metadata.input_shape
        axes = list(range(len(input_shape)))[::-1] if self.axes is None else self.axes
        output_shape = tuple(input_shape[ax] for ax in axes)
        return const_metadata.copy(input_shape=output_shape)

    def __call__(self, data):
        return self.xp.transpose(data, self.axes)


class ScanConversion:
    """
    Scan conversion (interpolation to target mesh).

    Currently linear interpolation is used by default, values outside
    the input mesh will be set to 0.0.

    Currently the op is implement for CPU only.
    :param x_grid: a vector of grid points along OX axis [m]
    :param z_grid: a vector of grid points along OZ axis [m]
    """

    def __init__(self, x_grid, z_grid):
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
        # TODO check if angle is zero and tx aperture is increasing
        # TODO compute center angle, etc.
        probe = const_metadata.context.device.probe.model
        medium = const_metadata.context.medium
        data_desc = const_metadata.data_description
        raw_seq = const_metadata.context.raw_sequence

        if not probe.is_convex_array():
            raise ValueError(
                "Scan conversion currently works for convex probes data only.")

        n_samples, _ = const_metadata.input_shape
        seq = const_metadata.context.sequence
        custom_data = const_metadata.context.custom_data
        if raw_seq is None:
            start_sample = custom_data["start_sample"]
        else:
            start_sample = raw_seq.ops[0].rx.sample_range[0]
        fs = data_desc.sampling_frequency

        if seq.speed_of_sound is not None:
            c = seq.speed_of_sound
        else:
            c = medium.speed_of_sound

        tx_ap_cent_ang, _, _ = arrus.kernels.imaging.get_tx_aperture_center_coords(seq, probe)

        z_grid_moved = self.z_grid.T + probe.curvature_radius - np.max(
            probe.element_pos_z)

        self.radGridIn = (
                (start_sample / fs + np.arange(0, n_samples) / fs)
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

    def __call__(self, data):
        if self.is_gpu:
            data = data.get()
        data[np.isnan(data)] = 0.0
        interpolator = scipy.interpolate.RegularGridInterpolator(
            (self.radGridIn, self.azimuthGridIn), data, method="linear",
            bounds_error=False, fill_value=0)
        res = interpolator(self.dst_points).reshape(self.dst_shape)
        return res


class LogCompression:
    """
    Converts to decibel scale.
    """
    def __init__(self):
        pass

    def set_pkgs(self, **kwargs):
        # Intentionally ignoring num. package -
        # currently numpy is available only.
        pass

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def __call__(self, data):
        if not isinstance(data, np.ndarray):
            data = data.get()
        data[data == 0] = 1e-9
        return 20 * np.log10(data)


class DynamicRangeAdjustment:

    def __init__(self, min=20, max=80):
        self.min = min
        self.max = max
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def __call__(self, data):
        return self.xp.clip(data, a_min=self.min, a_max=self.max)


class ToGrayscaleImg:

    def __init__(self):
        self.xp = None

    def set_pkgs(self, num_pkg, **kwargs):
        self.xp = num_pkg

    def _prepare(self, const_metadata: arrus.metadata.ConstMetadata):
        return const_metadata

    def __call__(self, data):
        data = data - self.xp.min(data)
        data = data/self.xp.max(data)*255
        return data.astype(self.xp.uint8)


def _get_rx_aperture_origin(sequence):
    rx_aperture_size = sequence.rx_aperture_size
    rx_aperture_center_element = sequence.rx_aperture_center_element
    rx_aperture_origin = np.round(rx_aperture_center_element -
                               (rx_aperture_size - 1) / 2 + 1e-9)
    return rx_aperture_origin


