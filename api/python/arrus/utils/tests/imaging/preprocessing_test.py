import unittest
import numpy as np
from arrus.ops.imaging import LinSequence
from arrus.ops.us4r import Pulse
from arrus.utils.tests.utils import ArrusImagingTestCase
from arrus.utils.imaging import (
    QuadratureDemodulation,
    EnvelopeDetection,
    LogCompression,
    DynamicRangeAdjustment,
    ToGrayscaleImg,
    ScanConversion)


def get_n_scanlines(self):
    txapcel = self.context.sequence.tx_aperture_center_element
    n_scanlines = len(txapcel)
    return n_scanlines


def get_sample_range(self):
    return self.context.sequence.rx_sample_range


def get_n_samples(self):
    sample_range = get_sample_range(self)
    n_samples = sample_range[1] - sample_range[0]
    return n_samples


def get_grid_data(self, nx_grid_samples, nz_grid_samples):
    pitch = self.context.device.probe.model.pitch
    fs = self.context.device.sampling_frequency
    n_elements = self.context.device.probe.model.n_elements
    probe_width = (n_elements - 1) * pitch
    c = self.context.sequence.speed_of_sound
    txapcel = self.context.sequence.tx_aperture_center_element
    n_scanlines = get_n_scanlines(self)
    sample_range = get_sample_range(self)
    n_samples = get_n_samples(self)
    dz = c / fs / 2
    zmax = (sample_range[1] - 1) * dz
    zmin = sample_range[0] * dz
    x_grid = np.linspace(-probe_width / 2, probe_width / 2, nx_grid_samples)
    z_grid = np.linspace(zmin, zmax, nz_grid_samples)
    return x_grid, z_grid


class QuadratureDemodulationTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = QuadratureDemodulation
        self.context = self.get_default_context()

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 3:
            dim_diff = 3-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    # Corner cases:
    def test_no_input_signal(self):
        """Empty input array should not be accepted. """
        with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
            self.run_op(data=[])

    def test_0(self):
        # Given
        data = [0]
        # Run
        result = self.run_op(data=data)
        # Expect
        expected = 0j
        np.testing.assert_equal(result, expected)

    def test_1(self):
        # Given
        data = [1]
        # Run
        result = self.run_op(data=data)
        # Expect
        expected = 2+0j
        np.testing.assert_equal(result, expected)

    def test_negative1(self):
        # Given
        data = [-1]
        # Run
        result = self.run_op(data=data)
        # Expect
        expected = -2+0j
        np.testing.assert_equal(result, expected)

    def test_1D(self):
        ''' Test uses vector data.'''

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)

        # Run
        result = self.run_op(data=data)

        # Expect
        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency
        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        np.testing.assert_equal(result, expected)

    def test_2D(self):
        ''' Test uses 2D array data.'''

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (10, 2))

        # Run
        result = self.run_op(data=data)

        # Expect
        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency
        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        np.testing.assert_equal(result, expected)

    def test_3D(self):
        ''' Test uses 3D array data.'''

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (13, 11, 3))

        # Run
        result = self.run_op(data=data)

        # Expect
        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency
        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        np.testing.assert_equal(result, expected)


class EnvelopeDetectionTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = EnvelopeDetection
        self.context = self.get_default_context()

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 3:
            dim_diff = 3-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    # Corner cases:
    def test_no_input_signal(self):
        """Empty input array should not be accepted. """
        with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
            self.run_op(data=[])

    def test_is_positive(self):
        # Given
        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (13, 11, 3))
        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (2 * np.cos(-2 * np.pi * fc * t)
             + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        data = np.squeeze(m * data)

        # Run
        result = self.run_op(data=data)

        # Expect
        self.assertTrue(np.all(result >= 0))

    def test_envelope(self):
        ''' Test uses 3D array data.'''

        # Given
        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (13, 11, 3))
        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        data = np.squeeze(m * data)

        # Run
        result = self.run_op(data=data)

        # Expect
        expected = np.abs(data)
        np.testing.assert_almost_equal(result, expected, decimal=5)


class LogCompressionTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = LogCompression
        self.context = self.get_default_context()

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 3:
            dim_diff = 3-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    # Corner cases:
    def test_no_input_signal(self):
        """Empty input array should not be accepted. """
        with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
            self.run_op(data=[])

    def test_log(self):
        # Given
        data = np.arange(10) + 1e-9
        # Run
        result = self.run_op(data=data)
        # Expect
        expected = 20*np.log10(data)
        np.testing.assert_almost_equal(result, expected, decimal=12)


class DynamicRangeAdjustmentTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = DynamicRangeAdjustment
        self.context = self.get_default_context()

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 3:
            dim_diff = 3-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    def test_clip(self):
        # Given
        data = np.arange(5)
        # Run
        result = self.run_op(data=data,min=2,max=3)
        # Expect
        expected = np.array([2, 2, 2, 3, 3])
        np.testing.assert_equal(result, expected)


class ToGrayscaleImgTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = ToGrayscaleImg
        self.context = self.get_default_context()

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 3:
            dim_diff = 3-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    def test_is_in_range(self):
        # Given
        data = np.arange(-10, 1024)
        # Run
        result = self.run_op(data=data)
        # Expect
        self.assertTrue((np.all(result >= 0))
                        and (np.all(result <= 255))
                        and result.dtype == 'uint8')

    def test_is_correct(self):
        # Given
        data = np.arange(-128, 128)
        # Run
        result = self.run_op(data=data)
        # Expect
        expected = data - np.min(data)
        expected = expected/np.max(expected)*255
        np.testing.assert_equal(expected, result)


class AbstractScanConversionTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        sequence = LinSequence(
            tx_aperture_center_element=np.arange(0, 65, 8),
            tx_aperture_size=32,
            rx_aperture_center_element=np.arange(0, 65, 8),
            rx_aperture_size=32,
            tx_focus=50e-6,
            pulse=Pulse(center_frequency=6e6, n_periods=2,
                        inverse=False),
            rx_sample_range=(0, 2048),
            downsampling_factor=1,
            speed_of_sound=1490,
            pri=100e-6,
            sri=50e-3,
            tgc_start=0,
            tgc_slope=12,
        )
        self.op = ScanConversion
        self.context = self.get_default_context(sequence=sequence)

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 2:
            raise ValueError("Currently data supports at most 2 dimensions.")
        if len(data.shape) < 2:
            dim_diff = 2-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)


class ScanConversionLinearArrayTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        sequence = LinSequence(
            tx_aperture_center_element=np.arange(0, 65, 8),
            tx_aperture_size=32,
            rx_aperture_center_element=np.arange(0, 65, 8),
            rx_aperture_size=32,
            tx_focus=50e-6,
            pulse=Pulse(center_frequency=6e6, n_periods=2,
                        inverse=False),
            rx_sample_range=(0, 2048),
            downsampling_factor=1,
            speed_of_sound=1490,
            pri=100e-6,
            sri=50e-3,
            tgc_start=0,
            tgc_slope=12,
        )
        self.op = ScanConversion
        self.context = self.get_default_context(sequence=sequence)

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 2:
            raise ValueError("Currently data supports at most 2 dimensions.")
        if len(data.shape) < 2:
            dim_diff = 2-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    def test_identity(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        data = np.arange(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid)

        # Expect
        expected = np.arange(n_scanlines)
        expected = np.tile(expected, (nz_grid_samples, 1))
        np.testing.assert_equal(expected, result)

    def test_extrapolation_left(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        x_grid_moved_left = x_grid - (x_grid[-1]-x_grid[0])
        data = np.arange(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid_moved_left,
                             z_grid=z_grid)
        # Expect
        expected = np.zeros((nz_grid_samples, n_scanlines))
        np.testing.assert_equal(expected, result)

    def test_extrapolation_right(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        x_grid_moved_right = x_grid + (x_grid[-1] - x_grid[0] + x_grid[1] - x_grid[0])
        data = np.arange(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid_moved_right,
                             z_grid=z_grid)

        # Expect
        expected = np.zeros((nz_grid_samples, n_scanlines))
        np.testing.assert_equal(expected, result)

    def test_extrapolation_up(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        z_grid_moved_up = z_grid - z_grid[-1] - z_grid[1] + z_grid[0]
        data = np.arange(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid_moved_up)

        # Expect
        expected = np.zeros((nz_grid_samples, n_scanlines))
        np.testing.assert_equal(expected, result)

    def test_extrapolation_down(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        z_grid_moved_down = z_grid + z_grid[-1] + z_grid[1] - z_grid[0]
        data = np.arange(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid_moved_down)

        # Expect
        expected = np.zeros((nz_grid_samples, n_scanlines))
        np.testing.assert_equal(expected, result)

    def test_linear_interpolation_in_x_axis(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines * 2
        nz_grid_samples = n_samples
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        data = np.arange(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid)

        # Expect
        expected = np.arange(0, n_scanlines, 0.5)
        expected = np.tile(expected, (n_samples, 1))
        expected = expected.astype(int)
        np.testing.assert_almost_equal(expected, result, decimal=0)

    def test_linear_interpolation_in_z_axis(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = n_samples * 2
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        data = np.arange(n_samples)[..., np.newaxis]
        data = np.tile(data, (1, n_scanlines))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid)

        # Expect
        expected = np.arange(0, n_samples, 0.5)[..., np.newaxis]
        expected = np.tile(expected, (1, n_scanlines))
        expected = expected.astype(int)
        np.testing.assert_almost_equal(expected, result, decimal=0)

    def test_zeros(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        data = np.zeros(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid)

        # Expect
        expected = np.zeros(n_scanlines)
        expected = np.tile(expected, (nz_grid_samples,1))
        np.testing.assert_equal(expected, result)


class ScanConversionConvexArrayTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        sequence = LinSequence(
            tx_aperture_center_element=np.arange(0, 65, 8),
            tx_aperture_size=32,
            rx_aperture_center_element=np.arange(0, 65, 8),
            rx_aperture_size=32,
            tx_focus=50e-6,
            pulse=Pulse(center_frequency=6e6,
                        n_periods=2,
                        inverse=False),
            rx_sample_range=(32, 64),
            downsampling_factor=1,
            speed_of_sound=1490,
            pri=100e-6,
            sri=50e-3,
            tgc_start=0,
            tgc_slope=12)

        device = self.get_ultrasound_device(
            probe=self.get_probe_model_instance(
                n_elements=64,
                pitch=0.2e-3,
                curvature_radius=0.1),
            sampling_frequency=65e6
        )
        self.op = ScanConversion
        self.context = self.get_default_context(sequence=sequence, device=device)

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 2:
            raise ValueError("Currently data supports at most 2 dimensions.")
        if len(data.shape) < 2:
            dim_diff = 2 - len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    def test_zeros(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        data = np.zeros(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid)

        # Expect
        expected = np.zeros(n_scanlines)
        expected = np.tile(expected, (nz_grid_samples, 1))
        np.testing.assert_equal(expected, result)

    def test_onesbelt(self):
        # Given
        n_scanlines = get_n_scanlines(self)
        n_samples = get_n_samples(self)
        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid, z_grid = get_grid_data(self,
                                       nx_grid_samples,
                                       nz_grid_samples)
        data = np.zeros(n_scanlines)
        data = np.tile(data, (n_samples, 1))
        data[0:8, :] = np.ones(n_scanlines)

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid)

        # Expect
        expected = np.zeros(n_scanlines)
        expected = np.tile(expected, (nz_grid_samples,1))
        expected[1, 0] = 1
        expected[1, 8] = 1
        expected[2, 1] = 1
        expected[2, 7] = 1
        expected[3, 1:3] = 1
        expected[3, 6:8] = 1
        expected[4, 2:7] = 1
        expected[5, 3:6] = 1
        np.testing.assert_equal(expected, result)


if __name__ == "__main__":
    unittest.main()
