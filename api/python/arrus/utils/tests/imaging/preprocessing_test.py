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
    ScanConversion,
)


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
        np.testing.assert_equal(result, 0j)

    def test_1(self):
        # Given
        data = [1]
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, 2+0j)

    def test_m1(self):
        # Given
        data = [-1]
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, -2+0j)

    def test_1D(self):
        ''' Test use vector data.'''

        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)

        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
             + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, expected)

    def test_2D(self):
        ''' Test use 2D array data.'''

        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (10,2))

        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        # Run
        result = self.run_op(data=data)
        # Expect
        np.testing.assert_equal(result, expected)

    def test_3D(self):
        ''' Test use 3D array data.'''

        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (13,11,3))

        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        expected = np.squeeze(m * data)
        # Run
        result = self.run_op(data=data)
        # Expect
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

        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (13, 11, 3))

        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (2 * np.cos(-2 * np.pi * fc * t)
             + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        data = np.squeeze(m * data)
        expected = np.abs(data)
        # Run
        result = self.run_op(data=data)
        # Expect
        self.assertTrue(np.all(result >= 0))

    def test_envelope(self):
        ''' Test use 3D array data.'''

        fs = self.context.device.sampling_frequency
        fc = self.context.sequence.pulse.center_frequency

        # Given
        data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
        data = np.tile(data, (13,11,3))

        n_samples = np.shape(data)[-1]
        t = (np.arange(0, n_samples) / fs)
        m = (  2 * np.cos(-2 * np.pi * fc * t)
               + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
        m = m.astype(np.complex64)
        data = np.squeeze(m * data)
        expected = np.abs(data)
        # Run
        result = self.run_op(data=data)
        # Expect
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
        expected = 20*np.log10(data)
        # Run
        result = self.run_op(data=data)
        # Expect
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
        expected = np.array([2,2,2,3,3])
        # Run
        result = self.run_op(data=data,min=2,max=3)
        # Expect
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
        data = np.arange(-10.,1024)
        # expected = data - np.min(data)
        # expected = expected/np.max(expected)*255
        # Run
        result = self.run_op(data=data)
        # Expect
        self.assertTrue((np.all(result >= 0))
                        and (np.all(result <= 255))
                        and result.dtype == 'uint8')

    def test_is_correct(self):
        # Given
        data = np.arange(-128, 128)
        expected = data - np.min(data)
        expected = expected/np.max(expected)*255
        # Run
        result = self.run_op(data=data)
        # Expect
        # self.assertEqual(expected, result)
        np.testing.assert_equal(expected, result)


class ScanConversionLinearArrayTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        sequence = LinSequence(
            tx_aperture_center_element = np.arange(0,65,8),
            tx_aperture_size = 32,
            rx_aperture_center_element = np.arange(0,65,8),
            rx_aperture_size = 32,
            tx_focus = 50e-6,
            pulse=Pulse(center_frequency=6e6, n_periods=2,
                        inverse=False),
            rx_sample_range=(0, 2048),
            downsampling_factor=1,
            speed_of_sound=1490,
            pri=100e-6,
            sri=50e-3,
            tgc_start=0,
            tgc_slope=12,
            # init_delay='tx_start'
        )
        self.op = ScanConversion
        self.context = self.get_default_context(sequence=sequence)

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 2:
            dim_diff = 2-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    def test_identic(self):
        # Given
        pitch = self.context.device.probe.model.pitch
        fs = self.context.device.sampling_frequency
        n_elements = self.context.device.probe.model.n_elements
        probe_width = (n_elements-1)*pitch
        c = self.context.sequence.speed_of_sound
        txapcel = self.context.sequence.tx_aperture_center_element
        n_scanlines = len(txapcel)
        sample_range = self.context.sequence.rx_sample_range
        n_samples = sample_range[1] - sample_range[0]
        dz = c/fs/2
        zmax = (sample_range[1] - 1)*dz
        zmin = sample_range[0]*dz

        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid = np.linspace(-probe_width/2, probe_width/2, nx_grid_samples)
        z_grid = np.linspace(zmin, zmax, nz_grid_samples)
        data = np.arange(n_scanlines)
        expected = data
        data = np.tile(data, (n_samples, 1))
        expected = np.tile(expected, (nz_grid_samples,1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid,
                             )

        # Expect
        np.testing.assert_equal(expected, result)

    def test_lininterp(self):
        # Given
        pitch = self.context.device.probe.model.pitch
        fs = self.context.device.sampling_frequency
        n_elements = self.context.device.probe.model.n_elements
        probe_width = (n_elements - 1) * pitch
        c = self.context.sequence.speed_of_sound
        txapcel = self.context.sequence.tx_aperture_center_element
        n_scanlines = len(txapcel)
        sample_range = self.context.sequence.rx_sample_range
        n_samples = sample_range[1] - sample_range[0]
        dz = c / fs / 2
        zmax = (sample_range[1] - 1) * dz
        zmin = sample_range[0] * dz

        nx_grid_samples = n_scanlines*2
        nz_grid_samples = n_samples
        x_grid = np.linspace(-probe_width / 2, probe_width / 2, nx_grid_samples)
        z_grid = np.linspace(zmin, zmax, nz_grid_samples)

        data = np.arange(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        expected = np.arange(0,n_scanlines,0.5)
        expected = np.tile(expected, (n_samples, 1))
        expected = expected.astype(int)

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid,
                             )
        # print('data:')
        # print(data)
        # print('result:')
        # print(result)
        # print('expected: ')
        # print(expected)

        # Expect
        np.testing.assert_almost_equal(expected, result, decimal=0)



class ScanConversionConvexArrayTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        sequence = LinSequence(
            tx_aperture_center_element = np.arange(0,65,8),
            tx_aperture_size = 32,
            rx_aperture_center_element = np.arange(0,65,8),
            rx_aperture_size = 32,
            tx_focus = 50e-6,
            pulse=Pulse(center_frequency=6e6, n_periods=2,
                        inverse=False),
            rx_sample_range=(0, 2048),
            downsampling_factor=1,
            speed_of_sound=1490,
            pri=100e-6,
            sri=50e-3,
            tgc_start=0,
            tgc_slope=12,
            # init_delay='tx_start'
        )

        device = self.get_ultrasound_device(
            probe=self.get_probe_model_instance(
                n_elements=64,
                pitch=0.2e-3,
                curvature_radius=0.1
            ),
            sampling_frequency=65e6
        )
        self.op = ScanConversion
        self.context = self.get_default_context(sequence=sequence, device=device)

    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 2:
            dim_diff = 2-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)

    def test_identic(self):
        # Given
        pitch = self.context.device.probe.model.pitch
        fs = self.context.device.sampling_frequency
        n_elements = self.context.device.probe.model.n_elements
        probe_width = (n_elements-1)*pitch
        c = self.context.sequence.speed_of_sound
        txapcel = self.context.sequence.tx_aperture_center_element
        n_scanlines = len(txapcel)
        sample_range = self.context.sequence.rx_sample_range
        n_samples = sample_range[1] - sample_range[0]
        dz = c/fs/2
        zmax = (sample_range[1] - 1)*dz
        zmin = sample_range[0]*dz

        nx_grid_samples = n_scanlines
        nz_grid_samples = 8
        x_grid = np.linspace(-probe_width/2, probe_width/2, nx_grid_samples)
        z_grid = np.linspace(zmin, zmax, nz_grid_samples)
        data = np.arange(n_scanlines)
        expected = data
        data = np.tile(data, (n_samples, 1))
        expected = np.tile(expected, (nz_grid_samples,1))

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid,
                             )

        print('data:')
        print(data)
        print('result:')
        print(result)
        print('expected: ')
        print(expected)
        # Expect
        np.testing.assert_equal(expected, result)

    def test_lininterp(self):
        # Given
        pitch = self.context.device.probe.model.pitch
        fs = self.context.device.sampling_frequency
        n_elements = self.context.device.probe.model.n_elements
        probe_width = (n_elements - 1) * pitch
        c = self.context.sequence.speed_of_sound
        txapcel = self.context.sequence.tx_aperture_center_element
        n_scanlines = len(txapcel)
        sample_range = self.context.sequence.rx_sample_range
        n_samples = sample_range[1] - sample_range[0]
        dz = c / fs / 2
        zmax = (sample_range[1] - 1) * dz
        zmin = sample_range[0] * dz

        nx_grid_samples = n_scanlines*2
        nz_grid_samples = n_samples
        x_grid = np.linspace(-probe_width / 2, probe_width / 2, nx_grid_samples)
        z_grid = np.linspace(zmin, zmax, nz_grid_samples)

        data = np.arange(n_scanlines)
        data = np.tile(data, (n_samples, 1))

        expected = np.arange(0,n_scanlines,0.5)
        expected = np.tile(expected, (n_samples, 1))
        expected = expected.astype(int)

        # Run
        result = self.run_op(data=data,
                             x_grid=x_grid,
                             z_grid=z_grid,
                             )
        # print('data:')
        # print(data)
        # print('result:')
        # print(result)
        # print('expected: ')
        # print(expected)

        # Expect
        np.testing.assert_almost_equal(expected, result, decimal=0)



if __name__ == "__main__":
    unittest.main()
