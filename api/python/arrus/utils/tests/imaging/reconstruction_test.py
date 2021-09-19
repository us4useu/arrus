import matplotlib.pyplot as plt
import unittest
import numpy as np

from arrus.utils.tests.utils import ArrusImagingTestCase
from arrus.utils.imaging import (
    ReconstructLri,
    RxBeamforming)


def get_max_ndx(data):
    s = data.shape
    ix = np.argmax(data)
    return np.unravel_index(ix, s)

def get_wire_indexes(wire_coords, x_grid, z_grid,):
    x = wire_coords[0]
    z = wire_coords[1]
    xi = np.abs(x_grid - x).argmin(axis=0)
    zi = np.abs(z_grid - z).argmin(axis=0)
    return (xi, zi)


def get_system_parameters(context):
    '''
    Auxiliary tool for pull out selected probe parameters.
    '''
    fs = context.device.sampling_frequency
    probe = context.device.probe.model
    n_elements = probe.n_elements
    pitch = probe.pitch
    curvature_radius = probe.curvature_radius
    return fs, n_elements, pitch, curvature_radius

def show_image(data):

    ncol, nsamp = np.shape(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data)

    #aspect = 1/np.round(nsamp/ncol).astype(int)
    ax.set_aspect('auto')
    plt.show()



def get_lin_coords(n_elements=128, pitch=0.2*1e-3):
    '''
    Auxiliary tool for generating array transducer elements coordinates for linear array.

    :param nel: number of elements,
    :param pitch: distance between elements,
    :return: numpy array with elements coordinates (x,z)
    '''
    elx = np.linspace(-(n_elements-1)*pitch/2, (n_elements-1)*pitch/2, n_elements)
    elz = np.zeros(n_elements)
    coords = np.array(list(zip(elx,elz)))
    return coords


def gen_data(el_coords=None, dels=None, wire_coords=None,
             c=1540, fs=65e6, wire_amp=100):
    '''
    Function for generation of artificial non-beamformed data
    corresponding to single point (wire) within empty medium.

    :param el_coords: coordinates of transducer elements (numpy array),
    :param dels: initial delays,
    :param wire_coords: wire coordinates,
    :param c: speed of sound,
    :param fs: sampling frequency,
    :param wire_amp: amplitude of the wire
    :return: 2D numpy array of zeros and single pixel
             with amplitude equal to 'wire_amp' parameter.
    '''

    # check input and get default parameters if needed
    if el_coords is None:
        el_coords = get_lin_coords()

    nel, _  = np.shape(el_coords)
    if dels is None:
        dels = np.zeros(nel)

    if wire_coords is None:
        wire_coords = (0, 5*1e-3)

    # estimate distances between transducer elements and the 'wire'
    dist = np.zeros(nel)
    for i in range(nel):
        dist[i] = np.sqrt((el_coords[i, 0]-wire_coords[0])**2
                         +(el_coords[i, 1]-wire_coords[1])**2)
    # create output array
    nsamp = np.floor((2*dist/c + dels)*fs + 1).astype(int)
    nmax = 2*np.max(nsamp)
    data = np.zeros((nel,nmax))
    for i in range(nel):
        data[i, nsamp[i]] = wire_amp


    return data



class PwiReconstrutionTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = ReconstructLri
        self.context = self.get_default_context()
        fs, n_elements, pitch, curvature_radius = get_system_parameters(self.context)
        self.x_grid = np.linspace(-5*1e-3, 5*1e-3, 128)
        self.z_grid = np.linspace(0*1e-3, 10.*1e-3, 256)

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
        fs, n_elements, pitch, curvature_radius = get_system_parameters(self.context)

        return np.squeeze(result)

    # Corner cases:
#    def test_empty_x_grid(self):
#        """Empty input array should not be accepted. """
#        with self.assertRaisesRegex(ValueError, "Empty array") as ctx:
#            #pass
#        #    self.run_op(data=[], x_grid=[], z_grid=[])
#            self.run_op(data=0, x_grid=[], z_grid=self.z_grid)



    def test_empty(self):
        # Given
        data = []
        # Run
        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
        # Expect
        expected_shape = (self.x_grid.size, self.z_grid.size )
        expected = np.zeros(expected_shape, dtype=complex)
        np.testing.assert_equal(result, expected)

    def test_0(self):
        # Given
        data = 0
        # Run
        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
        # Expect
        expected_shape = (self.x_grid.size, self.z_grid.size )
        expected = np.zeros(expected_shape, dtype=complex)
        np.testing.assert_equal(result, expected)

    def test_pwi_angle0(self):
        # Given
        wire_coords = (0, 5*1e-3)
        fs, n_elements, pitch, curvature_radius = get_system_parameters(self.context)
        el_coords = get_lin_coords(n_elements=n_elements, pitch=pitch)
        data = gen_data(el_coords=el_coords,
                         dels=None,
                         wire_coords=wire_coords,
                         c=1540,
                         fs=65e6,
                         wire_amp=100)

        # Run
        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
        result = np.abs(result)

        show_image(np.abs(result.T))
        show_image(data)

        # Expect
        ix, iz = get_wire_indexes(wire_coords, self.x_grid, self.z_grid)
        i, j = get_max_ndx(result)
        #TODO kiedy jest instrukcja print, to result jest równy wszędzie 0?
        print(i)
        print(j)
        #print(f'wire coordinates: {wire_coords}')
        ##print(f'x_grid: {self.x_grid}')
        #print(f'value at estimated wire indexes in reconstructed image: {result[ix, iz]}')
        #print(f'max value in reconstructed image: {np.nanmax(result)}')

        #expected_shape = (self.x_grid.size, self.z_grid.size)
        #expected = np.zeros(expected_shape, dtype=complex)
        #np.testing.assert_equal(result, expected)






    # def test_1(self):
    #     # Given
    #     data = [1]
    #     # Run
    #     result = self.run_op(data=data)
    #     # Expect
    #     expected = 2+0j
    #     np.testing.assert_equal(result, expected)

    # def test_negative1(self):
    #     # Given
    #     data = [-1]
    #     # Run
    #     result = self.run_op(data=data)
    #     # Expect
    #     expected = -2+0j
    #     np.testing.assert_equal(result, expected)

    # def test_1D(self):
    #     ''' Test uses vector data.'''

    #     # Given
    #     data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)

    #     # Run
    #     result = self.run_op(data=data)

    #     # Expect
    #     fs = self.context.device.sampling_frequency
    #     fc = self.context.sequence.pulse.center_frequency
    #     n_samples = np.shape(data)[-1]
    #     t = (np.arange(0, n_samples) / fs)
    #     m = (  2 * np.cos(-2 * np.pi * fc * t)
    #            + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
    #     m = m.astype(np.complex64)
    #     expected = np.squeeze(m * data)
    #     np.testing.assert_equal(result, expected)

    # def test_2D(self):
    #     ''' Test uses 2D array data.'''

    #     # Given
    #     data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
    #     data = np.tile(data, (10, 2))

    #     # Run
    #     result = self.run_op(data=data)

    #     # Expect
    #     fs = self.context.device.sampling_frequency
    #     fc = self.context.sequence.pulse.center_frequency
    #     n_samples = np.shape(data)[-1]
    #     t = (np.arange(0, n_samples) / fs)
    #     m = (  2 * np.cos(-2 * np.pi * fc * t)
    #            + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
    #     m = m.astype(np.complex64)
    #     expected = np.squeeze(m * data)
    #     np.testing.assert_equal(result, expected)

    # def test_3D(self):
    #     ''' Test uses 3D array data.'''

    #     # Given
    #     data = np.array([-1., 10, 0, -20, 1]).astype(np.float32)
    #     data = np.tile(data, (13, 11, 3))

    #     # Run
    #     result = self.run_op(data=data)

    #     # Expect
    #     fs = self.context.device.sampling_frequency
    #     fc = self.context.sequence.pulse.center_frequency
    #     n_samples = np.shape(data)[-1]
    #     t = (np.arange(0, n_samples) / fs)
    #     m = (  2 * np.cos(-2 * np.pi * fc * t)
    #            + 2 * np.sin(-2 * np.pi * fc * t) * 1j)
    #     m = m.astype(np.complex64)
    #     expected = np.squeeze(m * data)
    #     np.testing.assert_equal(result, expected)


if __name__ == "__main__":
    unittest.main()

