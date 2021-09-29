import matplotlib.pyplot as plt
import unittest
import numpy as np
import cupy as cp
from arrus.utils.tests.utils import ArrusImagingTestCase
from arrus.ops.us4r import Scheme, Pulse
from arrus.ops.imaging import PwiSequence
from arrus.utils.imaging import get_bmode_imaging, get_extent
from arrus.utils.imaging import (
    ReconstructLri,
    RxBeamforming)


def get_max_ndx(data):
    '''
    The function returns indexes of max value in array.
    '''
    s = data.shape
    ix = np.nanargmax(data)
    return np.unravel_index(ix, s)


#def get_wire_indexes(wire_coords, x_grid, z_grid,):
#    x = wire_coords[0]
#    z = wire_coords[1]
#    xi = np.abs(x_grid - x).argmin(axis=0)
#    zi = np.abs(z_grid - z).argmin(axis=0)
#    return (xi, zi)



def show_image(data):
    '''
    Simple function for showing array image.
    '''
    #ncol, nsamp = np.shape(data)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data)
    ax.set_aspect('auto')
    plt.show()


#def get_lin_coords(n_elements=128, pitch=0.2*1e-3):
#    '''
#    Auxiliary tool for generating array transducer elements coordinates for linear array.
#
#    :param nel: number of elements,
#    :param pitch: distance between elements,
#    :return: numpy array with elements coordinates (x,z)
#    '''
#    elx = np.linspace(-(n_elements-1)*pitch/2, (n_elements-1)*pitch/2, n_elements)
#    elz = np.zeros(n_elements)
#    coords = np.array(list(zip(elx,elz)))
#    return coords


#def gen_data(el_coords=None, dels=None, wire_coords=(0, 5*1e-3),
#             c=1540, fs=65e6, wire_amp=100, wire_diameter=10):
#    '''
#    Function for generation of artificial non-beamformed data
#    corresponding to single point (wire) within empty medium.
#
#    :param el_coords: coordinates of transducer elements (numpy array),
#    :param dels: initial delays,
#    :param wire_coords: wire coordinates,
#    :param c: speed of sound,
#    :param fs: sampling frequency,
#    :param wire_amp: amplitude of the wire
#    :return: 2D numpy array of zeros and single pixel
#             with amplitude equal to 'wire_amp' parameter.
#    '''
#
#    # check input and get default parameters if needed
#    if el_coords is None:
#        el_coords = get_lin_coords()
#
#    nel, _  = np.shape(el_coords)
#    if dels is None:
#        dels = np.zeros(nel)
#
#    #if wire_coords is None:
#    #    wire_coords = (0, 5*1e-3)
#
#    # estimate distances between transducer elements and the 'wire'
#    dist = np.zeros(nel)
#    for i in range(nel):
#        dist[i] = np.sqrt((el_coords[i, 0]-wire_coords[0])**2
#                         +(el_coords[i, 1]-wire_coords[1])**2)
#    # create output array
#    nsamp = np.floor((2*dist/c + dels)*fs + 1).astype(int)
#    nmax = 2*np.max(nsamp)
#    data = np.zeros((nel,nmax))
#    for i in range(nel):
#        start = nsamp[i] - wire_diameter
#        stop = nsamp[i] + wire_diameter
#        data[i, start:stop] = wire_amp
#
#    return data


class PwiReconstrutionTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = ReconstructLri
        #self.context = self.get_default_context()
        self.context = self.get_pwi_context(angle=0)
        n_elements = self.get_system_parameter('n_elements')
        pitch = self.get_system_parameter('pitch')
        probe_width = (n_elements-1)*pitch
        fs = self.get_system_parameter('sampling_frequency')
        c = self. get_system_parameter('speed_of_sound')
        ds = c/fs
        ncol = np.round(probe_width/ds).astype(int)+1
        ztop = 0
        zbot = 50*1e-3
        nrow = np.round((zbot - ztop)/ds + 1).astype(int)
        self.x_grid = np.linspace(-probe_width/2, probe_width/2, ncol)
        self.z_grid = np.linspace(ztop, zbot , nrow)


    def run_op(self, **kwargs):
        data = kwargs['data']
        data = np.array(data).astype(np.complex64)
        if len(data.shape) > 3:
            raise ValueError("Currently data supports at most 3 dimensions.")
        if len(data.shape) < 3:
            dim_diff = 3-len(data.shape)
            data = np.expand_dims(data, axis=tuple(np.arange(dim_diff)))
            kwargs["data"] = data
        result = super().run_op(**kwargs)
        return np.squeeze(result)


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


#    def test_pwi_angle0_wire_in_the_middle(self):
#        # Given
#        wire_coords = (0, 5*1e-3)
#
#        #el_coords = self.get_lin_coords()
#        data = self.gen_data(txdelays=None,
#                             wire_coords=wire_coords,
#                             wire_amp=100,
#                             wire_radius=4)
#
#        # Run
#        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
#        result = np.abs(result)
#        # show_image(np.abs(result.T))
#        # show_image(data)
#
#        # Expect
#        # Indexes corresponding to wire coordinates
#        iwire, jwire = self.get_wire_indexes(wire_coords)
#
#        # indexes corresponding to max value of beamformed amplitude image
#        i, j = get_max_ndx(result)
#
#        # information about indexes (for debugging)
#        #print(f'expected wire row index value: {iwire}')
#        #print(f'obtained wire row index value: {i}')
#        #print(f'expected wire column index value: {jwire}')
#        #print(f'obtained wire column index valuej: {j}')
#
#        # (arbitrary) tolerances for indexes of maximum value in beamformed image
#        xtol = 8
#        ztol = 8
#
#        idiff = np.abs(iwire-i)
#        jdiff = np.abs(jwire-j)
#        self.assertLessEqual(idiff, xtol)
#        self.assertLessEqual(jdiff, ztol)

#
#    def test_pwi_angle0_wire_on_the_left(self):
#        # Given
#        wire_coords = (-5*1e-3, 20*1e-3)
#
#        #el_coords = self.get_lin_coords()
#        data = self.gen_data(txdelays=None,
#                             wire_coords=wire_coords,
#                             wire_amp=100,
#                             wire_radius=4)
#
#        # Run
#        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
#        result = np.abs(result)
#        #show_image(np.abs(result.T))
#        # show_image(data)
#
#        # Expect
#        # Indexes corresponding to wire coordinates
#        #iwire, jwire = get_wire_indexes(wire_coords, self.x_grid, self.z_grid)
#        iwire, jwire = self.get_wire_indexes(wire_coords)
#
#        # indexes corresponding to max value of beamformed amplitude image
#        i, j = get_max_ndx(result)
#
#        # information about indexes (for debugging)
#        #print(f'expected wire row index value: {iwire}')
#        #print(f'obtained wire row index value: {i}')
#        #print(f'expected wire column index value: {jwire}')
#        #print(f'obtained wire column index valuej: {j}')
#
#        # (arbitrary) tolerances for indexes of maximum value in beamformed image
#        xtol = 8
#        ztol = 8
#
#        idiff = np.abs(iwire-i)
#        jdiff = np.abs(jwire-j)
#        self.assertLessEqual(idiff, xtol)
#        self.assertLessEqual(jdiff, ztol)





    def test_pwi_angle0(self):
        # Given
        angle = 0
        self.context = self.get_pwi_context(angle=angle)
        max_x = np.max(self.x_grid)
        min_x = np.min(self.x_grid)
        max_z = np.max(self.z_grid)
        min_z = np.min(self.z_grid)
        xmargin = (max_x-min_x)*0.1
        zmargin = (max_z-min_z)*0.1
        wire_x = np.arange(min_x+xmargin, max_x-xmargin, 1)
        wire_z = np.arange(min_z+zmargin, max_z-zmargin, 5)

        for x in wire_x:
            for z in wire_z:
                wire_coords = (x, z)
                data = self.gen_pwi_data(angle=angle,
                                     wire_coords=wire_coords,
                                     wire_amp=100,
                                     wire_radius=4)
                # Run
                result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
                result = np.abs(result)
                show_image(np.abs(result.T))
                # show_image(data)

                # Expect
                # Indexes corresponding to wire coordinates in beamformed image
                iwire, jwire = self.get_wire_indexes(wire_coords)

                # indexes corresponding to max value of beamformed amplitude image
                i, j = get_max_ndx(result)

                # information about indexes (for debugging)
                print('----------------------------')
                print(f'current wire: ({x},{z})')
                print(f'expected wire row index value: {iwire}')
                print(f'obtained wire row index value: {i}')
                print(f'expected wire column index value: {jwire}')
                print(f'obtained wire column index valuej: {j}')
                print('')
                print('')

                # (arbitrary) tolerances for indexes of maximum value in beamformed image
                xtol = 16
                ztol = 16

                idiff = np.abs(iwire-i)
                jdiff = np.abs(jwire-j)
                self.assertLessEqual(idiff, xtol)
                self.assertLessEqual(jdiff, ztol)




#--------------------------------------------------------------------------
#                         TOOLS 
#--------------------------------------------------------------------------

    def get_pwi_context(self, angle):
        '''
        Function generate context data for pwi tests.
        '''
        sequence = PwiSequence(
            angles=np.array([angle])*np.pi/180,
            pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
            rx_sample_range=(0, 1024*4),
            downsampling_factor=1,
            speed_of_sound=1450,
            pri=200e-6,
            tgc_start=14,
            tgc_slope=2e2,
            )
        probe = self.get_probe_model_instance(
            n_elements=128,
            pitch=0.2e-3,
            curvature_radius=0.0
            )
        device = self.get_ultrasound_device(
            probe=probe,
            sampling_frequency=65e6
            )

        return self.get_default_context(
            sequence=sequence,
            device=device,
            )

    def get_wire_indexes(self, wire_coords):
        x = wire_coords[0]
        z = wire_coords[1]
        xi = np.abs(self.x_grid - x).argmin(axis=0)
        zi = np.abs(self.z_grid - z).argmin(axis=0)
        return (xi, zi)

    def get_lin_coords(self):
        '''
        Auxiliary tool for generating array transducer elements coordinates for linear array.

        :param nel: number of elements,
        :param pitch: distance between elements,
        :return: numpy array with elements coordinates (x,z)
        '''
        fs = self.get_system_parameter('sampling_frequency')
        n_elements = self.get_system_parameter('n_elements')
        pitch = self.get_system_parameter('pitch')

        elx = np.linspace(-(n_elements-1)*pitch/2, (n_elements-1)*pitch/2, n_elements)
        elz = np.zeros(n_elements)
        coords = np.array(list(zip(elx,elz)))
        return coords


    def get_lin_txdelays(self, angle):
        '''
        The functtion generate delays of PWI scheme for linear array.
        '''
        speed_of_sound = self.get_system_parameter('speed_of_sound')
        el_coords = self.get_lin_coords()
        delays = el_coords[:,0]*np.tan(angle)/speed_of_sound
        delays = delays - np.min(delays)
        return delays


    def gen_data(self, txdelays=None,
                 wire_coords=(0, 5*1e-3),
                 wire_amp=100,
                 wire_radius=10):
        '''
        Function for generation of artificial non-beamformed data
        corresponding to single point (wire) within empty medium.

        :param txdelays: initial delays,
        :param wire_coords: wire coordinates,
        :param wire_amp: amplitude of the wire
        :param wire_raduys: 'radius' of the wire
        :return: 2D numpy array of zeros and single pixel
                 with amplitude equal to 'wire_amp' parameter.
        '''

        c = self.get_system_parameter('speed_of_sound')
        fs = self.get_system_parameter('sampling_frequency')

        # check input and get default parameters if needed
        el_coords = self.get_lin_coords()
        nel, _  = np.shape(el_coords)
        if txdelays is None:
            txdelays = np.zeros(nel)

        # estimate distances between transducer elements and the 'wire'
        dist = np.zeros(nel)
        for i in range(nel):
            dist[i] = np.sqrt((el_coords[i, 0]-wire_coords[0])**2
                             +(el_coords[i, 1]-wire_coords[1])**2)
        # create output array
        nsamp = np.floor((2*dist/c + txdelays)*fs + 1).astype(int)
        nmax = 2*np.max(nsamp)
        data = np.zeros((nel,nmax))
        for i in range(nel):
            start = nsamp[i] - wire_radius
            stop = nsamp[i] + wire_radius
            data[i, start:stop] = wire_amp

        return data

    def gen_pwi_data(self, angle,
                     wire_coords=(0, 5*1e-3),
                     wire_amp=100,
                     wire_radius=10):
        txdelays = self.get_lin_txdelays(angle)
        data = self.gen_data(txdelays=txdelays,
                             wire_coords=wire_coords,
                             wire_amp=wire_amp,
                             wire_radius=wire_radius)


    def get_system_parameter(self, parameter):
        '''
        The function returns selected system.
        '''
        if parameter == 'sampling_frequency':
            return self.context.device.sampling_frequency
        if parameter == 'center_frequency':
            return self.context.sequence.pulse.center_frequency
        if parameter == 'n_elements':
            return self.context.device.probe.model.n_elements
        if parameter == 'pitch':
            return self.context.device.probe.model.pitch
        if parameter == 'curvature_radius':
            return self.context.device.probe.model.curvature_radius
        if parameter == 'speed_of_sound':
            return self.context.sequence.speed_of_sound




if __name__ == "__main__":
    unittest.main()

