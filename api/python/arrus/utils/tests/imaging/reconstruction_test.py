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



class PwiReconstrutionTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        self.op = ReconstructLri
        self.context = self.get_pwi_context(angle=0)
        n_elements = self.get_system_parameter('n_elements')
        pitch = self.get_system_parameter('pitch')
        probe_width = (n_elements-1)*pitch
        fs = self.get_system_parameter('sampling_frequency')
        c = self. get_system_parameter('speed_of_sound')
        ds = c/fs
        ncol = np.round(probe_width/ds).astype(int)+1
        ztop = 0
        zbot = 100*1e-3
        nrow = np.round((zbot - ztop)/ds + 1).astype(int)
        self.x_grid = np.linspace(-probe_width/2, probe_width/2, ncol)
        self.z_grid = np.linspace(ztop, zbot , nrow)
        # set arbitrary tolerances (in samples) i.e. possible differences
        # between expected and obtained maxima in b-mode image of a wire
        self.xtol = 32
        self.ztol = 32


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


    def test_pwi_angle0(self):
        # Given
        print_wire_info = 1
        angle = 0*np.pi/180
        self.context = self.get_pwi_context(angle=angle)
        max_x = np.max(self.x_grid)
        min_x = np.min(self.x_grid)
        max_z = np.max(self.z_grid)
        min_z = np.min(self.z_grid)
        xmargin = (max_x-min_x)*0.1
        zmargin = (max_z-min_z)*0.1
        wire_x = np.linspace(min_x+xmargin, max_x-xmargin, 10)
        wire_x = np.array([min_x+xmargin])
        wire_z = np.linspace(min_z+zmargin, max_z-zmargin, 10)

        self.angle = angle
        self.wire_amp = 100
        self.wire_radius = 1

        for x in wire_x:
            for z in wire_z:
                wire_coords = (x, z)
                self.wire_coords = wire_coords
                data = self.gen_pwi_data()
                #data = self.get_syntetic_data()
                # Run
                result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
                result = np.abs(result)
                #show_image(np.abs(result.T))
                #show_image(data)

                # Expect
                # Indexes corresponding to wire coordinates in beamformed image
                iwire, jwire = self.get_wire_indexes()

                # indexes corresponding to max value of beamformed amplitude image
                i, j = get_max_ndx(result)

                # information about indexes (for debugging)
                if print_wire_info:
                    print('----------------------------')
                    print(f'current wire: ({x},{z})')
                    print(f'max value in reult array: {np.nanmax(result)}')
                    print(f'expected wire row index value (x): {iwire}')
                    print(f'obtained wire row index value (x): {i}')
                    print(f'expected wire column index value (z): {jwire}')
                    print(f'obtained wire column index valuej (z): {j}')
                    print('')
                    print('')

                # (arbitrary) tolerances for indexes of maximum value in beamformed image

                idiff = np.abs(iwire-i)
                jdiff = np.abs(jwire-j)
                self.assertLessEqual(idiff, self.xtol)
                self.assertLessEqual(jdiff, self.ztol)




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

    def get_wire_indexes(self):
        x = self.wire_coords[0]
        z = self.wire_coords[1]
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


    def get_lin_txdelays(self):
        '''
        The functtion generate transmit delays of PWI scheme for linear array.
        '''
        speed_of_sound = self.get_system_parameter('speed_of_sound')
        el_coords = self.get_lin_coords()
        angle = self.angle
        delays = el_coords[:,0]*np.tan(angle)/speed_of_sound
        #delays = delays - np.min(delays)
        #delays = delays + np.max(delays)
        #delays = np.flipud(delays)
        return delays

    def get_delays(self):
        wire_coords = self.wire_coods
        speed_of_sound = get_system_parameter('speed_of_sound')
        el_coords = self.get_lin_coords
        txdelays = get_lin_txdelays()
        
        # estimate distances between transducer elements and the 'wire'
        dist = np.zeros(nel)
        for i in range(nel):
            dist[i] = np.sqrt((el_coords[i, 0]-wire_coords[0])**2
                             +(el_coords[i, 1]-wire_coords[1])**2)

        

    def get_pwi_txdelay(self):
        '''
        Function enumerate txdelay i.e. time between the first element excitation
        and the moment when wave front reach the wire.
        '''
        speed_of_sound = self.get_system_parameter('speed_of_sound')
        el_coords = self.get_lin_coords()
        angle = self.angle
        wire_x = self.wire_coords[0]
        wire_z = self.wire_coords[1]
        if angle < 0:
            xe = el_coords[0,0]
        else:
            xe = el_coords[-1,0]

        xe = 0
        # estimate txdelay i.e. from start of the transmission to moment 
        # when wave front reach the wire
        a = np.abs(angle)
        path = wire_z/np.cos(a)  \
               + (np.abs(wire_x-xe) \
                  - wire_z*np.tan(a))*np.sin(a)
        delay = path/speed_of_sound
        return delay

    def gen_data(self, txdelays=None):
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

        # get needed parameters
        angle = self.angle
        wire_coords = self.wire_coords
        wire_amp = self.wire_amp
        wire_radius = self.wire_radius
        c = self.get_system_parameter('speed_of_sound')
        fs = self.get_system_parameter('sampling_frequency')
        el_coords = self.get_lin_coords()
        nel, _  = np.shape(el_coords)

        # check input and get default parameters if needed
        if txdelays is None:
            txdelays = np.zeros(nel)

        # estimate distances between transducer elements and the 'wire'
        dist = np.zeros(nel)
        for i in range(nel):
            dist[i] = np.sqrt((el_coords[i, 0]-wire_coords[0])**2
                             +(el_coords[i, 1]-wire_coords[1])**2)
        # create output array
        nsamp = np.floor((dist/c + txdelays)*fs + 1).astype(int)
        nmax = 2*np.max(nsamp)
        data = np.zeros((nel,nmax))
        for i in range(nel):
            start = nsamp[i] - wire_radius
            stop = nsamp[i] + wire_radius
            data[i, start:stop] = wire_amp

        return data

    def gen_pwi_data(self):
        angle = self.angle
        wire_coords = self.wire_coords
        wire_amp = self.wire_amp
        wire_radius = self.wire_radius
        txdelay = self.get_pwi_txdelay()
        data = self.gen_data(txdelays=txdelay)
        return data



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
        if parameter == 'rx_sample_range':
            return self.context.sequence.rx_sample_range


    def get_syntetic_data(self, txdelays=None):
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

        # get needed parameters
        angle = self.angle
        wire_coords = self.wire_coords
        wire_amp = self.wire_amp
        wire_radius = self.wire_radius
        wire_radius = 1
        c = self.get_system_parameter('speed_of_sound')
        fs = self.get_system_parameter('sampling_frequency')
        nmax = self.get_system_parameter('rx_sample_range')[1]
        el_coords = self.get_lin_coords()
        nel, _  = np.shape(el_coords)
        txdelays = self.get_lin_txdelays()

        # check input and get default parameters if needed
        if txdelays is None:
            txdelays = np.zeros(nel)

        # estimate distances between transducer elements and the 'wire'
        dist = np.zeros(nel)
        for i in range(nel):
            dist[i] = np.sqrt((el_coords[i, 0]-wire_coords[0])**2
                             +(el_coords[i, 1]-wire_coords[1])**2)
        # create output array
        data = np.zeros((nel,nmax))
        for irx in range(nel):
            for itx in range(nel):
                path = dist[irx] + dist[itx] + txdelays[itx]*c
                weight = np.exp(-2*path)
                nsamp = np.floor(path/c*fs + 1).astype(int)
                start = nsamp - wire_radius
                stop = nsamp + wire_radius
                data[irx, start:stop] +=  wire_amp*weight

        return data


if __name__ == "__main__":
    unittest.main()

