import matplotlib.pyplot as plt
import unittest
import numpy as np
import cupy as cp
from arrus.utils.tests.utils import ArrusImagingTestCase
from arrus.ops.us4r import Scheme, Pulse
from arrus.ops.imaging import PwiSequence, LinSequence
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
    #data = 20*np.log10(1e-6+ data/data.max())
    ax.imshow(data)
    ax.set_aspect('auto')
    plt.show()

def plot_line(data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(data)
    plt.show()

def show_surface(data):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.imshow(data)


class ReconstructionTestCase(ArrusImagingTestCase):

    def setUp(self) -> None:
        device = self.get_device()
        #self.op = ReconstructLri
        n_elements = self.get_system_parameter('n_elements')
        pitch = self.get_system_parameter('pitch')
        probe_width = (n_elements-1)*pitch
        fs = self.get_system_parameter('sampling_frequency')
        c = self. get_system_parameter('speed_of_sound')
        ds = c/fs

        # define x_grid vector
        ncol = np.round(probe_width/ds).astype(int)+1
        self.x_grid = np.linspace(-probe_width/2, probe_width/2, ncol)
        # define z_grid vector
        ztop = 0
        zbot = 50*1e-3
        nrow = np.round((zbot - ztop)/ds + 1).astype(int)
        self.z_grid = np.linspace(ztop, zbot , nrow)

        # set arbitrary tolerances (in [mm]) i.e. possible differences
        # between expected and obtained maxima in b-mode image of a wire
        xtolerance_mm = 1.5
        ztolerance_mm = 1.5
        self.xtol = np.round(xtolerance_mm/ds*1e-3).astype(int)
        self.ztol = np.round(ztolerance_mm/ds*1e-3).astype(int)

        # set range of angles used in pwi scheme
        angles = np.array([-10,0,10])*np.pi/180
        self.angles = angles

        # set wire parameters
        self.wire_amp = 100
        self.wire_radius = 0
        self.wire_coords = (0, 30e-3)

        # set if print some info or not
        self.verbose = False



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



#--------------------------------------------------------------------------
#                         TOOLS 
#--------------------------------------------------------------------------

    def get_device(self):
        probe = self.get_probe_model_instance(
            n_elements=128,
            pitch=0.2e-3,
            curvature_radius=0.0
            )
        device = self.get_ultrasound_device(
            probe=probe,
            sampling_frequency=65e6
            )
        return device


    def get_pulse_length(self):
       fs = self.get_system_parameter('sampling_frequency')
       fc = self.get_system_parameter('center_frequency')
       c = self.get_system_parameter('speed_of_sound')
       n_periods = self.get_system_parameter('n_periods')
       pulse_length = np.round(fs/fc*n_periods).astype(int)
       return pulse_length


    def get_wire_indexes(self):
        x = self.wire_coords[0]
        z = self.wire_coords[1]
        xi = np.abs(self.x_grid - x).argmin(axis=0)
        zi = np.abs(self.z_grid - z).argmin(axis=0)
        return (xi, zi)

    def get_lin_el_coords(self, n_elements=None):
        '''
        Auxiliary tool for generating array transducer elements coordinates for linear array.

        :param nel: number of elements,
        :param pitch: distance between elements,
        :return: numpy array with elements coordinates (x,z)
        '''
        fs = self.get_system_parameter('sampling_frequency')
        pitch = self.get_system_parameter('pitch')
        if n_elements is None:
            n_elements = self.get_system_parameter('n_elements')

        elx = np.linspace(-(n_elements-1)*pitch/2, (n_elements-1)*pitch/2, n_elements)
        elz = np.zeros(n_elements)
        coords = np.array(list(zip(elx,elz)))
        return coords


    def get_system_parameter(self, parameter):
        '''
        The function returns selected system parameter.
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
        if parameter == 'n_periods':
            return self.context.sequence.pulse.n_periods
        if parameter == 'tx_focus':
            return self.context.sequence.tx_focus
        if parameter == 'angles':
            return self.context.sequence.angles
        if parameter == 'tx_aperture_size':
            return self.context.sequence.tx_aperture_size



    def get_syntetic_data(self, aperture=None, txdelays=None):
        '''
        Function for generation of artificial non-beamformed data
        corresponding to single point (wire) within empty medium.

        :param txdelays: initial transmit delays,
        :return: 2D numpy array of zeros and single pixel
                 with amplitude equal to 'wire_amp' parameter.
        '''


        # get needed parameters
        #angle = self.angle
        wire_coords = self.wire_coords
        wire_amp = self.wire_amp
        wire_radius = self.wire_radius
        c = self.get_system_parameter('speed_of_sound')
        fs = self.get_system_parameter('sampling_frequency')
        nmax = 2*self.get_system_parameter('rx_sample_range')[1]
        pulse_length = self.get_pulse_length()
        el_coords = self.get_lin_el_coords()
        nel, _  = np.shape(el_coords)

        # check input and get default parameters if needed
        if txdelays is None:
            txdelays = np.zeros(nel)

        if aperture is None:
            aperture = np.arange(nel)

        # estimate distances between transducer elements and the 'wire'
        dist = np.zeros(nel)
        apod = np.zeros(nel)
        for i in aperture:
            dist[i] = np.sqrt((el_coords[i, 0]-wire_coords[0])**2
                             +(el_coords[i, 1]-wire_coords[1])**2)
            #apod[i] = (wire_coords[1]/dist[i])**4
            cos_alpha = wire_coords[1]/dist[i]
            alpha = np.arccos(cos_alpha)
            apod[i] = np.sinc(alpha)

        # create output array
        aperture_n_elements = np.size(aperture)
        data = np.zeros((aperture_n_elements,nmax))
        for i in range(aperture_n_elements):
            irx = aperture[i]
            for j in range(aperture_n_elements):
                itx = aperture[j]
                path = dist[irx] + dist[itx]
                weight = apod[irx]*apod[itx]
                nsamp = np.floor((path/c + txdelays[j])*fs + 1).astype(int)
                start = nsamp - wire_radius
                stop = nsamp + wire_radius
                data[i, start:stop+1] +=  wire_amp*weight
                #data[i, start:stop+1] +=  wire_amp

        return data


    def get_grid_resolution(self):
        dx = self.x_grid[1] - self.x_grid[0]
        dz = self.z_grid[1] - self.z_grid[0]
        return dx, dz


    def get_coords_difference(self, result, verbose=False):
        '''
        Function returns difference between x and z coordinates
        of the assumed wire and max value in resulted amplitude image.
        If verbose = True, function prints some info on the differences.
        '''
        angle = self.get_system_parameter('angles')
        x, z = self.wire_coords
        x = np.round(x*1e3, 1)
        z = np.round(z*1e3, 1)
        maxval = np.nanmax(result)
        dx, dz = self.get_grid_resolution()

        # Expect
        # Indexes corresponding to wire coordinates in beamformed image
        iwire, jwire = self.get_wire_indexes()

        # indexes corresponding to max value of beamformed amplitude image
        i, j = get_max_ndx(result)

        # difference between expected and obtained
        idiff = np.abs(iwire-i)
        jdiff = np.abs(jwire-j)
        xdiff = np.round((idiff-1)*dx*1e3, 2)
        zdiff = np.round((jdiff-1)*dz*1e3, 2)

        if verbose:
            print('----------------------------------------------')
            print(f'current angle [deg]: {angle*180/np.pi}')
            print(f'current wire coordinates [mm]: ({x},{z})')
            print(f'max value in result array: {np.round(maxval)}')
            print('------')
            print(f'expected wire row index value (x): {iwire}')
            print(f'obtained wire row index value (x): {i}')
            print(f'difference in x axis [samples]: {idiff}')
            print(f'difference in x axis [mm]: {xdiff}')
            print('------')
            print(f'expected wire column index value (z): {jwire}')
            print(f'obtained wire column index valuej (z): {j}')
            print(f'difference in z axis [samples]: {jdiff}')
            print(f'difference in z axis [mm]: {zdiff}')
            print('')
            print('')

        return idiff, jdiff



class PwiReconstructionTestCase(ReconstructionTestCase):

    def setUp(self) -> None:
        self.context = self.get_context(angle=0)
        self.op = ReconstructLri
        super().setUp()


    def test_0(self):
        # Given
        data = 0
        # Run
        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
        # Expect
        expected_shape = (self.x_grid.size, self.z_grid.size )
        expected = np.zeros(expected_shape, dtype=complex)
        np.testing.assert_equal(result, expected)


    def test_empty(self):
        # Given
        data = []
        # Run
        result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
        # Expect
        expected_shape = (self.x_grid.size, self.z_grid.size )
        expected = np.zeros(expected_shape, dtype=complex)
        np.testing.assert_equal(result, expected)

    def test_pwi_angles(self):
        # Given
        max_x = np.max(self.x_grid)
        min_x = np.min(self.x_grid)
        max_z = np.max(self.z_grid)
        min_z = np.min(self.z_grid)
        xmargin = (max_x-min_x)*0.1
        zmargin = (max_z-min_z)*0.1
        wire_x = np.linspace(min_x+xmargin, max_x-xmargin, 3)
        wire_z = np.linspace(min_z+zmargin, max_z-zmargin, 3)

        # 3 lines below are temporary - for testing only
        #wire_x = np.array([0])
        #wire_z = np.array([10e-3])
        #self.angles = np.array([0])


        for angle in self.angles:
            #self.angle = angle
            self.context = self.get_context(angle=angle)
            for x in wire_x:
                for z in wire_z:
                    wire_coords = (x, z)
                    self.wire_coords = wire_coords
                    #data = self.gen_pwi_data()
                    data = self.get_syntetic_pwi_data()
                    # Run
                    result = self.run_op(data=data, x_grid=self.x_grid, z_grid=self.z_grid)
                    #plot_line(data[64,:])
                    result = np.abs(result)
                    #show_image(np.abs(result.T))
                    #show_image(data)
                    idiff, jdiff = self.get_coords_difference(result, verbose=self.verbose)
                    self.assertLessEqual(idiff, self.xtol)
                    self.assertLessEqual(jdiff, self.ztol)

#--------------------------------------------------------------------------
#                         TOOLS 
#--------------------------------------------------------------------------


    def get_context(self, angle):
        '''
        Function generate context data for pwi tests.
        '''
        device = self.get_device()
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

        return self.get_default_context(
            sequence=sequence,
            device=device,
            )


    def get_syntetic_pwi_data(self):
        txdelays = self.get_pwi_txdelays()
        return self.get_syntetic_data(txdelays=txdelays)


    def get_pwi_txdelays(self):
        '''
        The function generate transmit delays of PWI scheme for linear array.
        '''
        speed_of_sound = self.get_system_parameter('speed_of_sound')
        el_coords = self.get_lin_el_coords()
        angle = self.get_system_parameter('angles')
        delays = el_coords[:,0]*np.tan(angle)/speed_of_sound
        return delays


class BfrReconstructionTestCase(ReconstructionTestCase):

    def setUp(self) -> None:
        self.context = self.get_context(tx_focus=30e-3)
        self.op = RxBeamforming
        super().setUp()

        tx_aperture_size = self.get_system_parameter('tx_aperture_size')
        n_elements = self.get_system_parameter('n_elements')
        n_emissions = n_elements - tx_aperture_size + 1
        pitch = self.get_system_parameter('pitch')
        roi_width = (n_emissions-1)*pitch
        fs = self.get_system_parameter('sampling_frequency')
        c = self. get_system_parameter('speed_of_sound')
        ds = c/fs/2 # dlaczego musi być /2 ? (bez tego róznice są ok 2x)

        # define x_grid vector
        ncol = n_emissions
        self.x_grid = np.linspace(-roi_width/2, roi_width/2, ncol)
        # define z_grid vector
        ztop = 0
        zbot = 50*1e-3
        nrow = np.round((zbot - ztop)/ds + 1).astype(int)
        self.z_grid = np.linspace(ztop, zbot , nrow)


    def test_0(self):
        # Given
        data = 0

        # Run
        result = self.run_op(data=data)

        # Expect
        expected_shape = (self.x_grid.size, self.z_grid.size )
        expected = np.zeros(expected_shape, dtype=complex)
        np.testing.assert_equal(result, expected)


    # TODO: RxBeamforming nie obsluguje pustych danych na wejsciu?
    #def test_empty(self):
    #    # Given
    #    data = []
    #    # Run
    #    result = self.run_op(data=data)
    #    # Expect
    #    expected_shape = (self.x_grid.size, self.z_grid.size )
    #    expected = np.zeros(expected_shape, dtype=complex)
    #    np.testing.assert_equal(result, expected)


    def test_x(self):
        # Given
        self.context = self.get_context(tx_focus=20e-3)
        self.wire_coords = (5e-3, 10e-3)
        data = self.get_syntetic_bfr_data()

        # Run
        result = self.run_op(data=data)
        result = np.abs(result)
        #show_image(result.T)
        #show_surface(result)
        idiff, jdiff = self.get_coords_difference(result)
        self.assertLessEqual(idiff, self.xtol)
        self.assertLessEqual(jdiff, self.ztol)



#--------------------------------------------------------------------------
#                         TOOLS 
#--------------------------------------------------------------------------


    def get_context(self, tx_focus):

        device = self.get_device()
        n_elements = device.probe.model.n_elements
        sequence = LinSequence(
            tx_aperture_center_element=np.arange(0, n_elements),
            tx_aperture_size=64,
            tx_focus=tx_focus,
            pulse=Pulse(center_frequency=6e6, n_periods=2, inverse=False),
            rx_aperture_center_element=np.arange(0, n_elements),
            rx_aperture_size=64,
            rx_sample_range=(0, 2048),
            pri=200e-6,
            tgc_start=14,
            tgc_slope=2e2,
            downsampling_factor=1,
            speed_of_sound=1450,
            )
        return self.get_default_context(
            sequence=sequence,
            device=device,
            )

    def get_bfr_txdelays(self):
        '''
        The function generate transmit delays of PWI scheme for linear array.
        '''
        speed_of_sound = self.get_system_parameter('speed_of_sound')
        tx_aperture_size = self.get_system_parameter('tx_aperture_size')
        el_coords = self.get_lin_el_coords(tx_aperture_size)
        angle = self.get_system_parameter('angles')
        tx_focus = self.get_system_parameter('tx_focus')

        delays_angle = el_coords[:,0]*np.tan(angle)/speed_of_sound
        delays_focus = (np.sqrt(el_coords[:,0]**2 + tx_focus**2) - tx_focus)/speed_of_sound
        delays = delays_angle + delays_focus
        return delays

        return self.get_default_context(
            sequence=sequence,
            device=device,
            )

    def get_syntetic_bfr_data(self):
        tx_aperture_size = self.get_system_parameter('tx_aperture_size')
        txdelays = self.get_bfr_txdelays()
        data0 = self.get_syntetic_data(aperture=np.arange(0, tx_aperture_size), txdelays=txdelays)
        _, nsamp = data0.shape
        n_elements = self.get_system_parameter('n_elements')
        n_emissions = n_elements - tx_aperture_size + 1
        data = np.zeros((n_emissions, tx_aperture_size, nsamp))
        data[0,:,:] = data0
        for itx in range(1, n_emissions):
            aperture=np.arange(itx, itx+tx_aperture_size)
            data[itx,:,:] = self.get_syntetic_data(
                                aperture=np.arange(itx, itx+tx_aperture_size),
                                txdelays=txdelays,
                                )
        return data.astype(np.float32)






if __name__ == "__main__":
    unittest.main()

