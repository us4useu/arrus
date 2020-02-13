import python.bmode.parameters as parameters
import unittest
import os
import tempfile

class TestReadingCorrectData(unittest.TestCase):
    def setUp(self):
        self.filepath = os.path.join(
            tempfile.gettempdir(),
            self.__class__.__name__ + '.mat')

    def tearDown(self):
        os.remove(self.filepath)

    def perform_test(self, acq_params, sys_params):
        parameters.save_matlab_file(
            self.filepath,
            acquisition_parameters=acq_params,
            system_parameters=sys_params
        )
        # Load descriptors from the matlab file.
        read_sys_params, read_acq_params = \
            parameters.load_matlab_file(self.filepath)
        print(read_acq_params)

        # Verify.
        self.assertEqual(sys_params, read_sys_params)
        self.assertEqual(acq_params, read_acq_params)


class LoadMatlabStructureLin(TestReadingCorrectData):
    SYS_PARAMS = parameters.SystemParameters(
        n_elements=192, pitch=0.25e-3
    )
    TX = parameters.Tx(
        frequency=5e6,
        n_periods=1,
        angles=None, # angles are set to None
        focus=0.02,
        aperture_size=192
    )
    RX = parameters.Rx(
        sampling_frequency=50e6,
        aperture_size=192
    )
    ACQ_PARAMS = parameters.AcquisitionParameters(
        mode='lin',
        speed_of_sound=1500,
        tx=TX,
        rx=RX
    )

    def test_load_matlab_structure(self):
        self.perform_test(self.ACQ_PARAMS, self.SYS_PARAMS)


class LoadMatlabStructurePWI(TestReadingCorrectData):
    SYS_PARAMS = parameters.SystemParameters(
        n_elements=192, pitch=0.25e-3
    )
    TX = parameters.Tx(
        frequency=5e6,
        n_periods=1,
        angles=[0, 1, 2],
        focus=None, # focus is set to None
        aperture_size=192
    )
    RX = parameters.Rx(
        sampling_frequency=50e6,
        aperture_size=192
    )
    ACQ_PARAMS = parameters.AcquisitionParameters(
        mode='pwi',
        speed_of_sound=1540,
        tx=TX,
        rx=RX
    )

    def test_load_matlab_structure(self):
        self.perform_test(self.ACQ_PARAMS, self.SYS_PARAMS)


class LoadMatlabStructureSTA(TestReadingCorrectData):
    SYS_PARAMS = parameters.SystemParameters(
        n_elements=192, pitch=0.25e-3
    )
    TX = parameters.Tx(
        frequency=5e6,
        n_periods=1,
        angles=None, # focus and angles are None
        focus=None,
        aperture_size=192
    )
    RX = parameters.Rx(
        sampling_frequency=50e6,
        aperture_size=192
    )
    ACQ_PARAMS = parameters.AcquisitionParameters(
        mode='pwi',
        speed_of_sound=1450.0,
        tx=TX,
        rx=RX
    )

    def test_load_matlab_structure(self):
        self.perform_test(self.ACQ_PARAMS, self.SYS_PARAMS)

# TODO test reading incorrect data structure

if __name__ == '__main__':
    unittest.main()