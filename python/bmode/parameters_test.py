import python.bmode.parameters as parameters
import unittest
import os
import tempfile
import scipy.io
import numpy as np

########################################## Reading correct data.
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


########################################## Reading incorrect data.
class LoadIncorrectStructure(unittest.TestCase):
    def setUp(self):
        self.filepath = os.path.join(
            tempfile.gettempdir(),
            self.__class__.__name__ + '.mat')

    def tearDown(self):
        os.remove(self.filepath)

    def perform_test(self, structure,
                     structure_modifier,
                     expected_exception,
                     expected_msg=None):
        # Convert the structure to matlab version.
        matlab_struct = parameters._convert_to_matlab_structure(structure)
        matlab_struct = structure_modifier(matlab_struct)
        scipy.io.savemat(self.filepath, {"x": matlab_struct})

        # Load descriptors from the .mat file, should fail with given exception.
        with self.assertRaises(expected_exception) as ctx:
            read_matlab_data = scipy.io.loadmat(self.filepath)['x']
            parameters._load_matlab_structure(type(structure), read_matlab_data)
        if expected_msg is not None:
            self.assertTrue(expected_msg in str(ctx.exception))


class LoadFileWithMissingAttribute(LoadIncorrectStructure):
    TX = parameters.Tx(
        frequency=5e6,
        n_periods=1,
        angles=None, # focus and angles are None
        focus=None,
        aperture_size=192
    )

    def _get_modifier(self, key):
        def modifier(structure):
            result = dict(structure)
            del result[key]
            return result
        return modifier

    def test_missing_frequency(self):
        self.perform_test(
            structure=self.TX,
            structure_modifier=self._get_modifier('frequency'),
            expected_exception=ValueError,
            expected_msg="does not contain 'frequency'"
        )

    def test_missing_angles(self):
        self.perform_test(
            structure=self.TX,
            structure_modifier=self._get_modifier('angles'),
            expected_exception=ValueError,
            expected_msg="does not contain 'angles'"
        )

class LoadFileWithWrongAttributeType(LoadIncorrectStructure):
    TX = parameters.Tx(
        frequency=5e6,
        n_periods=1,
        angles=None, # focus and angles are None
        focus=None,
        aperture_size=192
    )

    def _get_modifier(self, key, value):
        def modifier(structure):
            result = dict(structure)
            result[key] = value
            return result
        return modifier

    def test_ensures_precision_safe_casting(self):
        self.perform_test(
            structure=self.TX,
            structure_modifier=self._get_modifier(
                'nPeriods',
                np.uint32(np.iinfo(np.uint32).max)
            ),
            expected_exception=ValueError,
            expected_msg="'n_periods': can't be cast from "
                         "'<class 'numpy.uint32'>',"
                         " to '<class 'numpy.uint16'>"
        )

    def test_ensures_proper_rank(self):
        self.perform_test(
            structure=self.TX,
            structure_modifier=self._get_modifier(
                'nPeriods',
                [1,2,3,4]
            ),
            expected_exception=ValueError,
            expected_msg="'n_periods' should be a scalar"
        )

    def test_ensures_proper_type(self):
        self.perform_test(
            structure=self.TX,
            structure_modifier=self._get_modifier(
                'angles', 0),
            expected_exception=ValueError
        )


if __name__ == '__main__':
    unittest.main()