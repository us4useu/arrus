from arrus.ops.imaging import (
    SimpleTxRxSequence,
    LinSequence,
    PwiSequence,
    StaSequence
)

import unittest
import numpy as np

def create_sequence(type, **kwargs):
    params = {
        "pulse": "pulse",
        "rx_sample_range": (0, 2048),
        "pri": 100e-6,
        "sri": None,
        "speed_of_sound": None,
        "tx_focus": np.inf,
        "angles": 0.0,
        "downsampling_factor": 1,
        "tx_aperture_center_element": None,
        "tx_aperture_center": None,
        "tx_aperture_size": None,
        "rx_aperture_center_element": None,
        "rx_aperture_center": None,
        "rx_aperture_size": None,
        "tgc_start": None,
        "tgc_slope": None,
        "tgc_curve": []
    }
    params = {**params, **kwargs}
    return type(**params)


class SimpleTxRxSequenceValidationTest(unittest.TestCase):
    def accept_scalars_and_lists(self):
        create_sequence(
            SimpleTxRxSequence,
            tx_aperture_center_element=[0, 1, 2, 3],
            rx_aperture_center_element=2,
            tx_focus=[-1.0, -1.0, -2.0, -1.0]
        )

    def rejects_tx_rx_aperture_center_element_and_center_parameters(self):
        with self.assertRaisesRegex(ValueError, "at most one") as ctx:
            create_sequence(
                SimpleTxRxSequence,
                tx_aperture_center_element=[0, 1, 2, 3],
                tx_aperture_center=[0.0, 1.0, 2.0, 3.0])

        with self.assertRaisesRegex(ValueError, "at most one") as ctx:
            create_sequence(
                SimpleTxRxSequence,
                rx_aperture_center_element=[0, 1, 2, 3],
                rx_aperture_center=[0.0, 1.0, 2.0, 3.0])

    def test_rejects_rx_aperture_size_collection(self):
        with self.assertRaisesRegex(ValueError, "scalar") as ctx:
            create_sequence(SimpleTxRxSequence, rx_aperture_size=[1, 2, 3])

    def test_rejects_different_size_lists(self):
        with self.assertRaisesRegex(ValueError, "should have the same length") as ctx:
            create_sequence(
                SimpleTxRxSequence,
                tx_aperture_center_element=[0, 1, 2, 3],
                rx_aperture_center_element=[0, 1, 2])

    def test_rejects_list_of_tx_focus(self):
        with self.assertRaisesRegex(ValueError, "scalar") as ctx:
            create_sequence(
                LinSequence,
                tx_focus=[0, 1, 2, 3])


class LinSequenceValidationTest(unittest.TestCase):

    def test_rejects_list_of_tx_angles(self):
        with self.assertRaisesRegex(ValueError, "scalar") as ctx:
            create_sequence(
                LinSequence,
                angles=[0, 1, 2, 3])


class PwiSequenceValidationTest(unittest.TestCase):

    def test_accepts_default_parameter_no_tx_focus(self):
        PwiSequence(
            pulse="pulse",
            rx_sample_range=(0, 2048),
            pri=100e-6,
            sri=None,
            speed_of_sound=None,
            angles=[0.0, 1.0, 2.0])

    def test_rejects_tx_focus_non_inf_scalar(self):
        with self.assertRaisesRegex(ValueError, "inf") as ctx:
            create_sequence(
                PwiSequence,
                tx_focus=10e-3)


if __name__ == "__main__":
    unittest.main()
