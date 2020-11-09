import numpy as np
import arrus.kernels.imaging
import arrus.ops.imaging
import arrus.medium
import unittest
import dataclasses


@dataclasses.dataclass(frozen=True)
class ProbeMock:
    pitch: float
    n_elements: int


@dataclasses.dataclass(frozen=True)
class DeviceMock:
    probe: ProbeMock


@dataclasses.dataclass(frozen=True)
class ContextMock:
    device: DeviceMock
    medium: arrus.medium.MediumDTO
    op: object


class LinSequenceTest(unittest.TestCase):

    def test_simple_sequence_with_paddings(self):

        # three tx/rxs with aperture centers: 0, 16, 31
        seq = arrus.ops.imaging.LinSequence(
            tx_aperture_center_element=np.array([0, 15, 16, 31]),
            tx_aperture_size=32,
            tx_focus=30e-3,
            pulse=arrus.ops.us4r.Pulse(center_frequency=5e6, n_periods=3,
                                       inverse=False),
            rx_aperture_center_element=np.array([0, 15, 16, 31]),
            rx_aperture_size=32,
            pri=1000e-6,
            downsampling_factor=1,
            rx_sample_range=(0, 4096))

        medium = arrus.medium.MediumDTO(name="test", speed_of_sound=1540)
        device = DeviceMock(probe=ProbeMock(pitch=0.2e-3, n_elements=32))
        context = ContextMock(device=device, medium=medium, op=seq)
        tx_rx_sequence = arrus.kernels.imaging.create_lin_sequence(context)

        # expected delays
        # TODO check also if appropriate delays are computed
        delays = arrus.kernels.imaging.enum_classic_delays(
            n_elem=32, pitch=0.2e-3, c=1540, focus=30e-3)

        # Expected aperture tx/rx 1
        expected_aperture = np.zeros((32,), dtype=bool)
        expected_aperture[0:16+1] = True
        expected_delays = delays[15:]
        tx_rx = tx_rx_sequence.ops[0]
        np.testing.assert_array_equal(tx_rx.tx.aperture, expected_aperture)
        np.testing.assert_array_equal(tx_rx.rx.aperture, expected_aperture)
        np.testing.assert_almost_equal(tx_rx.tx.delays, expected_delays)

        # Expected aperture tx/rx 2
        expected_aperture = np.ones((32,), dtype=bool)
        tx_rx = tx_rx_sequence.ops[1]
        np.testing.assert_array_equal(tx_rx.tx.aperture, expected_aperture)
        np.testing.assert_array_equal(tx_rx.rx.aperture, expected_aperture)

        # Expected aperture tx/rx 3
        expected_aperture = np.zeros((32,), dtype=bool)
        expected_aperture[1:] = True
        tx_rx = tx_rx_sequence.ops[2]
        np.testing.assert_array_equal(tx_rx.tx.aperture, expected_aperture)
        np.testing.assert_array_equal(tx_rx.rx.aperture, expected_aperture)

        # Expected aperture tx/rx 4
        expected_aperture = np.zeros((32,), dtype=bool)
        expected_aperture[16:] = True
        tx_rx = tx_rx_sequence.ops[3]
        np.testing.assert_array_equal(tx_rx.tx.aperture, expected_aperture)
        np.testing.assert_array_equal(tx_rx.rx.aperture, expected_aperture)



if __name__ == "__main__":
    unittest.main()
