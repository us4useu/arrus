import unittest

from arius.tests.tools import mock_import


# Module mocks.
class AriusMock:
    def __init__(self, index):
        self.index = index
        self.tx_mapping = [0]*self.GetNTxChannels()
        self.rx_mapping = [0]*self.GetNRxChannels()

    def GetID(self):
        return -self.index+100

    def IsPowereddown(self):
        return False

    def SetTxChannelMapping(self, srcChannel, dstChannel):
        self.tx_mapping[dstChannel] = srcChannel

    def SetRxChannelMapping(self, srcChannel, dstChannel):
        self.rx_mapping[dstChannel] = srcChannel

    def GetNRxChannels(self):
        return 32

    def GetNTxChannels(self):
        return 128

def GetArius(index):
    return AriusMock(index)

class DBARLiteMock:
    def __init__(self, ii2cmaster):
        self.ii2cmaster = ii2cmaster

    def GetI2CHV(self):
        return self.ii2cmaster

class HV256Mock:
    def __init__(self, ii2cmaster):
        self.ii2cmaster = ii2cmaster

mock_import(
    "arius.devices.iarius",
    IArius=AriusMock,
    GetArius=GetArius
)
mock_import(
    "arius.devices.idbarlite",
    IDBARLite=DBARLiteMock
)
mock_import(
    "arius.devices.ihv256",
    IHV256=HV256Mock
)

# Project imports.
import arius.session as session

class InteractiveSessionTest(unittest.TestCase):

    def test_init(self):
        # TODO(pjarosik) fix the test
        pass
        # sess = session.InteractiveSession()
        # probe = sess.get_device("/Probe:0")
        # hw_subapertures = probe.hw_subapertures
        # cards = [hw.card for hw in hw_subapertures]
        # hw_subapertures = [
        #     (hw_aperture.card.get_id(), hw_aperture.origin, hw_aperture.size)
        #     for hw_aperture in hw_subapertures
        # ]
        # self.assertEqual(("Arius:0", 0, 128), hw_subapertures[0])
        # self.assertEqual(("Arius:1", 0, 64), hw_subapertures[1])
        # self.assertEqual("Arius:0", probe.master_card.get_id())
        # self.assertEqual(sess.get_device("/HV256").hv256_handle.ii2cmaster.GetID(), cards[0].card_handle.GetID())
        # print(cards[0].card_handle.tx_mapping)
        # print(cards[0].card_handle.rx_mapping)
        # print(cards[1].card_handle.tx_mapping)
        # print(cards[1].card_handle.rx_mapping)


if __name__ == "__main__":
    unittest.main()