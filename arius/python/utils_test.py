import unittest
import numpy as np
from utils import create_aligned_array

class CreateAlignedArrayTest(unittest.TestCase):

    def test_output_appropriately_aligned_uint8(self):
        out = create_aligned_array((192, 4096), dtype=np.uint8, alignment=4096)
        self.assertEqual(out.ctypes.data % 4096, 0)

    def test_output_appropriately_aligned_int16(self):
        out = create_aligned_array((32, 1024), dtype=np.int16, alignment=1000)
        self.assertEqual(out.ctypes.data % 1000, 0)

    def test_output_appropriately_aligned_float64(self):
        out = create_aligned_array((32, 32), dtype=np.float64, alignment=124)
        self.assertEqual(out.ctypes.data % 124, 0)

    def test_output_appropriately_shaped(self):
        shape = (8, 1024)
        out = create_aligned_array(shape, dtype=np.int16, alignment=1024)
        self.assertEqual(out.shape, shape)

    def test_output_appropriate_type(self):
        dtype=np.int16
        out = create_aligned_array((8, 1024), dtype=dtype, alignment=1024)
        self.assertEqual(out.dtype, dtype)

if __name__ == "__main__":
    unittest.main()
