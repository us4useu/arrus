import scipy.io
import numpy as np
import arrus

dataset = scipy.io.loadmat(r"C:\Users\pjarosik\src\x-files\customers\nanoecho\dataset10.mat")

sess = arrus.MockSession(dataset)
us4r = sess.get_device("/Us4R:0")

sequence = arrus.ops.LinSequence(
    tx_aperture_center_element=np.arange(0, 192),
    tx_aperture_size=64,
    tx_focus=30e-3,
    tx_angle=0,
    pulse=arrus.SineWave(center_frequency=4e6, n_periods=2, inverse=False),
    rx_aperture_center_element=np.arange(0, 192),
    rx_aperture_size=64,
    sampling_frequency=32.5e6)

buffer = us4r.upload(sequence)
us4r.start()

data, metadata = buffer.pop()

print(data)
print(metadata.context)
print(metadata.data_description)
print(metadata.custom)

us4r.stop()

