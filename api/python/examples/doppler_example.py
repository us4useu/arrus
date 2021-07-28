import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import numpy as np
import scipy.signal
import cupyx.scipy.ndimage
import math
import matplotlib.pyplot as plt

from arrus.ops.us4r import (
    Scheme,
    Pulse,
    DataBufferSpec
)
from arrus.ops.imaging import (
    PwiSequence
)
from arrus.utils.imaging import (
    Pipeline,
    Transpose,
    BandpassFilter,
    FirFilter,
    Decimation,
    QuadratureDemodulation,
    EnvelopeDetection,
    LogCompression,
    Enqueue,
    RxBeamformingImg,
    ReconstructLri,
    Mean,
    Lambda,
    SelectFrames
)
from arrus.utils.us4r import (
    RemapToLogicalOrder
)
from arrus.utils.gui import (
    Display2D,
    Layer2D
)

import cupy as cp


arrus.set_clog_level(arrus.logging.INFO)
arrus.add_log_file("test.log", arrus.logging.INFO)

source = r"""
#include <cupy/complex.cuh>
extern "C" __global__ 
void doppler(float *color, 
             float *power, 
             const complex<float> *iqFrames, 
             const int nFrames, 
             const int nx, 
             const int nz)
                  
{
    int z = blockIdx.x * blockDim.x + threadIdx.x;
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (z >= nz || x >= nx) {
        return;
    }
    
    /* Color estimation */
    complex<float> iqCurrent, iqPrevious;
    float ic, qc, ip, qp, pwr, nom = 0.0f, den = 0.0f;

    iqCurrent = iqFrames[z + x*nz];
    ic = real(iqCurrent);
    qc = imag(iqCurrent);
    pwr = ic*ic + qc*qc;
    
    for (int iFrame = 0; iFrame < nFrames; ++iFrame) {
        // previous I and Q values
        ip = ic;
        qp = qc;
        
        // current I and Q values
        iqCurrent = iqFrames[z + x*nz + iFrame*nz*nx];
        ic = real(iqCurrent);
        qc = imag(iqCurrent);
        
        pwr += ic*ic + qc*qc;
        den += ic*ip + qc*qp;
        nom += qc*ip - ic*qp;
    }
    color[z + x*nz] = atan2f(nom, den);
    power[z + x*nz] = pwr/nFrames;
}

"""
doppler = cp.RawKernel(source, 'doppler')


class ColorDoppler:

    def __init__(self):
        pass

    def prepare(self, input_shape):
        self.nframes, self.nx, self.nz = input_shape
        self.output_dtype = cp.float32
        self.output_shape = self.nx, self.nz
        self.output = cp.zeros(self.output_shape, dtype=self.output_dtype)
        self.power_output = cp.zeros(self.output_shape, dtype=self.output_dtype)
        self.block = (32, 32)
        self.grid = math.ceil(self.nz/self.block[1]), math.ceil(self.nx/self.block[0])

    def process(self, data):
        params = (
            self.output,
            self.power_output,
            data,
            self.nframes, self.nx, self.nz
        )
        doppler(self.grid, self.block, params)
        return self.output


class FilterWallClutter:

    def __init__(self, w_n, n):
        self.w_n = w_n
        self.n = n

    def prepare(self):
        if self.n %2 == 0:
            self.actual_n = self.n+1
        self.taps = scipy.signal.firwin(self.actual_n, self.w_n, pass_zero=False)
        self.taps = cp.array(self.taps)

    def process(self, data):
        output= cupyx.scipy.ndimage.convolve1d(data, self.taps, axis=0)
        return output



def main():

    data = np.load("C:/Users/pjarosik/Desktop/doppler_data.npz")
    iq = data["iq"]
    c = data["c"]
    tx_frequency = data["tx_frequency"]

    iq_gpu = cp.array(iq)
    input_shape = iq.shape
    f = FilterWallClutter(w_n=0.19, n=128)
    f.prepare()
    dc = ColorDoppler()
    dc.prepare(input_shape=input_shape)

    out = f.process(iq_gpu)
    out = dc.process(out)
    out = out.T
    out_host = out.get()

    plt.imshow(out_host, cmap="hot")
    plt.show()


    # angle = 0
    # center_frequency = 6e6
    # seq = PwiSequence(
    #     angles=np.array([angle]*1)*np.pi/180,
    #     pulse=Pulse(center_frequency=center_frequency, n_periods=2, inverse=False),
    #     rx_sample_range=(256, 1024*3),
    #     downsampling_factor=1,
    #     speed_of_sound=1450,
    #     pri=200e-6,
    #     tgc_start=14,
    #     tgc_slope=2e2)
    #
    # x_grid = np.arange(-15, 15, 0.2) * 1e-3
    # z_grid = np.arange(5, 35, 0.2) * 1e-3
    # taps = scipy.signal.firwin(64, np.array([0.5, 1.5])*center_frequency,
    #                            pass_zero=False, fs=65e6)
    #
    # import cupy as cp
    #
    # scheme = Scheme(
    #     tx_rx_sequence=seq,
    #     rx_buffer_size=4,
    #     output_buffer=DataBufferSpec(type="FIFO", n_elements=12),
    #     work_mode="HOST",
    #     processing=Pipeline(
    #         steps=(
    #             RemapToLogicalOrder(),
    #             Transpose(axes=(0, 2, 1)),
    #             FirFilter(taps),
    #             QuadratureDemodulation(),
    #             Decimation(decimation_factor=4, cic_order=2),
    #             ReconstructLri(x_grid=x_grid, z_grid=z_grid),
    #             Mean(axis=0),
    #             EnvelopeDetection(),
    #             Transpose(),
    #             LogCompression(),
    #             Pipeline(
    #                 steps=(
    #                     Lambda(lambda data: cp.log(cp.exp(data))),
    #                 ),
    #                 placement="/GPU:0"
    #             ),
    #             Lambda(lambda data: data)
    #         ),
    #         placement="/GPU:0"
    #     )
    # )
    #
    # const_metadata = None
    # # Here starts communication with the device.
    # with arrus.Session(r"C:\Users\Public\us4r.prototxt") as sess:
    #     us4r = sess.get_device("/Us4R:0")
    #     us4r.set_hv_voltage(10)
    #
    #     # Upload sequence on the us4r-lite device.
    #     buffer, (bmode_metadata, doppler_metadata) = sess.upload(scheme)
    #     display = Display2D(
    #         # The order of layers determines how the data is displayed.
    #         layers=(
    #             Layer2D(metadata=bmode_metadata, value_range=(20, 80),
    #                     cmap="gray", output=1),
    #             Layer2D(metadata=doppler_metadata, value_range=(75, 80),
    #                     cmap="hot", clip="transparent", output=0),
    #         )
    #     )
    #     sess.start_scheme()
    #     display.start(buffer)
    #
    #     print("Display closed, stopping the script.")
    # print("Stopping the example.")


if __name__ == "__main__":
    main()