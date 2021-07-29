import arrus
import arrus.session
import arrus.utils.imaging
import arrus.utils.us4r
import numpy as np
import scipy.signal
import cupyx.scipy.ndimage
import math
import matplotlib.pyplot as plt
import pickle
import arrus.metadata
import dataclasses

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
    SelectFrames,
    Squeeze
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
    
    for (int iFrame = 1; iFrame < nFrames; ++iFrame) {
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


class ColorDoppler(arrus.utils.imaging.Operation):

    def __init__(self):
        pass

    def prepare(self, const_metadata):
        self.nframes, self.nx, self.nz = const_metadata.input_shape
        self.output_dtype = cp.float32
        self.output_shape = (2, self.nx, self.nz)  # color, power
        self.output = cp.zeros(self.output_shape, dtype=self.output_dtype)
        self.block = (32, 32)
        self.grid = math.ceil(self.nz/self.block[1]), math.ceil(self.nx/self.block[0])
        self.angle = set(const_metadata.context.sequence.angles.tolist())
        if len(self.angle) > 1:
            raise ValueError("Color doppler mode is implemented only for a "
                             "sequence with a single finite-value transmit "
                             "angle.")
        self.angle = next(iter(self.angle))
        self.angle = self.angle/180*np.pi
        self.pri = const_metadata.context.sequence.pri
        self.tx_frequency = const_metadata.context.sequence.pulse.center_frequency
        self.c = const_metadata.context.sequence.speed_of_sound
        self.scale = self.c/(2*np.pi*self.pri*self.tx_frequency*2*math.cos(self.angle))
        return const_metadata.copy(input_shape=self.output_shape,
                                   dtype=cp.float32, is_iq_data=False)

    def process(self, data):
        params = (
            self.output[0],  # Color
            self.output[1],  # Power
            data,
            self.nframes, self.nx, self.nz
        )
        doppler(self.grid, self.block, params)
        result = self.output
        result[0] = result[0]*self.scale    # [m/s]
        result[1] = 20*cp.log10(result[1])  # [dB]
        return result


class FilterWallClutter(arrus.utils.imaging.Operation):

    def __init__(self, w_n, n):
        self.w_n = w_n
        self.n = n

    def set_pkgs(self, **kwargs):
        pass

    def prepare(self, const_metadata):
        if self.n %2 == 0:
            self.actual_n = self.n+1
        self.taps = scipy.signal.firwin(self.actual_n, self.w_n,
                                        pass_zero=False)
        self.taps = cp.array(self.taps)
        return const_metadata

    def process(self, data):
        output= cupyx.scipy.ndimage.convolve1d(data, self.taps, axis=0)
        return output


def main():
    angle = 10
    n_angles = 32
    center_frequency = 8.6e6
    seq = PwiSequence(
        angles=np.array([angle]*n_angles)*np.pi/180,
        pulse=Pulse(center_frequency=center_frequency, n_periods=2,
                    inverse=False),
        rx_sample_range=(512, 1024*3),
        downsampling_factor=1,
        speed_of_sound=1540,
        pri=100e-6,
        tgc_start=24,
        tgc_slope=0)

    x_grid = np.arange(-15, 15, 0.15) * 1e-3
    z_grid = np.arange(5, 20, 0.15) * 1e-3
    taps = scipy.signal.firwin(32, np.array([0.5, 1.5])*center_frequency,
                               pass_zero=False, fs=65e6)

    scheme = Scheme(
        tx_rx_sequence=seq,
        rx_buffer_size=4,
        output_buffer=DataBufferSpec(type="FIFO", n_elements=12),
        work_mode="HOST",
        processing=Pipeline(
            steps=(
                RemapToLogicalOrder(),
                Transpose(axes=(0, 2, 1)),
                FirFilter(taps),
                # BandpassFilter(),
                QuadratureDemodulation(),
                Decimation(decimation_factor=4, cic_order=2),
                ReconstructLri(x_grid=x_grid, z_grid=z_grid),
                Pipeline(
                    steps=(
                        FilterWallClutter(w_n=0.2, n=32),
                        ColorDoppler(),
                        Transpose(axes=(0, 2, 1)),
                        # Lambda(lambda data: (print(data.get()[1].tolist()), data)[1])
                    ),
                    placement="/GPU:0"
                ),
                SelectFrames([0]),
                Squeeze(),
                # Mean(axis=0),
                EnvelopeDetection(),
                Transpose(),
                LogCompression(),
            ),
            placement="/GPU:0"
        )
    )
    # Here starts communication with the device.
    with arrus.Session(r"C:\Users\Public\us4r.prototxt") as sess:
        us4r = sess.get_device("/Us4R:0")
        us4r.set_hv_voltage(60)

        # Upload sequence on the us4r-lite device.
        buffer, (doppler_metadata, bmode_metadata) = sess.upload(scheme)

        def value_func(data):
            values = data[0]
            mask = data[0]
            mask2 = data[1]
            mask = np.logical_and(mask > -5e-3, mask < 5e-3)
            mask = np.logical_or(mask, mask2 < 60)
            values[mask] = None
            return values

        display = Display2D(
            # The order of layers determines how the data is displayed.
            layers=(
                Layer2D(metadata=bmode_metadata, value_range=(20, 80),
                        cmap="gray", output=1),
                Layer2D(metadata=doppler_metadata, value_range=(-100e-3, 100e-3),
                        cmap="bwr", output=0, value_func=value_func),
            )
        )
        sess.start_scheme()
        display.start(buffer)

        print("Display closed, stopping the script.")
    print("Stopping the example.")


if __name__ == "__main__":
    main()