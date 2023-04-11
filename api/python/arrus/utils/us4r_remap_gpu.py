import cupy as cp
import os
from pathlib import Path


current_dir = os.path.dirname(os.path.join(os.path.abspath(__file__)))
_kernel_source = Path(os.path.join(current_dir, "us4r_remap_gpu.cu")).read_text()
remap_module = cp.RawModule(code=_kernel_source)

remap_v1_kernel = remap_module.get_function("arrusRemap")
remap_v2_kernel = remap_module.get_function("arrusRemapV2")


def get_default_grid_block_size(fcm_frames, n_samples, batch_size):
    # Note the kernel implementation
    block_size = (32, 32)
    n_frames, n_channels = fcm_frames.shape
    grid_size = (
        (n_channels - 1) // block_size[0] + 1,
        (n_samples - 1) // block_size[1] + 1,
        n_frames*batch_size
    )
    return grid_size, block_size


def run_remap_v1(grid_size, block_size, params):
    """
    :param params: a list: data_out, data_in, fcm_frames, fcm_channels,
       n_frames, n_samples, n_channels
    :return: data with shape (n_sequences, n_frames, n_samples, n_elements)
    """
    return remap_v1_kernel(grid_size, block_size, params)


def run_remap_v2(grid_size, block_size, params):
    """
    :param params: a list: data_out, data_in, fcm_frames,
      fcm_channels, n_frames, n_samples, n_channels
    :return: array (n_sequences, n_frames, n_samples, n_elements, n_values),
      where n_values is equal 1 for raw channel data, and 2 for I/Q data
    """
    return remap_v2_kernel(grid_size, block_size, params)



