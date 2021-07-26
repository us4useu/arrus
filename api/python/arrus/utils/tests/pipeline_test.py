import time
import unittest
from collections import deque, namedtuple
from collections.abc import Iterable
import numpy as np
import cupy as cp

from arrus.utils.imaging import (
    Buffer, BufferElement, PipelineRunner, Pipeline, Lambda
)

MetadataMock = namedtuple("MetadataMock", ["input_shape", "dtype"])

class PipelineRunnerTestCase(unittest.TestCase):
    """
    NOTE: these tests requires Host computer with GPU installed.
    NOTE: the speed of producer and consumer is hardware dependent and should
    be should be treated as a rough assumptions.
    """

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()
        if hasattr(self, "runner"):
            self.in_buffer = None
            self.gpu_buffer = None
            self.out_buffer = None
            self.runner.stop()
            self.runner = None

    def __create_runner(self, data_shape,
                        in_buffer_size=2, gpu_buffer_size=2, out_buffer_size=2,
                        n_out_buffers=1,
                        pipeline=None, callbacks=None,
                        buffer_type="locked"):
        self.data_shape = data_shape
        self.dtype = np.int32
        self.in_buffer = Buffer(n_elements=in_buffer_size,
                                shape=data_shape,
                                dtype=self.dtype, math_pkg=np,
                                type=buffer_type)
        self.gpu_buffer = Buffer(n_elements=gpu_buffer_size,
                                 shape=data_shape,
                                 dtype=self.dtype, math_pkg=cp,
                                 type=buffer_type)
        self.out_buffer = [Buffer(n_elements=out_buffer_size,
                                 shape=data_shape,
                                 dtype=self.dtype, math_pkg=np,
                                 type=buffer_type)
                           for _ in range(n_out_buffers)]

        for i, element in enumerate(self.in_buffer.elements):
            element.data = np.zeros(data_shape, dtype=self.dtype)

        self.runner = PipelineRunner(
            self.in_buffer, self.gpu_buffer, self.out_buffer, pipeline,
            callbacks)
        return self.runner

    def __run_increment_sync(self, buffer, n_runs):
        value = 0
        for i in range(n_runs):
            for j in range(len(buffer.elements)):
                element = buffer.acquire(j)
                element.data[:] = value
                self.runner.process(element)
                value += 1
        self.runner.sync()

    def __verify_increment(self, n_runs, buffer_size, result_arrays):
        self.assertEqual(len(result_arrays), n_runs*buffer_size)
        for i in range(n_runs):
            for j in range(buffer_size):
                expected_array = np.zeros(self.data_shape, dtype=self.dtype)
                expected_array[:] = i*buffer_size + j + 1
                actual_array = result_arrays[j+i*buffer_size]
                np.testing.assert_equal(actual_array, expected_array)

    def test_in_producer_faster_than_consumer_lock_based(self):
        aux_data_size = 1000
        aux_data = cp.arange(0, aux_data_size*aux_data_size)
        aux_data = aux_data.reshape((aux_data_size, aux_data_size))

        data_shape = (10, 10)
        result_arrays = []

        def compute_heavy_on_aux_data(data):
            res = aux_data*cp.int32(2)
            # And do the regular stuff on on the input data
            return (data + 1, )

        def copy_result(element):
            result_arrays.append(element.data.copy())
            element.release()

        buffer_size = 2
        runner = self.__create_runner(
            data_shape=data_shape,
            pipeline=compute_heavy_on_aux_data,
            callbacks=copy_result)
        # Run
        n_runs = 10000
        self.__run_increment_sync(self.in_buffer, n_runs=n_runs)
        # Verify THE result arrays are as expected.
        self.__verify_increment(n_runs=n_runs, buffer_size=buffer_size,
                                result_arrays=result_arrays)

    def test_in_producer_faster_than_consumer_async(self):
        # The sized are reversed - we are doing some calculations on a small
        # input array, and transferring huge input data.

        aux_data_size = 10
        aux_data = cp.arange(0, aux_data_size*aux_data_size)
        aux_data = aux_data.reshape((aux_data_size, aux_data_size))

        data_shape = (1000, 1000)
        result_arrays = []

        def compute_lightly_on_aux_data(data):
            res = aux_data + 1
            return (data, )  # No computation on input data.

        def copy_result(element):
            result_arrays.append(element.data.copy())
            element.release()

        runner = self.__create_runner(
            data_shape=data_shape,
            pipeline=compute_lightly_on_aux_data,
            callbacks=copy_result,
            buffer_type="async")
        # Run
        n_runs = 20
        with self.assertRaisesRegex(ValueError, "override") as ctx:
            self.__run_increment_sync(self.in_buffer, n_runs=n_runs)

    def test_multi_output_pipeline(self):
        data_shape = (1000, 1000)
        results_1 = deque()
        results_2 = deque()
        pipeline = Pipeline(
            steps=(
                Lambda(lambda data: data+1),
                Pipeline(
                    steps=(
                        Lambda(lambda data: data+1),
                    ),
                    placement="GPU:0"
                ),
                Lambda(lambda data: data)  # Identity function to bypass results
            ),
            placement="GPU:0"
        )
        pipeline.prepare(MetadataMock(input_shape=data_shape, dtype=cp.int32))

        def copy_result_1(element):
            results_1.append(element.data.copy())
            element.release()

        def copy_result_2(element):
            results_2.append(element.data.copy())
            element.release()

        buffer_size = 2
        runner = self.__create_runner(
            data_shape=data_shape,
            pipeline=pipeline,
            n_out_buffers=2,
            callbacks=(copy_result_1, copy_result_2))
        # Run
        n_runs = 500
        self.__run_increment_sync(self.in_buffer, n_runs=n_runs)
        # Verify result 1
        exp_value = 2
        for array in results_1:
            expected_array = np.zeros(data_shape, dtype=self.dtype) + exp_value
            np.testing.assert_equal(expected_array, array)
            exp_value += 1
        exp_value = 1
        for array in results_2:
            expected_array = np.zeros(data_shape, dtype=self.dtype) + exp_value
            np.testing.assert_equal(expected_array, array)
            exp_value += 1


if __name__ == "__main__":
    unittest.main()
