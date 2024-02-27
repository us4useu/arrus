import time
import unittest
from collections import deque, namedtuple
from collections.abc import Iterable
import numpy as np
import cupy as cp
from dataclasses import dataclass

from arrus.utils.imaging import (
    Buffer, BufferElement, ProcessingRunner, Pipeline, Lambda, Processing,
    ProcessingBuffer
)

@dataclass
class MetadataMock:
    input_shape: tuple
    dtype: object


class PipelineMock:

    def __init__(self, func, n_outputs, output_shape, output_dtype):
        self.output_dtype = output_dtype
        self.func = func
        self.n_outputs = n_outputs
        self.output_shape = output_shape

    def prepare(self, const_metadata):
        return [MetadataMock(input_shape=self.output_shape, dtype=self.output_dtype)
                for i in range(self.n_outputs)]

    def process(self, data):
        return self.func(data)

    def get_parameters(self):
        return {}

    def __call__(self, data):
        return self.process(data)


class InputBufferElementMock:

    def __init__(self, shape, dtype):
        self.data = np.zeros(shape, dtype=dtype)
        self.size = self.data.nbytes


class InputBufferMock:
    """
    Mock class for device buffer.
    """
    def __init__(self, n_elements, shape, dtype):
        self.elements = [InputBufferElementMock(shape, dtype)
                         for _ in range(n_elements)]
        self.callbacks = []
        self.i = 0
        self.n = n_elements

    def append_on_new_data_callback(self, func):
        self.callbacks.append(func)

    def produce(self):
        for cb in self.callbacks:
            cb(self.elements[self.i])
        self.i = (self.i + 1) % self.n

    def acquire(self, i):
        return self.elements[i]


class ProcessingRunnerTestCase(unittest.TestCase):
    """
    NOTE: these tests requires Host computer with GPU installed.
    NOTE: the speed of producer and consumer is hardware dependent and should
    be should be treated as a rough assumptions.
    """

    def setUp(self) -> None:
        super().setUp()
        self.dtype = np.int32

    def tearDown(self) -> None:
        super().tearDown()
        if hasattr(self, "runner"):
            self.in_buffer = None
            self.gpu_buffer = None
            self.out_buffer = None
            self.runner.close()
            self.runner = None

    def __create_runner(self, data_shape,
                        in_buffer_size=2, gpu_buffer_size=2, out_buffer_size=2,
                        n_out_buffers=1,
                        pipeline=None, callback=None,
                        buffer_type="locked"):
        self.data_shape = data_shape

        self.in_buffer = InputBufferMock(
            n_elements=in_buffer_size,
            shape=data_shape,
            dtype=self.dtype)

        input_buffer_spec = ProcessingBuffer(
            size=gpu_buffer_size,
            type=buffer_type
        )
        output_buffer_spec = ProcessingBuffer(
            size=out_buffer_size,
            type=buffer_type
        )
        metadata = MetadataMock(
            input_shape=self.data_shape,
            dtype=self.dtype
        )

        self.runner = ProcessingRunner(
            input_buffer=self.in_buffer,
            const_metadata=metadata,
            processing=Processing(
                input_buffer=input_buffer_spec,
                output_buffer=output_buffer_spec,
                pipeline=pipeline,
                callback=callback
            ))
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
            return data + 1,

        pipeline = PipelineMock(
            func=compute_heavy_on_aux_data,
            n_outputs=1,
            output_shape=data_shape,
            output_dtype=self.dtype
        )

        def callback(elements):
            result_arrays.append(elements[0].data.copy())
            elements[0].release()

        buffer_size = 2
        runner = self.__create_runner(
            data_shape=data_shape,
            pipeline=pipeline,
            callback=callback)
        # Run
        n_runs = 10000
        self.__run_increment_sync(self.in_buffer, n_runs=n_runs)
        # Verify THE result arrays are as expected.
        self.__verify_increment(n_runs=n_runs, buffer_size=buffer_size,
                                result_arrays=result_arrays)

    # NOTE: the below is no longer valid for arrus.utils.imaging.PipelineRunner,
    # however the logic should be used in ARRUS v0.9.0 C++ Pipeline
    # implementation.
    # def test_in_producer_faster_than_consumer_async(self):
    #     # The sized are reversed - we are doing some calculations on a small
    #     # input array, and transferring huge input data.
    #
    #     aux_data_size = 10
    #     aux_data = cp.arange(0, aux_data_size*aux_data_size)
    #     aux_data = aux_data.reshape((aux_data_size, aux_data_size))
    #
    #     data_shape = (1000, 1000)
    #     result_arrays = []
    #
    #     def compute_lightly_on_aux_data(data):
    #         res = aux_data + 1
    #         return (data, )  # No computation on input data.
    #
    #     pipeline = PipelineMock(compute_lightly_on_aux_data, n_outputs=1,
    #                             output_shape=data_shape,
    #                             output_dtype=self.dtype)
    #
    #     def copy_result(elements):
    #         result_arrays.append(elements[0].data.copy())
    #         elements[0].release()
    #
    #     runner = self.__create_runner(
    #         data_shape=data_shape,
    #         pipeline=pipeline,
    #         callback=copy_result,
    #         buffer_type="async")
    #     # Run
    #     n_runs = 20
    #     with self.assertRaisesRegex(ValueError, "override") as ctx:
    #         self.__run_increment_sync(self.in_buffer, n_runs=n_runs)

    def test_multi_output_pipeline(self):
        data_shape = (1000, 1000)
        results = deque()
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

        def copy_results(elements):
            copies = []
            for element in elements:
                copies.append(element.data.copy())
                element.release()
            results.append(copies)

        buffer_size = 2
        runner = self.__create_runner(
            data_shape=data_shape,
            pipeline=pipeline,
            n_out_buffers=2,
            callback=copy_results)
        # Run
        n_runs = 500
        self.__run_increment_sync(self.in_buffer, n_runs=n_runs)
        # Verify result 1
        for i, (array1, array2) in enumerate(results):
            # +1 -> +1 -> output
            expected_array_1 = np.zeros(data_shape, dtype=self.dtype) + i + 1
            np.testing.assert_equal(expected_array_1, array1)
            # +1 -> output
            expected_array_2 = np.zeros(data_shape, dtype=self.dtype) + i + 2
            np.testing.assert_equal(expected_array_2, array2)


if __name__ == "__main__":
    unittest.main()
