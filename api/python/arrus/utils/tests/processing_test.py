import time
import unittest
from collections import deque, namedtuple
from collections.abc import Iterable
import numpy as np
import cupy as cp
from dataclasses import dataclass

from arrus.utils.imaging import (
    Buffer, BufferElement, ProcessingRunner, Pipeline, Lambda, Processing,
    ProcessingBufferDef, Graph
)

@dataclass
class SequenceMock:
    name: str

@dataclass
class ContextMock:
    sequence: SequenceMock

@dataclass
class MetadataMock:
    input_shape: tuple
    dtype: object
    name: str

    @property
    def context(self):
        return ContextMock(SequenceMock(name=self.name))

    def copy(self, **kwargs):
        d = dict(input_shape=self.input_shape, dtype=self.dtype, name=self.name)
        return MetadataMock(**{**kwargs, **d})


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

    def __init__(self, array, arrays):
        self.array = array
        self.arrays = arrays
        self.size = self.array.nbytes


class InputBufferMock:
    """
    Mock class for device buffer.
    """
    def __init__(self, data, array_views):
        """
        :param element_arrays: element arrays: a list of tuple [(array_1, array_2)...]
        """
        self.elements = [InputBufferElementMock(d, a) for d, a in zip(data, array_views)]
        self.callbacks = []
        self.i = 0
        self.n = len(self.elements)

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

    def __create_setup(
            self, elements, sequences,
            in_buffer_size=2,
            gpu_buffer_size=2,
            out_buffer_size=2,
            graph=None, callback=None,
            buffer_type="locked"):

        # arrays: a list of tuples [(a1, a2,..), ...]
        data = []
        arrays = []
        array_definitions = []
        for element in elements:
            n_bytes = np.sum([array.nbytes for array in element])
            element_data = np.empty((n_bytes, ), dtype=np.uint8)
            views = []
            addr = 0
            for i, array in enumerate(element):
                array_size = array.nbytes
                view = element_data[addr:(addr+array_size)]
                addr += array_size
                view = view.view(array.dtype).reshape(array.shape)
                view[:] = array
                views.append(view)
                if i == 0:
                    array_definitions.append((array.shape, array.dtype))
            arrays.append(views)
            data.append(element_data)
        self.in_buffer = InputBufferMock(data, arrays)
        input_buffer_def = ProcessingBufferDef(
            size=gpu_buffer_size,
            type=buffer_type
        )
        output_buffer_def = ProcessingBufferDef(
            size=out_buffer_size,
            type=buffer_type
        )
        metadata = [MetadataMock(input_shape=shape, dtype=dtype, name=name)
                    for (shape, dtype), name in zip(array_definitions, sequences)]

        self.runner = ProcessingRunner(
            input_buffer=self.in_buffer,
            metadata=metadata,
            processing=Processing(
                input_buffer=input_buffer_def,
                output_buffer=output_buffer_def,
                graph=graph,
                callback=callback
            ))
        return self.in_buffer, self.runner

    def test_simple_graph(self):
        sequences = ["SequenceA", "SequenceB"]
        a1 = np.zeros((2, 2), dtype=np.int16) + 1
        b1 = np.zeros((2, 2), dtype=np.int16) + 2
        a2 = np.zeros((2, 2), dtype=np.int16) + 3
        b2 = np.zeros((2, 2), dtype=np.int16) + 4

        elements = [(a1, b1), (a2, b2)]

        graph = Graph(
            operations={
                Pipeline(name="A", placement="/GPU:0", steps=(
                    Lambda(lambda data: (
                        print(f"A: {data}"),
                        data+1)[1]),
                )),
                Pipeline(name="B", placement="/GPU:0", steps=(
                    Lambda(lambda data: (
                        print(f"B: {data}"),
                        data**2)[1]),
                )),
                Pipeline(name="C", placement="/GPU:0", steps=(
                    Lambda(lambda xs: xs[0]+xs[1],
                           lambda ms: ms[0].copy(input_shape=ms[0].input_shape)),
                ))
            },
            dependencies={
                "A": "SequenceA",
                "B": "SequenceB",
                "C": ("A/Output:0", "B/Output:0"),
                "Output:0": "C/Output:0"
            }
        )
        input_buffer, runner = self.__create_setup(elements=elements, graph=graph, sequences=sequences)
        # print(runner._get_ops_sequence())
        # print(runner._target_pos)
        buffer, metadata = runner.outputs
        for i in range(3):
            input_buffer.produce()
            outputs = buffer.get()
            print(outputs)


    # def __run_increment_sync(self, buffer, n_runs):
    #     value = 0
    #     for i in range(n_runs):
    #         for j in range(len(buffer.elements)):
    #             element = buffer.acquire(j)
    #             element.data[:] = value
    #             self.runner.process(element)
    #             value += 1
    #     self.runner.sync()
    #
    # def __verify_increment(self, n_runs, buffer_size, result_arrays):
    #     self.assertEqual(len(result_arrays), n_runs*buffer_size)
    #     for i in range(n_runs):
    #         for j in range(buffer_size):
    #             expected_array = np.zeros(self.data_shape, dtype=self.dtype)
    #             expected_array[:] = i*buffer_size + j + 1
    #             actual_array = result_arrays[j+i*buffer_size]
    #             np.testing.assert_equal(actual_array, expected_array)

    # def test_in_producer_faster_than_consumer_lock_based(self):
    #     aux_data_size = 1000
    #     aux_data = cp.arange(0, aux_data_size*aux_data_size)
    #     aux_data = aux_data.reshape((aux_data_size, aux_data_size))
    #
    #     data_shape = (10, 10)
    #     result_arrays = []
    #
    #     def compute_heavy_on_aux_data(data):
    #         res = aux_data*cp.int32(2)
    #         # And do the regular stuff on on the input data
    #         return data + 1,
    #
    #     pipeline = PipelineMock(
    #         func=compute_heavy_on_aux_data,
    #         n_outputs=1,
    #         output_shape=data_shape,
    #         output_dtype=self.dtype
    #     )
    #
    #     def callback(elements):
    #         result_arrays.append(elements[0].data.copy())
    #         elements[0].release()
    #
    #     buffer_size = 2
    #     runner = self.__create_runner(
    #         data_shape=data_shape,
    #         pipeline=pipeline,
    #         callback=callback)
    #     # Run
    #     n_runs = 10000
    #     self.__run_increment_sync(self.in_buffer, n_runs=n_runs)
    #     # Verify THE result arrays are as expected.
    #     self.__verify_increment(n_runs=n_runs, buffer_size=buffer_size,
    #                             result_arrays=result_arrays)

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

    # def test_multi_output_pipeline(self):
    #     data_shape = (1000, 1000)
    #     results = deque()
    #     pipeline = Pipeline(
    #         steps=(
    #             Lambda(lambda data: data+1),
    #             Pipeline(
    #                 steps=(
    #                     Lambda(lambda data: data+1),
    #                 ),
    #                 placement="GPU:0"
    #             ),
    #             Lambda(lambda data: data)  # Identity function to bypass results
    #         ),
    #         placement="GPU:0"
    #     )
    #     pipeline.prepare(MetadataMock(input_shape=data_shape, dtype=cp.int32))
    #
    #     def copy_results(elements):
    #         copies = []
    #         for element in elements:
    #             copies.append(element.data.copy())
    #             element.release()
    #         results.append(copies)
    #
    #     buffer_size = 2
    #     runner = self.__create_runner(
    #         data_shape=data_shape,
    #         pipeline=pipeline,
    #         n_out_buffers=2,
    #         callback=copy_results)
    #     # Run
    #     n_runs = 500
    #     self.__run_increment_sync(self.in_buffer, n_runs=n_runs)
    #     # Verify result 1
    #     for i, (array1, array2) in enumerate(results):
    #         # +1 -> +1 -> output
    #         expected_array_1 = np.zeros(data_shape, dtype=self.dtype) + i + 1
    #         np.testing.assert_equal(expected_array_1, array1)
    #         # +1 -> output
    #         expected_array_2 = np.zeros(data_shape, dtype=self.dtype) + i + 2
    #         np.testing.assert_equal(expected_array_2, array2)


if __name__ == "__main__":
    unittest.main()
