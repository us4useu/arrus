import time
import unittest
from collections import deque, namedtuple
from collections.abc import Iterable
import numpy as np
from types import SimpleNamespace
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


    def __create_simple_setup(self, output_shape=(2, 2), n_updates_counter=None):
        """
        A single sequence, a single output pipeline.
        """
        a1 = np.zeros((2, 2), dtype=np.int16) + 1
        a2 = np.zeros((2, 2), dtype=np.int16) + 3
        pipeline = Pipeline(name="A", placement="/GPU:0", steps=(
            Lambda(lambda data: data+1),
        ))
        graph = Graph(
            operations={pipeline},
            dependencies={"A": "SequenceA", "Output:0": "A/Output:0"}
        )
        return self.__create_setup(elements=[(a1, ), (a2, )], graph=graph, sequences=["SequenceA"])

    def test_update_keeps_the_buffers_and_the_graph(self):
        """
        The runner update should not re-create the GPU buffers, nor the processing graph.
        """
        input_buffer, runner = self.__create_simple_setup()
        gpu_buffer, output_buffer, ops = runner.gpu_input_buffer, runner.output_buffer, runner._ops
        _, old_metadata = runner.outputs
        new_metadata = [MetadataMock(input_shape=(2, 2), dtype=np.int16, name="SequenceA")]

        buffer, metadata = runner.update(input_buffer, new_metadata)

        self.assertIs(runner.gpu_input_buffer, gpu_buffer)
        self.assertIs(runner.output_buffer, output_buffer)
        self.assertIs(runner._ops, ops)
        self.assertEqual(runner.input_metadata, new_metadata)
        # The processing should still work after the update.
        for i in range(2):
            input_buffer.produce()
            output = buffer.get()
            self.assertEqual(output[0].shape, (2, 2))

    def test_update_rejects_the_change_of_the_input_shape(self):
        input_buffer, runner = self.__create_simple_setup()
        new_metadata = [MetadataMock(input_shape=(4, 2), dtype=np.int16, name="SequenceA")]
        with self.assertRaises(ValueError):
            runner.update(input_buffer, new_metadata)

    def test_update_rejects_the_change_of_the_number_of_arrays(self):
        input_buffer, runner = self.__create_simple_setup()
        metadata = [MetadataMock(input_shape=(2, 2), dtype=np.int16, name="SequenceA")]*2
        with self.assertRaises(ValueError):
            runner.update(input_buffer, metadata)

    def test_update_calls_update_on_the_graph_operations(self):
        """
        The operations should be updated (not prepared) -- e.g. the kernels should not be re-compiled.
        """
        calls = []

        class OpMock(Lambda):
            def prepare(self, const_metadata):
                calls.append("prepare")
                return super().prepare(const_metadata)

            def update(self, const_metadata):
                calls.append("update")
                return super().update(const_metadata)

        a1 = np.zeros((2, 2), dtype=np.int16) + 1
        pipeline = Pipeline(name="A", placement="/GPU:0", steps=(
            OpMock(lambda data: data+1),
        ))
        graph = Graph(
            operations={pipeline},
            dependencies={"A": "SequenceA", "Output:0": "A/Output:0"}
        )
        input_buffer, runner = self.__create_setup(
            elements=[(a1, )], graph=graph, sequences=["SequenceA"])
        self.assertEqual(calls, ["prepare"])
        runner.update(input_buffer, [MetadataMock(input_shape=(2, 2), dtype=np.int16, name="SequenceA")])
        self.assertEqual(calls, ["prepare", "update", "prepare"])

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



class GpuConstMemoryPoolTest(unittest.TestCase):
    """Operations updated after session.set_subsequences must not exhaust the constant memory pool."""

    def make_pool(self, total_size):
        from unittest import mock
        import arrus.utils.imaging as imaging
        const_array = mock.MagicMock()
        with mock.patch.object(imaging, "_get_const_memory_array", return_value=const_array):
            pool = imaging.GpuConstMemoryPool(kernel_module=None, variable_name="test",
                                              total_size=total_size, dtype=np.float32)
        return pool, const_array

    def test_reuse_offset_does_not_reserve_more_memory(self):
        pool, const_array = self.make_pool(total_size=256)
        x = np.arange(64, dtype=np.float32)
        offset = pool.reserve_new_array(x)
        # Many more updates than the pool could hold if each one reserved a new part.
        for _ in range(100):
            self.assertEqual(pool.reserve_new_array(x, reuse_offset=offset), offset)
        self.assertEqual(pool.currently_reserved, 64)
        # Unchanged content: the constant memory is uploaded only once (by the first reservation).
        self.assertEqual(const_array.set.call_count, 1)

    def test_reuse_offset_overwrites_changed_content(self):
        pool, const_array = self.make_pool(total_size=256)
        offset = pool.reserve_new_array(np.zeros(64, dtype=np.float32))
        new = np.full(64, 3.0, dtype=np.float32)
        self.assertEqual(pool.reserve_new_array(new, reuse_offset=offset), offset)
        np.testing.assert_array_equal(pool.reference_array[offset:offset+64], new)
        self.assertEqual(const_array.set.call_count, 2)

    def test_different_length_reserves_new_part(self):
        pool, _ = self.make_pool(total_size=256)
        offset = pool.reserve_new_array(np.zeros(64, dtype=np.float32))
        new_offset = pool.reserve_new_array(np.zeros(32, dtype=np.float32), reuse_offset=offset)
        self.assertEqual(new_offset, 64)
        self.assertEqual(pool.currently_reserved, 96)

    def test_without_reuse_offset_the_pool_is_exhausted(self):
        pool, _ = self.make_pool(total_size=256)
        x = np.zeros(64, dtype=np.float32)
        for _ in range(4):
            pool.reserve_new_array(x)
        with self.assertRaises(ValueError):
            pool.reserve_new_array(x)


class DeriveSubsequenceFcmTest(unittest.TestCase):
    """The frame channel mapping of a sub-sequence, derived from the uploaded sequence's one."""

    def make_full(self, n_ops=8, n_channels=4, n_us4oems=2):
        """A mapping where each TX/RX is received by both us4OEMs (half of the channels each)."""
        us4oems = np.zeros((n_ops, n_channels), dtype=np.uint8)
        us4oems[:, n_channels//2:] = 1
        frames = np.repeat(np.arange(n_ops, dtype=np.int16)[:, None], n_channels, axis=1)
        channels = np.tile(np.arange(n_channels//2, dtype=np.int8), (n_ops, 2))
        return SimpleNamespace(us4oems=us4oems, frames=frames, channels=channels,
                               frame_offsets=np.array([0, n_ops], dtype=np.uint32),
                               n_frames=np.array([n_ops, n_ops], dtype=np.uint32), batch_size=1)

    def test_frames_are_renumbered_per_us4oem(self):
        from arrus.utils.core import derive_subsequence_fcm
        full = self.make_full()
        us4oems, frames, channels, frame_offsets, n_frames = derive_subsequence_fcm(full, [1, 4, 5])
        np.testing.assert_array_equal(frames, np.repeat(np.arange(3)[:, None], 4, axis=1))
        np.testing.assert_array_equal(us4oems, full.us4oems[[1, 4, 5]])
        np.testing.assert_array_equal(channels, full.channels[[1, 4, 5]])
        np.testing.assert_array_equal(n_frames, [3, 3])
        np.testing.assert_array_equal(frame_offsets, [0, 3])

    def test_us4oem_that_receives_only_some_of_the_selected_ops(self):
        """A TX/RX whose RX aperture is clipped to one us4OEM: the other one has no frame for it."""
        from arrus.utils.core import derive_subsequence_fcm
        full = self.make_full()
        full.channels[3, 2:] = -1     # op 3: no valid channels on us4OEM 1
        _, frames, _, frame_offsets, n_frames = derive_subsequence_fcm(full, [2, 3, 6])
        # us4OEM 0 receives all three; us4OEM 1 only ops 2 and 6, so they are its frames 0 and 1.
        np.testing.assert_array_equal(frames[:, :2], np.repeat(np.arange(3)[:, None], 2, axis=1))
        np.testing.assert_array_equal(frames[[0, 2], 2:], np.repeat(np.array([0, 1])[:, None], 2, axis=1))
        np.testing.assert_array_equal(n_frames, [3, 2])
        np.testing.assert_array_equal(frame_offsets, [0, 3])

    def test_the_whole_sequence_is_unchanged(self):
        from arrus.utils.core import derive_subsequence_fcm
        full = self.make_full()
        us4oems, frames, channels, frame_offsets, n_frames = derive_subsequence_fcm(
            full, list(range(full.frames.shape[0])))
        np.testing.assert_array_equal(frames, full.frames)
        np.testing.assert_array_equal(us4oems, full.us4oems)
        np.testing.assert_array_equal(channels, full.channels)
        np.testing.assert_array_equal(n_frames, full.n_frames)
        np.testing.assert_array_equal(frame_offsets, full.frame_offsets)

if __name__ == "__main__":
    unittest.main()
