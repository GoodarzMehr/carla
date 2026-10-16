# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / 'examples' / 'compare_voxel_detection.py'
SPEC = importlib.util.spec_from_file_location('voxel_detection_benchmark', SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class TestVoxelDetectionBenchmark(unittest.TestCase):
    def run_benchmark(self, arguments=(), mismatch=False, argument_error=False):
        # The blueprint exposes only current sensor attributes.
        attributes = {'range': 50.0, 'voxel_size': 0.5, 'lower_limit': -5.0,
                      'upper_limit': 10.0, 'chunk_x': 4, 'chunk_y': 4, 'chunk_z': 8,
                      'use_triangle_reuse': False, 'use_gpu': False, 'draw_debug': False,
                      'show_calculation_time': False, 'use_complex_collision': True,
                      'ignore_self': False}
        blueprint = mock.Mock()

        def set_attribute(name, value):
            kind = type(attributes[name])
            attributes[name] = value.lower() == 'true' if kind is bool else kind(value)

        def get_attribute(name):
            value = attributes[name]
            return SimpleNamespace(type=type(value), as_bool=lambda: value,
                                   as_int=lambda: value, as_float=lambda: value,
                                   as_str=lambda: value)

        blueprint.set_attribute.side_effect = set_attribute
        blueprint.get_attribute.side_effect = get_attribute
        blueprint.has_attribute.side_effect = attributes.__contains__
        original_settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)
        settings = SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=None)
        transform = SimpleNamespace(location=SimpleNamespace(z=0.0),
                                    rotation=SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0))
        world = mock.Mock()
        world.get_blueprint_library.return_value.find.return_value = blueprint
        world.get_actors.return_value.filter.return_value = []
        world.get_settings.side_effect = [original_settings, settings]
        world.get_map.return_value = SimpleNamespace(
            name='FakeTown', get_spawn_points=lambda: [transform])
        sensors = []

        def spawn_actor(_blueprint, _transform):
            sensor = mock.Mock(id=len(sensors) + 1,
                               triangle_reuse=attributes['use_triangle_reuse'],
                               use_gpu=attributes['use_gpu'], emitted_frames=[])
            sensors.append(sensor)
            return sensor

        def tick(_timeout):
            frame = world.tick.call_count
            for sensor in sensors:
                sensor.emitted_frames.append(frame)
                payload = b'\x00\x01\x00'
                if mismatch and sensor is sensors[-1]:
                    payload = b'\x00\x00\x01'
                sensor.listen.call_args.args[0](SimpleNamespace(frame=frame, raw_data=payload))
            return frame

        world.spawn_actor.side_effect = spawn_actor
        world.tick.side_effect = tick
        carla = SimpleNamespace(Client=mock.Mock(), ActorAttributeType=SimpleNamespace(
            Bool=bool, Int=int, Float=float, String=str))
        carla.Client.return_value.get_world.return_value = world
        printed = io.StringIO()
        command = [str(SCRIPT), '--frames', '2', '--warmup', '1', '--json'] + list(arguments)
        with mock.patch.dict(sys.modules, {'carla': carla}):
            with mock.patch.object(sys, 'argv', command), contextlib.redirect_stdout(printed):
                if argument_error:
                    with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        benchmark.main()
                    self.assertEqual(error.exception.code, 2)
                elif mismatch:
                    with self.assertRaisesRegex(RuntimeError, 'voxel bytes differ'):
                        benchmark.main()
                else:
                    benchmark.main()
        for sensor in sensors:
            sensor.stop.assert_called_once_with()
            sensor.destroy.assert_called_once_with()
        if argument_error:
            self.assertFalse(sensors)
            world.apply_settings.assert_not_called()
        else:
            self.assertIs(world.apply_settings.call_args.args[0], original_settings)
        output = printed.getvalue()
        result = None if mismatch or argument_error else json.loads(output.splitlines()[-1])
        return sensors, result, output

    def test_default_benchmarks_one_sensor_with_triangle_reuse_disabled(self):
        sensors, result, output = self.run_benchmark()
        self.assertEqual([sensor.triangle_reuse for sensor in sensors], [False])
        self.assertIs(result['attributes']['use_triangle_reuse'], False)
        self.assertIs(result['attributes']['use_gpu'], False)
        self.assertEqual([sensor.use_gpu for sensor in sensors], [False])
        self.assertNotIn('triangle_reuse_comparison', result)
        self.assertNotIn('gpu_comparison', result)
        self.assertIn('Capture latency:', output)

    def test_single_sensor_can_benchmark_triangle_reuse_enabled(self):
        sensors, result, _ = self.run_benchmark(['--attribute', 'use_triangle_reuse=true'])
        self.assertEqual([sensor.triangle_reuse for sensor in sensors], [True])
        self.assertIs(result['attributes']['use_triangle_reuse'], True)
        self.assertNotIn('triangle_reuse_comparison', result)

    def test_comparison_checks_matching_frames_and_cleans_both_sensors(self):
        sensors, result, output = self.run_benchmark(
            ['--compare-triangle-reuse', '--attribute', 'use_triangle_reuse=false',
             '--attribute', 'use_gpu=true'])
        self.assertEqual([sensor.triangle_reuse for sensor in sensors], [False, True])
        self.assertEqual([sensor.use_gpu for sensor in sensors], [False, False])
        self.assertIs(result['attributes']['use_triangle_reuse'], True)
        self.assertEqual(result['triangle_reuse_comparison'], {
            'baseline_enabled': False, 'target_enabled': True, 'matched_frames': 3,
            'identical_bytes': True})
        self.assertIn('two sensors; not isolated', output)

    def test_single_sensor_can_benchmark_gpu_without_changing_triangle_flag(self):
        sensors, result, output = self.run_benchmark(['--attribute', 'use_gpu=true'])
        self.assertEqual([sensor.use_gpu for sensor in sensors], [True])
        self.assertEqual([sensor.triangle_reuse for sensor in sensors], [False])
        self.assertIs(result['attributes']['use_gpu'], True)
        self.assertNotIn('gpu_comparison', result)
        self.assertIn('Capture latency:', output)

    def test_gpu_comparison_matches_frames_against_cpu_triangle_reuse(self):
        sensors, result, output = self.run_benchmark(
            ['--compare-gpu', '--attribute', 'use_triangle_reuse=false',
             '--attribute', 'use_gpu=false'])
        self.assertEqual([sensor.use_gpu for sensor in sensors], [False, True])
        self.assertEqual([sensor.triangle_reuse for sensor in sensors], [True, True])
        self.assertIs(result['attributes']['use_gpu'], True)
        self.assertIs(result['attributes']['use_triangle_reuse'], True)
        self.assertEqual(result['gpu_comparison'], {
            'baseline_use_gpu': False, 'target_use_gpu': True,
            'cpu_triangle_reuse': True, 'matched_frames': 3, 'identical_bytes': True})
        self.assertNotIn('triangle_reuse_comparison', result)
        self.assertIn('two sensors; not isolated', output)

    def test_gpu_comparison_rejects_same_count_different_bytes_during_warmup(self):
        sensors, _, _ = self.run_benchmark(['--compare-gpu'], mismatch=True)
        self.assertEqual([sensor.use_gpu for sensor in sensors], [False, True])
        self.assertEqual([sensor.emitted_frames for sensor in sensors], [[1], [1]])

    def test_comparison_flags_cannot_be_combined(self):
        self.run_benchmark(['--compare-gpu', '--compare-triangle-reuse'], argument_error=True)

    def test_comparison_rejects_equal_counts_with_different_bytes_and_cleans_both(self):
        sensors, _, _ = self.run_benchmark(['--compare-triangle-reuse'], mismatch=True)
        self.assertEqual([sensor.triangle_reuse for sensor in sensors], [False, True])

    def test_invalid_grid_is_rejected_before_changing_world_settings(self):
        for attribute in ('voxel_size=0', 'chunk_z=0'):
            with self.subTest(attribute=attribute):
                self.run_benchmark(['--attribute', attribute], argument_error=True)


if __name__ == '__main__':
    unittest.main()
