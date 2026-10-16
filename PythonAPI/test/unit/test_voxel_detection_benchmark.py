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
        # Removed implementation-selection flags deliberately do not exist.
        attributes = {'range': 50.0, 'voxel_size': 0.5, 'lower_limit': -5.0,
                      'upper_limit': 10.0, 'chunk_x': 4, 'chunk_y': 4, 'chunk_z': 8,
                      'super_chunk_factor': 2, 'draw_debug': False,
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
            sensor = mock.Mock(id=len(sensors) + 1, factor=attributes['super_chunk_factor'])
            sensors.append(sensor)
            return sensor

        def tick(_timeout):
            frame = world.tick.call_count
            for sensor in sensors:
                payload = b'\x00\x01\x00'
                if mismatch and sensor.factor != 1:
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

    def test_default_benchmarks_one_sensor_without_removed_flags(self):
        sensors, result, output = self.run_benchmark()
        self.assertEqual([sensor.factor for sensor in sensors], [2])
        self.assertEqual(result['attributes']['super_chunk_factor'], 2)
        self.assertIsInstance(result['attributes']['super_chunk_factor'], int)
        self.assertNotIn('super_chunk_comparison', result)
        self.assertIn('Capture latency:', output)

    def test_comparison_checks_matching_frames_and_cleans_both_sensors(self):
        sensors, result, output = self.run_benchmark(
            ['--compare-super-chunks', '--attribute', 'super_chunk_factor=4'])
        self.assertEqual([sensor.factor for sensor in sensors], [1, 4])
        self.assertEqual(result['attributes']['super_chunk_factor'], 4)
        self.assertEqual(result['super_chunk_comparison'], {
            'baseline_factor': 1, 'target_factor': 4, 'matched_frames': 3,
            'identical_bytes': True})
        self.assertIn('two sensors; not isolated', output)

    def test_comparison_rejects_equal_counts_with_different_bytes_and_cleans_both(self):
        sensors, _, _ = self.run_benchmark(['--compare-super-chunks'], mismatch=True)
        self.assertEqual([sensor.factor for sensor in sensors], [1, 2])

    def test_invalid_factors_are_rejected_before_changing_world_settings(self):
        for factor in ('0', '3', '16'):
            with self.subTest(factor=factor):
                self.run_benchmark(['--attribute', 'super_chunk_factor=' + factor],
                                   argument_error=True)
        self.run_benchmark(['--compare-super-chunks', '--attribute', 'super_chunk_factor=1'],
                           argument_error=True)


if __name__ == '__main__':
    unittest.main()
