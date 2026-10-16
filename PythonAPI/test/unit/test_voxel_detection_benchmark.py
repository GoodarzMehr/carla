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
    def run_benchmark(self, arguments=(), payload=None, frame_offset=0,
                      error=None, argument_error=False):
        # The blueprint exposes only current sensor attributes.
        attributes = {'range': 50.0, 'voxel_size': 0.5, 'lower_limit': -5.0,
                      'upper_limit': 10.0, 'chunk_x': 4, 'chunk_y': 4, 'chunk_z': 8,
                      'draw_debug': False,
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
            sensor = mock.Mock(id=len(sensors) + 1, emitted_frames=[])
            sensors.append(sensor)
            return sensor

        def tick(_timeout):
            frame = world.tick.call_count
            for sensor in sensors:
                sensor.emitted_frames.append(frame)
                raw_data = payload if payload is not None else (
                    b'\x02\x03\x04' if frame == 1 else b'\x00\x01\x00')
                sensor.listen.call_args.args[0](SimpleNamespace(
                    frame=frame + frame_offset, raw_data=raw_data))
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
                elif error:
                    with self.assertRaisesRegex(RuntimeError, error):
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
        result = None if error or argument_error else json.loads(output.splitlines()[-1])
        return sensors, result, output

    def test_benchmark_measures_one_sensor_and_excludes_warmup(self):
        sensors, result, output = self.run_benchmark()
        self.assertEqual(len(sensors), 1)
        self.assertEqual(sensors[0].emitted_frames, [1, 2, 3])
        self.assertEqual(result['frames'], 2)
        self.assertEqual(result['warmup'], 1)
        self.assertEqual(result['voxel_count'], 3)
        self.assertEqual(result['occupied_voxels'], 1.0)
        self.assertEqual(result['chunks'], [4, 4, 8])
        self.assertIs(result['attributes']['use_complex_collision'], True)
        self.assertIn('Capture latency:', output)

    def test_simbev_preset_and_attribute_override_remain_typed(self):
        _, result, _ = self.run_benchmark(
            ['--preset', 'simbev', '--attribute', 'use_complex_collision=false'])
        self.assertEqual(result['attributes']['range'], 64.0)
        self.assertEqual(result['attributes']['voxel_size'], 0.1)
        self.assertEqual(result['chunks'], [4, 4, 8])
        self.assertIs(result['attributes']['use_complex_collision'], False)

    def test_empty_payload_and_future_frame_restore_settings_and_destroy_sensor(self):
        for options in ({'payload': b'', 'error': 'empty voxel payload'},
                        {'frame_offset': 1, 'error': 'another client may be ticking'}):
            with self.subTest(options=options):
                sensors, _, _ = self.run_benchmark(**options)
                self.assertEqual(len(sensors), 1)
                self.assertEqual(sensors[0].emitted_frames, [1])

    def test_receive_frame_discards_stale_samples_and_reports_timeout(self):
        queue = benchmark.Queue()
        queue.put(SimpleNamespace(frame=2))
        expected = SimpleNamespace(frame=3)
        queue.put(expected)
        self.assertIs(benchmark.receive_frame(queue, 3, 0.01), expected)
        with self.assertRaisesRegex(RuntimeError, 'Timed out waiting for sensor frame 4'):
            benchmark.receive_frame(queue, 4, 0)

    def test_invalid_grid_is_rejected_before_changing_world_settings(self):
        for attribute in ('voxel_size=0', 'chunk_z=0'):
            with self.subTest(attribute=attribute):
                self.run_benchmark(['--attribute', attribute], argument_error=True)


if __name__ == '__main__':
    unittest.main()
