# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[2] / 'examples' / 'sweep_voxel_tiles.py'
SPEC = importlib.util.spec_from_file_location('voxel_tile_sweep', SCRIPT)
sweep = importlib.util.module_from_spec(SPEC)
with mock.patch.dict(sys.modules, {'carla': None}):
    SPEC.loader.exec_module(sweep)


def result_for(command, median=10.0):
    attributes = dict(command[index + 1].split('=', 1)
                      for index, value in enumerate(command) if value == '--attribute')
    tiles = [int(attributes['tile_' + axis]) for axis in 'xyz']
    return {'tiles': tiles, 'attributes': attributes,
            'capture_median_ms': median, 'transform': {'x': 0.0, 'yaw': 37.0}}


class TestVoxelTileSweep(unittest.TestCase):
    def run_sweep(self, arguments, child, error=None):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.csv'
            printed = io.StringIO()
            with mock.patch.object(sweep, 'run_child', side_effect=child) as run:
                with mock.patch.object(sys, 'argv', [str(SCRIPT), '--output', str(output)] + arguments):
                    with contextlib.redirect_stdout(printed):
                        if error:
                            with self.assertRaisesRegex(RuntimeError, error):
                                sweep.main()
                        else:
                            self.assertEqual(sweep.main(), 0)
                calls = [call.args[0] for call in run.call_args_list]
            with output.open(newline='') as csv_file:
                rows = list(csv.DictReader(csv_file))
        return rows, calls, printed.getvalue()

    def test_power_of_two_rejects_invalid_dimensions(self):
        for value in ('1', '2', '32', '1024'):
            self.assertEqual(sweep.power_of_two(value), int(value))
        for value in ('0', '-2', '3', '6', '1.5', 'nan', 'text', str(1 << 31)):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                sweep.power_of_two(value)

    def test_sweep_covers_unique_axis_product_with_baseline_first(self):
        rows, calls, printed = self.run_sweep(
            ['--tile-x', '16', '32', '32', '--tile-y', '16', '32',
             '--tile-z', '64', '128', '128', '--retest-top', '0'],
            result_for)
        triples = [tuple(result_for(command)['tiles']) for command in calls]
        expected = {(x, y, z) for x in (16, 32) for y in (16, 32) for z in (64, 128)}
        self.assertEqual(triples[0], (32, 32, 128))
        self.assertEqual(len(triples), len(expected))
        self.assertEqual(set(triples), expected)
        self.assertEqual({(row['stage'], row['status']) for row in rows}, {('sweep', 'complete')})
        self.assertIn('Best tested tiles: 32x32x128', printed)

    def test_ground_offset_is_forwarded_instead_of_height(self):
        _, calls, _ = self.run_sweep(
            ['--tile-x', '4', '--tile-y', '4', '--tile-z', '8', '--retest-top', '0',
             '--ground-offset', '-0.537', '--attribute', 'use_complex_collision=false'],
            result_for)
        command = calls[0]
        self.assertEqual(command[command.index('--ground-offset') + 1], '-0.537')
        self.assertNotIn('--height', command)
        self.assertIn('use_complex_collision=false', command)
        _, calls, _ = self.run_sweep(
            ['--tile-x', '4', '--tile-y', '4', '--tile-z', '8', '--retest-top', '0'], result_for)
        self.assertEqual(calls[0][calls[0].index('--height') + 1], '2.0')
        self.assertNotIn('--ground-offset', calls[0])

    def test_retest_changes_winner(self):
        def child(command):
            result = result_for(command)
            x = result['tiles'][0]
            frames = int(command[command.index('--frames') + 1])
            result['capture_median_ms'] = (
                {2: 40.0, 4: 15.0}[x] if frames == 7 else
                {2: 10.0, 4: 20.0, 8: 30.0}[x])
            return result

        rows, calls, printed = self.run_sweep(
            ['--tile-x', '2', '4', '8', '--tile-y', '4', '--tile-z', '8',
             '--retest-top', '2', '--retest-frames', '7'], child)
        self.assertEqual([row['stage'] for row in rows],
                         ['sweep'] * 3 + ['retest'] * 2)
        self.assertEqual([result_for(command)['tiles'][0] for command in calls[3:5]], [2, 4])
        self.assertEqual(result_for(calls[-1])['tiles'], [4, 4, 8])
        self.assertIn('4x4x8; median capture 15.00 ms (retest)', printed)

    def test_sweep_preserves_requested_collision_setting(self):
        rows, calls, printed = self.run_sweep(
            ['--tile-x', '4', '--tile-y', '4', '--tile-z', '8',
             '--retest-top', '0', '--attribute', 'use_complex_collision=false'], result_for)
        self.assertEqual(result_for(calls[0])['attributes']['use_complex_collision'], 'false')
        self.assertIn('"use_complex_collision": "false"', rows[0]['attributes'])
        self.assertIn('Best tested tiles: 4x4x8', printed)

    def test_failed_trial_preserves_csv_results(self):
        def child(command):
            result = result_for(command)
            if result['tiles'][0] == 2:
                raise RuntimeError('Sensor frame timed out')
            return result

        rows, calls, _ = self.run_sweep(
            ['--tile-x', '4', '2', '--tile-y', '4', '--tile-z', '8', '--retest-top', '0'],
            child, error='Sensor frame timed out')
        self.assertEqual(len(calls), 2)
        self.assertEqual([(row['stage'], row['status']) for row in rows],
                         [('sweep', 'complete'), ('sweep', 'error')])
        self.assertIn('Sensor frame timed out', rows[-1]['error'])

    def test_child_reads_final_json_and_waits_for_interrupt_cleanup(self):
        process = mock.MagicMock()
        process.__enter__.return_value = process
        process.returncode = 0
        process.communicate.return_value = ('Trial complete\n{"tiles": [4, 4, 8]}\n', None)
        with mock.patch.object(sweep.subprocess, 'Popen', return_value=process):
            self.assertEqual(sweep.run_child(['benchmark']), {'tiles': [4, 4, 8]})
            process.communicate.side_effect = [KeyboardInterrupt, ('Cleanup complete', None)]
            with self.assertRaises(KeyboardInterrupt):
                sweep.run_child(['benchmark'])
        process.send_signal.assert_called_once_with(signal.SIGINT)
        self.assertEqual(process.communicate.call_count, 3)


if __name__ == '__main__':
    unittest.main()
