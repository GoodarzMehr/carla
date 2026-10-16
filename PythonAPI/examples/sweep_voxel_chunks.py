#!/usr/bin/env python3

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Find the fastest tested power-of-two voxel chunk dimensions.

Each trial runs compare_voxel_detection.py with one voxel sensor. Timings cover
world.tick() and sensor delivery, with Python payload inspection excluded.
The best three triples are measured again with more frames.
Use a static scene with no other client advancing the world.

    python3 sweep_voxel_chunks.py --dry-run
    python3 sweep_voxel_chunks.py --port 18731 --output chunks.csv
    python3 sweep_voxel_chunks.py --chunk-x 4 8 16 --chunk-y 4 8 16 --chunk-z 2 4 8

Results are flushed to CSV after each child run. Ctrl+C stops the sweep and
allows the active child to destroy its sensors and restore world settings.
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import shlex
import signal
import subprocess
import sys


def power_of_two(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError('expected a positive integer power of two')
    if number < 1 or number > (1 << 30) or number & (number - 1):
        raise argparse.ArgumentTypeError('expected a positive power of two fitting a 32-bit chunk dimension')
    return number


def run_child(command):
    # A separate session lets us forward one interrupt and wait for cleanup.
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, start_new_session=True) as process:
        try:
            output, _ = process.communicate()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.communicate()
            raise
    if process.returncode:
        raise RuntimeError('Benchmark exited with code %d:\n%s' %
                           (process.returncode, output[-3000:]))
    try:
        return json.loads(output.strip().splitlines()[-1])
    except (IndexError, ValueError):
        raise RuntimeError('Benchmark did not return its final JSON result:\n' + output[-3000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--chunk-x', nargs='+', type=power_of_two, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument('--chunk-y', nargs='+', type=power_of_two, default=[2, 4, 8, 16, 32, 64])
    parser.add_argument('--chunk-z', nargs='+', type=power_of_two, default=[1, 2, 4, 8, 16])
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--preset', choices=('default', 'simbev'), default='simbev')
    parser.add_argument('--attribute', action='append', default=[], metavar='NAME=VALUE')
    parser.add_argument('--spawn-point', type=int, default=0)
    parser.add_argument('--height', type=float, default=2.0)
    parser.add_argument('--yaw', type=float, default=0.0)
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--roll', type=float, default=0.0)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--retest-top', type=int, default=3,
                        help='retest this many sweep finalists; 0 disables retesting')
    parser.add_argument('--retest-frames', type=int, default=10)
    parser.add_argument('--output', type=Path, default=Path('voxel_chunk_sweep.csv'))
    parser.add_argument('--dry-run', action='store_true', help='print the plan without connecting')
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.retest_top < 0 or args.retest_frames < 1:
        parser.error('frames must be positive; warmup and retest-top must be nonnegative')
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('timeout must be finite and positive')

    controlled = {'chunk_x', 'chunk_y', 'chunk_z', 'sensor_tick',
                  'draw_debug', 'show_calculation_time'}
    for item in args.attribute:
        name, separator, value = item.partition('=')
        if not separator or not name or not value:
            parser.error('--attribute requires NAME=VALUE')
        if name in controlled:
            parser.error('%s is controlled by this sweep' % name)

    axes = [list(dict.fromkeys(axis)) for axis in (args.chunk_x, args.chunk_y, args.chunk_z)]
    triples = list(itertools.product(*axes))
    random.Random(args.seed).shuffle(triples)
    baseline = (4, 4, 8)
    if baseline in triples:
        triples.remove(baseline)
        triples.insert(0, baseline)
    retest_count = min(args.retest_top, len(triples))
    print('%d triples; %d frames + %d warmup each; %d retests of %d frames' %
          (len(triples), args.frames, args.warmup, retest_count, args.retest_frames), flush=True)

    helper = Path(__file__).with_name('compare_voxel_detection.py')
    forwarded = ['--host', args.host, '--port', str(args.port), '--timeout', str(args.timeout),
                 '--preset', args.preset, '--json']
    for name in ('spawn_point', 'height', 'yaw', 'pitch', 'roll'):
        forwarded.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    for attribute in args.attribute + ['draw_debug=false', 'show_calculation_time=false']:
        forwarded.extend(['--attribute', attribute])

    def command_for(chunks, frames, warmup):
        command = [sys.executable, str(helper)] + forwarded + [
            '--frames', str(frames), '--warmup', str(warmup)]
        for name, value in zip(('chunk_x', 'chunk_y', 'chunk_z'), chunks):
            command.extend(['--attribute', '%s=%d' % (name, value)])
        return command

    if args.dry_run:
        print('Axes X=%s Y=%s Z=%s; reproducible seed=%d' % (*axes, args.seed))
        print('First trial: ' + ' '.join(shlex.quote(part) for part in
                                       command_for(triples[0], args.frames, args.warmup)))
        print('CSV: %s (no file written)' % args.output)
        return 0

    fields = ['stage', 'status', 'chunk_x', 'chunk_y', 'chunk_z', 'preset', 'map',
              'transform', 'attributes', 'existing_voxel_sensors',
              'frames', 'warmup', 'voxel_count',
              'occupied_voxels', 'capture_median_ms', 'capture_min_ms', 'capture_max_ms',
              'tick_median_ms', 'error']
    with args.output.open('w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        csv_file.flush()
        completed = 0
        total_runs = len(triples) + retest_count

        def trial(stage, chunks, frames, warmup):
            nonlocal completed
            print('[%d/%d] %s %dx%dx%d, %d frames' %
                  (completed + 1, total_runs, stage, *chunks, frames), flush=True)
            row = dict(zip(('chunk_x', 'chunk_y', 'chunk_z'), chunks))
            row.update(stage=stage, preset=args.preset, frames=frames, warmup=warmup)
            try:
                result = run_child(command_for(chunks, frames, warmup))
                if result['chunks'] != list(chunks):
                    raise RuntimeError('Benchmark used unexpected chunks: %s' % result['chunks'])
                if not math.isfinite(result['capture_median_ms']):
                    raise RuntimeError('Benchmark returned a nonfinite capture time')
                if result.get('existing_voxel_sensors'):
                    print('Other voxel sensors contributing to capture time: %s' %
                          result['existing_voxel_sensors'], flush=True)
                row.update(result, status='complete')
                row['attributes'] = json.dumps(result['attributes'], sort_keys=True)
                row['existing_voxel_sensors'] = json.dumps(result.get('existing_voxel_sensors', []))
            except (Exception, KeyboardInterrupt) as error:
                row.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'error',
                           error=str(error) or 'Interrupted')
                writer.writerow(row)
                csv_file.flush()
                raise
            writer.writerow(row)
            csv_file.flush()
            completed += 1
            print('  median %.2f ms' % result['capture_median_ms'], flush=True)
            return result

        results = [trial('sweep', chunks, args.frames, args.warmup) for chunks in triples]
        finalists = sorted(results, key=lambda row: row['capture_median_ms'])[:retest_count]
        retests = [trial('retest', result['chunks'], args.retest_frames, args.warmup)
                   for result in finalists]
        winner = min(retests or results, key=lambda row: row['capture_median_ms'])

    print('Best tested chunks: %dx%dx%d; median capture %.2f ms (%s)' %
          (*winner['chunks'], winner['capture_median_ms'], 'retest' if retests else 'sweep'))
    attributes = dict(zip(('chunk_x', 'chunk_y', 'chunk_z'), map(str, winner['chunks'])))
    if 'use_triangle_reuse' in winner['attributes']:
        attributes['use_triangle_reuse'] = str(winner['attributes']['use_triangle_reuse']).lower()
    print('SimBEV attributes: ' + json.dumps(attributes))
    print('Results: %s' % args.output)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print('\nStopped; completed trial rows remain in CSV.', file=sys.stderr)
        sys.exit(130)
    except (OSError, RuntimeError, KeyError, TypeError) as error:
        print('Sweep stopped: %s' % error, file=sys.stderr)
        sys.exit(1)
