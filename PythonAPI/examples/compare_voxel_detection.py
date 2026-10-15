#!/usr/bin/env python3

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Compare voxel candidate reuse against the legacy queries on matching frames.

Run against a server and Python API built from this checkout, with no other
client advancing the world. The current map and actors are retained.

    python3 compare_voxel_detection.py --yaw 37 --pitch 10
    python3 compare_voxel_detection.py --attribute use_complex_collision=false
    python3 compare_voxel_detection.py --preset simbev --mode legacy --port 29354
    python3 compare_voxel_detection.py --preset simbev --mode reuse --port 29354

For timing, run --mode legacy and --mode reuse separately on the same static
scene. Reported latency includes world.tick(), sensor processing and delivery;
compare mode measures both sensors together. It is not a kernel-only benchmark.
Use --attribute show_calculation_time=true for sensor timings in the server log.
"""

import argparse
from queue import Empty, Queue
import statistics
import time


def receive_frame(queue, frame, timeout):
    deadline = time.perf_counter() + timeout
    while True:
        try:
            data = queue.get(timeout=max(0.0, deadline - time.perf_counter()))
        except Empty:
            raise RuntimeError('Timed out waiting for sensor frame %d' % frame)
        if data.frame == frame:
            return data
        if data.frame > frame:
            raise RuntimeError('Expected sensor frame %d, received %d; '
                               'another client may be ticking' % (frame, data.frame))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--mode', choices=('compare', 'legacy', 'reuse'), default='compare')
    parser.add_argument('--preset', choices=('default', 'simbev'), default='default',
                        help='simbev: range 64 m, voxels 0.1 m, height -1..7 m, chunks 4x4x2')
    parser.add_argument('--attribute', action='append', default=[], metavar='NAME=VALUE',
                        help='override a voxel blueprint attribute; repeat as needed')
    parser.add_argument('--spawn-point', type=int, default=0)
    parser.add_argument('--height', type=float, default=2.0,
                        help='height above the selected map spawn point, in meters')
    parser.add_argument('--yaw', type=float, default=0.0,
                        help='rotation added to the selected map spawn point, in degrees')
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--roll', type=float, default=0.0)
    args = parser.parse_args()
    if args.frames < 1 or args.warmup < 0 or args.timeout <= 0:
        parser.error('frames and timeout must be positive; warmup must be nonnegative')

    attributes = {'draw_debug': 'false', 'show_calculation_time': 'false'}
    if args.preset == 'simbev':
        attributes.update({'range': '64', 'voxel_size': '0.1',
                           'lower_limit': '-1', 'upper_limit': '7',
                           'chunk_x': '4', 'chunk_y': '4', 'chunk_z': '2'})
    for item in args.attribute:
        name, separator, value = item.partition('=')
        if not separator or not name or not value:
            parser.error('--attribute requires NAME=VALUE')
        if name in ('use_candidate_reuse', 'validate_candidate_reuse', 'sensor_tick'):
            parser.error('%s is controlled by this comparison script' % name)
        attributes[name] = value

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    blueprint = world.get_blueprint_library().find('sensor.other.voxel_detection')
    for name in ('use_candidate_reuse', 'validate_candidate_reuse'):
        if not blueprint.has_attribute(name):
            raise RuntimeError('Server lacks %s; rebuild CARLA with candidate reuse' % name)
    for name, value in attributes.items():
        blueprint.set_attribute(name, value)
    names = set(attributes) | {'range', 'voxel_size', 'lower_limit', 'upper_limit',
                               'chunk_x', 'chunk_y', 'chunk_z',
                               'use_complex_collision', 'ignore_self'}
    print('mode=%s preset=%s host=%s port=%d' %
          (args.mode, args.preset, args.host, args.port))
    print('Voxel attributes: ' + ', '.join('%s=%s' % (name, blueprint.get_attribute(name))
                                          for name in sorted(names)))
    existing = world.get_actors().filter('sensor.other.voxel_detection')
    print('Existing voxel sensors included in world tick: %s' %
          (', '.join(str(sensor.id) for sensor in existing) or 'none'))
    spawn_points = world.get_map().get_spawn_points()
    if not 0 <= args.spawn_point < len(spawn_points):
        parser.error('spawn-point must be between 0 and %d' % (len(spawn_points) - 1))
    transform = spawn_points[args.spawn_point]
    transform.location.z += args.height
    transform.rotation.yaw += args.yaw
    transform.rotation.pitch += args.pitch
    transform.rotation.roll += args.roll
    print('Sensor transform: %s' % transform)

    original_settings = world.get_settings()
    sensors = []
    latencies = []
    tick_latencies = []
    occupied = 0
    voxel_count = 0
    modes = ('legacy', 'reuse') if args.mode == 'compare' else (args.mode,)
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        for mode in modes:
            blueprint.set_attribute('use_candidate_reuse', str(mode == 'reuse').lower())
            blueprint.set_attribute('validate_candidate_reuse', 'false')
            if blueprint.has_attribute('sensor_tick'):
                blueprint.set_attribute('sensor_tick', '0.0')
            sensor = world.spawn_actor(blueprint, transform)
            queue = Queue()
            sensors.append((sensor, queue))
            sensor.listen(queue.put)
            print('%s sensor id=%d use_candidate_reuse=%s validate_candidate_reuse=false' %
                  (mode, sensor.id, str(mode == 'reuse').lower()))

        for index in range(args.warmup + args.frames):
            started = time.perf_counter()
            frame = world.tick(args.timeout)
            tick_elapsed = time.perf_counter() - started
            data = [receive_frame(queue, frame, args.timeout) for _, queue in sensors]
            elapsed = time.perf_counter() - started
            payloads = [bytes(sample.raw_data) for sample in data]
            if len(payloads) == 2 and payloads[0] != payloads[1]:
                left, right = payloads
                mismatches = sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))
                first = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b),
                             min(len(left), len(right)))
                raise RuntimeError('Frame %d: %d voxel mismatches, first flat index %d; '
                                   'legacy/reuse byte lengths %d/%d' %
                                   (frame, mismatches, first, len(left), len(right)))
            if not payloads[0]:
                raise RuntimeError('Frame %d: empty voxel payload' % frame)
            if index >= args.warmup:
                latencies.append(elapsed)
                tick_latencies.append(tick_elapsed)
                voxel_count = len(payloads[0])
                occupied += voxel_count - payloads[0].count(0)

        if args.mode == 'compare':
            print('PASS: identical bytes on %d matching frames (%d warmup frames also checked)' %
                  (args.frames, args.warmup))
        print('%s: %d voxels/frame, mean occupied voxels %.1f' %
              (args.mode, voxel_count, occupied / args.frames))
        print('Capture latency: median %.2f ms, min %.2f ms, max %.2f ms' %
              (1000 * statistics.median(latencies), 1000 * min(latencies),
               1000 * max(latencies)))
        print('world.tick() median: %.2f ms (world work; not isolated sensor time)' %
              (1000 * statistics.median(tick_latencies)))
    finally:
        try:
            for sensor, _ in sensors:
                try:
                    sensor.stop()
                finally:
                    sensor.destroy()
        finally:
            world.apply_settings(original_settings)


if __name__ == '__main__':
    main()
