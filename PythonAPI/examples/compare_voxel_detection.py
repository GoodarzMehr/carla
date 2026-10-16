#!/usr/bin/env python3

# Copyright (c) 2026 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Benchmark voxel detection capture latency.

Run against a server and Python API built from this checkout, with no other
client advancing the world. The current map and actors are retained.

    python3 compare_voxel_detection.py --yaw 37 --pitch 10
    python3 compare_voxel_detection.py --attribute use_complex_collision=false
    python3 compare_voxel_detection.py --preset simbev --port 29354
    python3 compare_voxel_detection.py --preset simbev --compare-triangle-reuse

Reported latency includes world.tick(), sensor processing and delivery.
Python payload inspection is excluded. Use the same static scene for timing.
Use --attribute show_calculation_time=true for sensor timings in the server log.
The log separates frame wall phases from sampled worker timings for gathering,
culling, triangle preparation, fine voxel tests, and priority ties. Worker timings
are not frame latency.
Triangle reuse defaults to false; use --attribute use_triangle_reuse=true to
benchmark the experiment. --compare-triangle-reuse enables the experiment and
checks full voxel bytes against a simultaneous disabled sensor on matching
frames. Its timings include both sensors, so use the default single-sensor
benchmark for performance measurements.
"""

import argparse
import json
import math
import statistics
import time

from queue import Empty, Queue


def receive_frame(queue, frame, timeout):
    '''Wait for the requested sensor frame, discarding older frames.'''
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
    parser.add_argument('--compare-triangle-reuse', action='store_true',
                        help='enable triangle reuse and compare against disabled on matching frames')
    parser.add_argument('--preset', choices=('default', 'simbev'), default='default',
                        help='simbev: range 64 m, voxels 0.1 m, height -1..7 m, chunks 4x4x8')
    parser.add_argument('--json', action='store_true',
                        help='print a JSON result on the final line for sweep_voxel_chunks.py')
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
    if args.frames < 1 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('frames and timeout must be positive; warmup must be nonnegative')
    attributes = {'draw_debug': 'false', 'show_calculation_time': 'false'}
    if args.preset == 'simbev':
        attributes.update({'range': '64', 'voxel_size': '0.1',
                           'lower_limit': '-1', 'upper_limit': '7',
                           'chunk_x': '4', 'chunk_y': '4', 'chunk_z': '8'})
    for item in args.attribute:
        name, separator, value = item.partition('=')
        if not separator or not name or not value:
            parser.error('--attribute requires NAME=VALUE')
        if name == 'sensor_tick':
            parser.error('sensor_tick is controlled by this benchmark')
        attributes[name] = value
    if args.compare_triangle_reuse:
        attributes['use_triangle_reuse'] = 'true'

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    blueprint = world.get_blueprint_library().find('sensor.other.voxel_detection')
    for name, value in attributes.items():
        blueprint.set_attribute(name, value)
    names = set(attributes) | {'range', 'voxel_size', 'lower_limit', 'upper_limit',
                               'chunk_x', 'chunk_y', 'chunk_z', 'use_triangle_reuse',
                               'use_complex_collision', 'ignore_self'}
    converters = {
        carla.ActorAttributeType.Bool: 'as_bool',
        carla.ActorAttributeType.Int: 'as_int',
        carla.ActorAttributeType.Float: 'as_float',
        carla.ActorAttributeType.String: 'as_str'
    }
    values = {}
    for name in sorted(names):
        attribute = blueprint.get_attribute(name)
        values[name] = getattr(attribute, converters[attribute.type])()
    chunk_names = ('chunk_x', 'chunk_y', 'chunk_z')
    chunks = [values[name] for name in chunk_names]
    triangle_reuse = values['use_triangle_reuse']
    dimensions = [values[name] for name in ('range', 'voxel_size', 'lower_limit', 'upper_limit')]
    if not all(math.isfinite(value) for value in dimensions):
        parser.error('voxel dimensions must be finite')
    if values['range'] <= 0 or values['voxel_size'] <= 0 or min(chunks) < 1:
        parser.error('range, voxel-size, and chunks must be positive')
    if values['upper_limit'] <= values['lower_limit']:
        parser.error('upper_limit must exceed lower_limit')
    width = math.ceil(2 * values['range'] / values['voxel_size'])
    height = math.ceil((values['upper_limit'] - values['lower_limit']) / values['voxel_size'])
    if width * width * height > 2147483647:
        parser.error('voxel grid exceeds the sensor\'s 32-bit index capacity')

    print('preset=%s host=%s port=%d' % (args.preset, args.host, args.port))
    print('Voxel attributes: ' + ', '.join('%s=%s' % (name, value)
                                          for name, value in values.items()))
    existing = [sensor.id for sensor in world.get_actors().filter('sensor.other.voxel_detection')]
    print('Existing voxel sensors included in world tick: %s' %
          (', '.join(map(str, existing)) or 'none'))
    current_map = world.get_map()
    spawn_points = current_map.get_spawn_points()
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
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        if blueprint.has_attribute('sensor_tick'):
            blueprint.set_attribute('sensor_tick', '0.0')
        reuse_settings = (False, True) if args.compare_triangle_reuse else (triangle_reuse,)
        for enabled in reuse_settings:
            blueprint.set_attribute('use_triangle_reuse', str(enabled).lower())
            sensor = world.spawn_actor(blueprint, transform)
            queue = Queue()
            sensors.append((sensor, queue))
            sensor.listen(queue.put)
            print('Voxel sensor id=%d use_triangle_reuse=%s' % (sensor.id, str(enabled).lower()))

        for index in range(args.warmup + args.frames):
            started = time.perf_counter()
            frame = world.tick(args.timeout)
            tick_elapsed = time.perf_counter() - started
            data = [receive_frame(queue, frame, args.timeout) for _, queue in sensors]
            elapsed = time.perf_counter() - started
            payloads = [bytes(sample.raw_data) for sample in data]
            if args.compare_triangle_reuse and payloads[0] != payloads[1]:
                raise RuntimeError('Frame %d: triangle reuse disabled/enabled voxel bytes differ '
                                   '(%d/%d bytes)' %
                                   (frame, len(payloads[0]), len(payloads[1])))
            payload = payloads[-1]
            if not payload:
                raise RuntimeError('Frame %d: empty voxel payload' % frame)
            if index >= args.warmup:
                latencies.append(elapsed)
                tick_latencies.append(tick_elapsed)
                voxel_count = len(payload)
                occupied += voxel_count - payload.count(0)
            # Release event storage and byte copies before receiving another large frame.
            del data, payloads, payload

        print('%d voxels/frame, mean occupied voxels %.1f' %
              (voxel_count, occupied / args.frames))
        if args.compare_triangle_reuse:
            print('PASS: identical bytes on %d matching frames (triangle reuse disabled/enabled)' %
                  (args.warmup + args.frames))
        capture_label = ('Comparison capture latency (two sensors; not isolated)'
                         if args.compare_triangle_reuse else 'Capture latency')
        print('%s: median %.2f ms, min %.2f ms, max %.2f ms' %
              (capture_label, 1000 * statistics.median(latencies), 1000 * min(latencies),
               1000 * max(latencies)))
        print('world.tick() median: %.2f ms (world work; not isolated sensor time)' %
              (1000 * statistics.median(tick_latencies)))
        result = {
            'preset': args.preset,
            'map': current_map.name,
            'transform': str(transform),
            'attributes': values,
            'existing_voxel_sensors': existing,
            'chunks': chunks,
            'frames': args.frames,
            'warmup': args.warmup,
            'voxel_count': voxel_count,
            'occupied_voxels': occupied / args.frames,
            'capture_median_ms': 1000 * statistics.median(latencies),
            'capture_min_ms': 1000 * min(latencies),
            'capture_max_ms': 1000 * max(latencies),
            'tick_median_ms': 1000 * statistics.median(tick_latencies)
        }
        if args.compare_triangle_reuse:
            result['triangle_reuse_comparison'] = {
                'baseline_enabled': False,
                'target_enabled': True,
                'matched_frames': args.warmup + args.frames,
                'identical_bytes': True
            }
    finally:
        try:
            for sensor, _ in sensors:
                try:
                    try:
                        sensor.stop()
                    finally:
                        sensor.destroy()
                except RuntimeError as error:
                    print('Could not clean up sensor %d: %s' % (sensor.id, error))
        finally:
            world.apply_settings(original_settings)

    if args.json:
        print(json.dumps(result))


if __name__ == '__main__':
    main()
