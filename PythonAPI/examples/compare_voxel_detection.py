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
    python3 compare_voxel_detection.py --preset simbev --ground-offset -0.537
    python3 compare_voxel_detection.py --preset simbev --save-payloads baseline
    python3 compare_voxel_detection.py --preset simbev --diff-saved baseline

Reported latency includes world.tick(), sensor processing and delivery.
Python payload inspection is excluded. Use the same static scene for timing.
Use --attribute show_calculation_time=true for sensor timings in the server log.
The log separates frame wall phases from worker timings summed over tiles for
gathering, triangle fetches, surface rasterization, PhysX queries, and priority
ties. Worker timings are not frame latency.

The ground is projected below the spawn point and reported relative to the sensor;
a warning is printed when it falls outside the voxel grid, because timings would
then exclude the ground surface. --ground-offset places the sensor so the ground
sits at that height in the sensor frame.

--save-payloads writes every frame, warmup included, and --diff-saved compares a
later run frame by frame. Use the same map, spawn point and attributes for both.
Use --attribute oracle_stride=N to compare every Nth voxel with a per-voxel engine
query inside the server; the results are in the server log.
"""

import argparse
import json
import math
import statistics
import time

from pathlib import Path
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


def diff_payloads(first, second, max_examples=5):
    '''Count differing voxels; equal 1 MiB slices are skipped at memcmp speed.

    Returns the number of differing bytes and up to max_examples
    (flat index, first label, second label) tuples.
    '''
    if len(first) != len(second):
        raise RuntimeError('Payload sizes differ: %d vs %d bytes' % (len(first), len(second)))
    slice_size = 1 << 20
    first_view = memoryview(first)
    second_view = memoryview(second)
    mismatches = 0
    examples = []
    for begin in range(0, len(first), slice_size):
        first_slice = first_view[begin:begin + slice_size]
        second_slice = second_view[begin:begin + slice_size]
        if first_slice == second_slice:
            continue
        for offset, (a, b) in enumerate(zip(first_slice, second_slice)):
            if a != b:
                mismatches += 1
                if len(examples) < max_examples:
                    examples.append((begin + offset, a, b))
    return mismatches, examples


def project_ground(world, location, carla):
    '''Height of the first surface below a point, or None if nothing is hit.'''
    start = carla.Location(location.x, location.y, location.z + 1.0)
    ground = world.ground_projection(start)
    return None if ground is None else ground.location.z


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--frames', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--preset', choices=('default', 'simbev'), default='default',
                        help='simbev: range 64 m, voxels 0.1 m, height -1..7 m, tiles 32x32x128')
    parser.add_argument('--json', action='store_true',
                        help='print a JSON result on the final line for sweep_voxel_tiles.py')
    parser.add_argument('--attribute', action='append', default=[], metavar='NAME=VALUE',
                        help='override a voxel blueprint attribute; repeat as needed')
    parser.add_argument('--spawn-point', type=int, default=0)
    placement = parser.add_mutually_exclusive_group()
    placement.add_argument('--height', type=float, default=2.0,
                           help='height above the selected map spawn point, in meters')
    placement.add_argument('--ground-offset', type=float, metavar='M',
                           help='place the sensor so the ground sits at M meters in the sensor '
                                'frame, e.g. -0.537; the ground is projected below the spawn point')
    parser.add_argument('--save-payloads', type=Path, metavar='DIR',
                        help='write every frame payload, warmup included, for --diff-saved')
    parser.add_argument('--diff-saved', type=Path, metavar='DIR',
                        help='compare every frame payload with one written by --save-payloads')
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
                           'tile_x': '32', 'tile_y': '32', 'tile_z': '128'})
    for item in args.attribute:
        name, separator, value = item.partition('=')
        if not separator or not name or not value:
            parser.error('--attribute requires NAME=VALUE')
        if name == 'sensor_tick':
            parser.error('sensor_tick is controlled by this benchmark')
        attributes[name] = value

    import carla

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    blueprint = world.get_blueprint_library().find('sensor.other.voxel_detection')
    for name, value in attributes.items():
        blueprint.set_attribute(name, value)
    names = set(attributes) | {'range', 'voxel_size', 'lower_limit', 'upper_limit',
                               'tile_x', 'tile_y', 'tile_z',
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
    tile_names = ('tile_x', 'tile_y', 'tile_z')
    tiles = [values[name] for name in tile_names]
    dimensions = [values[name] for name in ('range', 'voxel_size', 'lower_limit', 'upper_limit')]
    if not all(math.isfinite(value) for value in dimensions):
        parser.error('voxel dimensions must be finite')
    if values['range'] <= 0 or values['voxel_size'] <= 0 or min(tiles) < 1:
        parser.error('range, voxel-size, and tiles must be positive')
    if values['upper_limit'] <= values['lower_limit']:
        parser.error('upper_limit must exceed lower_limit')
    width = math.ceil(2 * values['range'] / values['voxel_size'])
    height = math.ceil((values['upper_limit'] - values['lower_limit']) / values['voxel_size'])
    if width * width * height > 2147483647:
        parser.error('voxel grid exceeds the sensor\'s 32-bit index capacity')
    if args.diff_saved and not args.diff_saved.is_dir():
        parser.error('diff-saved directory %s does not exist' % args.diff_saved)
    if args.save_payloads:
        args.save_payloads.mkdir(parents=True, exist_ok=True)

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
    ground = project_ground(world, transform.location, carla)
    if args.ground_offset is not None:
        if ground is None:
            parser.error('no ground found below spawn point %d' % args.spawn_point)
        transform.location.z = ground - args.ground_offset
    else:
        transform.location.z += args.height
    transform.rotation.yaw += args.yaw
    transform.rotation.pitch += args.pitch
    transform.rotation.roll += args.roll
    print('Sensor transform: %s' % transform)
    ground_offset = None if ground is None else ground - transform.location.z
    if ground_offset is None:
        print('Warning: no ground found below the sensor')
    else:
        print('Ground: %.3f m relative to the sensor' % ground_offset)
        if not values['lower_limit'] <= ground_offset <= values['upper_limit']:
            print('Warning: the ground is outside the voxel grid [%g, %g] m, so timings exclude '
                  'the ground surface; use --ground-offset' %
                  (values['lower_limit'], values['upper_limit']))
    metadata = {'map': current_map.name, 'transform': str(transform), 'attributes': values}
    if args.save_payloads:
        (args.save_payloads / 'meta.json').write_text(json.dumps(metadata, indent=2, sort_keys=True))
    if args.diff_saved:
        saved_metadata_path = args.diff_saved / 'meta.json'
        if saved_metadata_path.is_file():
            saved_metadata = json.loads(saved_metadata_path.read_text())
            for name in sorted(set(saved_metadata) | set(metadata)):
                if saved_metadata.get(name) != metadata.get(name):
                    print('Warning: %s differs from the saved run: %s vs %s' %
                          (name, saved_metadata.get(name), metadata.get(name)))

    original_settings = world.get_settings()
    sensor = None
    latencies = []
    tick_latencies = []
    occupied = 0
    voxel_count = 0
    diff_mismatches = 0
    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        if blueprint.has_attribute('sensor_tick'):
            blueprint.set_attribute('sensor_tick', '0.0')
        sensor = world.spawn_actor(blueprint, transform)
        queue = Queue()
        sensor.listen(queue.put)
        print('Voxel sensor id=%d' % sensor.id)

        for index in range(args.warmup + args.frames):
            started = time.perf_counter()
            frame = world.tick(args.timeout)
            tick_elapsed = time.perf_counter() - started
            data = receive_frame(queue, frame, args.timeout)
            elapsed = time.perf_counter() - started
            payload = bytes(data.raw_data)
            if not payload:
                raise RuntimeError('Frame %d: empty voxel payload' % frame)
            frame_file = 'frame_%03d.bin' % index
            if args.save_payloads:
                (args.save_payloads / frame_file).write_bytes(payload)
            if args.diff_saved:
                saved_path = args.diff_saved / frame_file
                if not saved_path.is_file():
                    raise RuntimeError('No saved payload %s' % saved_path)
                mismatches, examples = diff_payloads(saved_path.read_bytes(), payload)
                diff_mismatches += mismatches
                if mismatches:
                    print('Frame %d: %d voxels differ from %s' % (index, mismatches, saved_path))
                    for flat, saved_label, label in examples:
                        x, rest = divmod(flat, width * height)
                        y, z = divmod(rest, height)
                        print('  voxel (%d, %d, %d): saved %d, now %d' % (x, y, z, saved_label, label))
            if index >= args.warmup:
                latencies.append(elapsed)
                tick_latencies.append(tick_elapsed)
                voxel_count = len(payload)
                occupied += voxel_count - payload.count(0)
            # Release event storage and byte copies before receiving another large frame.
            del data, payload

        print('%d voxels/frame, mean occupied voxels %.1f' %
              (voxel_count, occupied / args.frames))
        print('Capture latency: median %.2f ms, min %.2f ms, max %.2f ms' %
              (1000 * statistics.median(latencies), 1000 * min(latencies),
               1000 * max(latencies)))
        print('world.tick() median: %.2f ms (world work; not isolated sensor time)' %
              (1000 * statistics.median(tick_latencies)))
        if args.diff_saved:
            print('Payload diff against %s: %d mismatched voxels over %d frames' %
                  (args.diff_saved, diff_mismatches, args.warmup + args.frames))
            if diff_mismatches:
                raise RuntimeError('Payloads differ from %s' % args.diff_saved)
        result = {
            'preset': args.preset,
            'map': current_map.name,
            'transform': str(transform),
            'attributes': values,
            'existing_voxel_sensors': existing,
            'tiles': tiles,
            'frames': args.frames,
            'warmup': args.warmup,
            'voxel_count': voxel_count,
            'occupied_voxels': occupied / args.frames,
            'capture_median_ms': 1000 * statistics.median(latencies),
            'capture_min_ms': 1000 * min(latencies),
            'capture_max_ms': 1000 * max(latencies),
            'tick_median_ms': 1000 * statistics.median(tick_latencies),
            'ground_offset': ground_offset,
            'diff_mismatches': diff_mismatches if args.diff_saved else None
        }
    finally:
        try:
            if sensor is not None:
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
