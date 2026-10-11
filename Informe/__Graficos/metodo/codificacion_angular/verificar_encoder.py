#!/usr/bin/env python3
"""Verifica límites en el nodo C++ y regenera la figura con su salida actual.

Usar un ROS_DOMAIN_ID aislado y un ejecutable recién compilado. Lee el bag
original; no modifica sus mensajes ni utiliza su DepthGrid como salida vigente.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import rclpy
from rclpy.qos import QoSProfile, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Header
from custom_interfaces.msg import DepthGrid

import generar_codificacion_angular as figure


def spin_until(node, predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=.05)
    if not predicate():
        raise RuntimeError('No se recibió la respuesta del encoder dentro del plazo.')


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=Path(__file__).parent/'rosbag')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'salida/codificacion_angular.png')
    args = parser.parse_args()
    # Reuse the figure reader's arguments without changing its command line.
    sample = argparse.Namespace(bag=args.source, storage_id='sqlite3', cloud_index=30,
        cloud_topic='/depth_obstacle_filter/obstacle_cloud', grid_topic='/perception/depth_grid',
        depth_topic='/camera/camera/depth/image_rect_raw',
        aperture_topic='/perception/depth_grid/aperture_state_deg',
        height_topic='/depth_obstacle_filter/camera_height', z_min=.25, z_max=5,
        y_min=-.20, y_default_max=2.20, y_margin=.10, sensor_aperture_deg=55,
        resolution_deg=1, max_points=25000, output=args.output)
    points, original_grid, depth, aperture, height = figure.read_sample(sample)
    os.environ.setdefault('RMW_IMPLEMENTATION', 'rmw_cyclonedds_cpp')
    rclpy.init()
    node = rclpy.create_node('angular_boundary_verification')
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    cloud_pub = node.create_publisher(PointCloud2, sample.cloud_topic, 10)
    height_pub = node.create_publisher(Float32, sample.height_topic, latched)
    command_pub = node.create_publisher(Float32, '/perception/depth_grid/aperture_deg', 10)
    grids, states = {}, []
    node.create_subscription(DepthGrid, sample.grid_topic,
        lambda m: grids.update({figure.stamp_ns(m): m}), 10)
    node.create_subscription(Float32, sample.aperture_topic,
        lambda m: states.append(float(m.data)), latched)
    log_path = Path('/tmp/impl-aperture-review/encoder-verification.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open('w')
    process = subprocess.Popen([str(args.encoder), '--ros-args', '--params-file', str(args.config),
        '-p', 'z_min_m:=0.25', '-p', 'z_max_m:=5.0',
        '-p', 'hold_last_nonempty_grid:=false', '-p', 'perf_log_enabled:=false'],
        stdout=log, stderr=subprocess.STDOUT)
    checks = []
    try:
        spin_until(node, lambda: cloud_pub.get_subscription_count() > 0 and bool(states))
        height_pub.publish(Float32(data=float(height)))
        # Wait for the transient height to reach the encoder.
        end = time.monotonic()+.3
        while time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=.05)

        def publish(points_array, stamp):
            header = Header(frame_id='gravity_aligned_frame')
            header.stamp.sec = stamp
            msg = point_cloud2.create_cloud_xyz32(header, points_array.astype(np.float32))
            key = stamp*1_000_000_000
            deadline, next_send = time.monotonic()+8., 0.
            while key not in grids and time.monotonic() < deadline:
                if time.monotonic() >= next_send:
                    cloud_pub.publish(msg)
                    next_send = time.monotonic()+.5
                rclpy.spin_once(node, timeout_sec=.05)
            if key not in grids:
                raise RuntimeError('El encoder no recibió la nube de prueba.')
            return grids[key]

        for index, (requested, effective, angles, expected) in enumerate([
            (45., 45., [-54., -45.1, -44.8, 0., 44.8, 45.1, 54.], 3),
            (70., 55., [-66., -55.1, -54.8, 0., 54.8, 55.1, 66.], 3),
            (5., 15., [-18., -15.1, -14.8, 0., 14.8, 15.1, 18.], 3),
        ]):
            command_pub.publish(Float32(data=requested))
            spin_until(node, lambda: states[-1] == effective)
            synthetic = np.array([[np.tan(np.deg2rad(a)), .2, 1.] for a in angles])
            grid = publish(synthetic, index+10)
            count = sum(cell.count for cell in grid.cells)
            assert count == expected, (requested, count, expected)
            assert grid.rows == 5 and grid.cols == 10
            checks.append({'requested_half_angle_deg': requested,
                'effective_half_angle_deg': effective, 'angles_deg': angles,
                'accepted_count': count, 'expected_count': expected})

        command_pub.publish(Float32(data=45.))
        spin_until(node, lambda: states[-1] == 45.)
        actual_grid = publish(points, 30)
        result = figure.reproduce(points, actual_grid, sample, 45., float(height)+.10)
        assert result['verified'], 'La reconstrucción no coincide con el nodo C++.'
        figure.plot(points, actual_grid, depth, result, sample, 45., float(height)+.10)
        record = {'source_bag': str(args.source.resolve()), 'cloud_index': 30,
            'encoder_sha256': hashlib.sha256(args.encoder.read_bytes()).hexdigest(),
            'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
            'boundary_checks': checks, 'current_half_angle_deg': 45.,
            'max_half_angle_deg': 55., 'internal_bins': result['high_cols'],
            'verified_against_cpp': result['verified'],
            'accepted_points': int(result['count'].sum()),
            'original_accepted_points': sum(c.count for c in original_grid.cells),
            'count': result['count'].tolist()}
        args.output.with_suffix('.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
