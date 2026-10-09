#!/usr/bin/env python3
"""Record real v2 outputs for the R03 method figure, without changing thresholds."""
import os
from pathlib import Path
import signal
import subprocess
import time
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
os.environ['ROS_DOMAIN_ID'] = '84'
os.environ['ROS_LOCALHOST_ONLY'] = '1'
logs = args.output.parent/(args.output.name+'_logs')
logs.mkdir(exist_ok=False)
children, handles = [], []
def start(name, command):
    handle = (logs/(name+'.log')).open('w')
    handles.append(handle)
    child = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
    children.append(child)
    return child
try:
    start('pipeline', ['ros2','launch',str(Path(__file__).with_name('trace.launch.py'))])
    time.sleep(9)
    interface = subprocess.check_output(['ros2','param','get','/depth_obstacle_filter',
        'publish_local_mapper_interface'], text=True, timeout=15)
    if 'True' not in interface:
        raise RuntimeError('Mapper evidence interface disabled: '+interface)
    recorder = start('recorder', ['ros2','bag','record','-o',str(args.output),
        '/local_mapper/occupancy_grid','/nav_odom','/odom_info',
        '/perception/depth_grid/aperture_state_deg','/spatial_awareness/collision_risk',
        '/spatial_awareness/debug/markers','/perception/haptic_grid','/perception/depth_grid','/clock',
        '/tf','/depth_obstacle_filter/camera_height','/rosout','/depth_obstacle_filter/obstacle_cloud','/depth_obstacle_filter/ground_evidence',
        '/depth_obstacle_filter/debug/depth_cloud','/depth_obstacle_filter/debug/ground_cloud',
        '/depth_obstacle_filter/debug/ceiling_cloud'])
    time.sleep(3)
    player = start('player', ['ros2','bag','play',str(args.source),'--clock','--rate','1.0','--topics',
        '/camera/camera/color/image_raw','/camera/camera/color/camera_info',
        '/camera/camera/aligned_depth_to_color/image_raw', '/camera/camera/aligned_depth_to_color/camera_info',
        '/camera/camera/depth/image_rect_raw','/camera/camera/depth/camera_info',
        '/camera/camera/imu','/camera/camera/gyro/sample','/camera/camera/accel/sample','/tf_static'])
    player.wait(timeout=240)
    time.sleep(2)
    if recorder.poll() is not None:
        raise RuntimeError('Recorder stopped early; see recorder.log')
finally:
    for child in reversed(children):
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGINT)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL)
                child.wait()
    for handle in handles:
        handle.close()
print(args.output)
