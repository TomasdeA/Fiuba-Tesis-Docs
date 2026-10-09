"""Replay harness: all nodes use the rosbag clock, with production parameters."""
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import SetParameter
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    launch_file = Path(get_package_share_directory('nav_bringup'))/'launch/nav.launch.py'
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(launch_file)), launch_arguments={
            'use_realsense':'false', 'use_bag':'false', 'use_hw':'false',
            'use_perception':'true', 'pipeline_mode':'filtered', 'use_local_mapper':'true',
            'odom_source':'rtabmap_odom', 'use_spatial_awareness':'true',
            'debug':'true', 'use_viz':'false', 'use_rviz':'false', 'performance':'false',
        }.items()),
    ])
