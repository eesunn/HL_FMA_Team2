"""AMCL 로컬라이제이션 launch.

사전 작성된 맵(maps/course_map.yaml)을 로드하고 LiDAR + 휠 오도메트리로 AMCL.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lidar_share = get_package_share_directory('lidar_stack')
    urdf_xacro = os.path.join(lidar_share, 'urdf', 'robot.urdf.xacro')

    use_sim_time = LaunchConfiguration('use_sim_time')
    map_yaml = LaunchConfiguration('map')

    rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lidar_share, 'launch', 'rplidar_a1.launch.py')
        )
    )

    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', urdf_xacro]), value_type=str
        )
    }

    rsp = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
    )

    map_server = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': map_yaml, 'use_sim_time': use_sim_time}],
    )

    amcl = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'global_frame_id': 'map',
            'scan_topic': '/scan',
            'min_particles': 500,
            'max_particles': 2000,
            'laser_model_type': 'likelihood_field',
            'laser_max_range': 12.0,
            'laser_min_range': 0.15,
            'transform_tolerance': 0.5,
            'robot_model_type': 'nav2_amcl::OmniMotionModel',
        }],
    )

    lifecycle_mgr = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'autostart': True,
            'node_names': ['map_server', 'amcl'],
        }],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'map',
            default_value=os.path.expanduser('~/capstone_ws/maps/course_map.yaml'),
        ),
        rplidar_launch,
        rsp,
        map_server,
        amcl,
        lifecycle_mgr,
    ])
