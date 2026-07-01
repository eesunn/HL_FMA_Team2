"""SLAM Toolbox 매핑 launch.

포함:
  - rplidar_a1
  - robot_state_publisher (URDF)
  - slam_toolbox online_async
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
    slam_params = LaunchConfiguration('slam_params')

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

    # 복도 degeneracy + loop closure 대응 파라미터는 config/slam_params.yaml 에.
    slam = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params, {'use_sim_time': use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument(
            'slam_params',
            default_value=os.path.expanduser('~/capstone_ws/config/slam_params.yaml'),
        ),
        rplidar_launch,
        rsp,
        slam,
    ])
