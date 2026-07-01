"""LiDAR SLAM 매핑 통합 launch (IMU + EKF 융합 포함).

포함:
  - control_stack: mecanum_bridge, can_bridge (odom 토픽만, TF 미발행)
  - imu_stack: hfi_a9 IMU 드라이버
  - robot_localization: ekf_node (odom + imu → /odometry/filtered, TF odom→base_link)
  - lidar_stack: slam_mapping (rplidar + robot_state_publisher + slam_toolbox)
  - rviz2

사용 시나리오:
  - 사용자가 차량을 수동(키보드 teleop 또는 PS2)으로 운전하면서 맵을 작성
  - 매핑 시에는 segment 노드들 미실행 (수동 운전)
  - 별도 터미널에서 `ros2 run nav2_map_server map_saver_cli -f maps/course_map` 로 저장
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    lidar_share = get_package_share_directory('lidar_stack')
    bringup_share = get_package_share_directory('capstone_bringup')
    imu_share = get_package_share_directory('imu_stack')

    control_params = LaunchConfiguration('control_params')
    ekf_params = LaunchConfiguration('ekf_params')
    rviz_config = LaunchConfiguration('rviz_config')
    imu_port = LaunchConfiguration('imu_port')

    mecanum_node = Node(
        package='control_stack',
        executable='mecanum_bridge_node',
        name='mecanum_bridge_node',
        output='screen',
        parameters=[control_params],
    )

    can_node = Node(
        package='control_stack',
        executable='can_bridge_node',
        name='can_bridge_node',
        output='screen',
        parameters=[control_params],
    )

    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(imu_share, 'launch', 'imu.launch.py')
        ),
        launch_arguments={'port': imu_port}.items(),
    )

    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_params],
    )

    slam_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lidar_share, 'launch', 'slam_mapping.launch.py')
        )
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'control_params',
            default_value=os.path.expanduser('~/capstone_ws/config/control_params.yaml'),
        ),
        DeclareLaunchArgument(
            'ekf_params',
            default_value=os.path.expanduser('~/capstone_ws/config/ekf_params.yaml'),
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(bringup_share, 'rviz', 'mapping.rviz'),
        ),
        DeclareLaunchArgument('imu_port', default_value='/dev/ttyUSB_IMU'),
        mecanum_node,
        can_node,
        imu_launch,
        ekf_node,
        slam_launch,
        rviz,
    ])
