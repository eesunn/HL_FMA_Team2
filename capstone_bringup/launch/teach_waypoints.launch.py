"""Teach-and-repeat waypoint 기록 launch.

차량을 TC275 PS2 조이스틱(ControlMode 2)으로 직접 운전하면서, AMCL 위치추정으로
얻은 실제 차량 자세(TF map->base_link)를 일정 거리마다 자동으로 waypoint 로 기록.
RViz 클릭이 아니라 차량을 실제로 그 지점에 몰고 가서 찍으므로 좌표가 헷갈리지 않음.

포함:
  - lidar_stack: localization (rplidar + RSP + map_server + amcl)  ← map->odom
  - imu_stack: hfi_a9
  - robot_localization: ekf_node                                   ← odom->base_link
  - control_stack: can_bridge (initial_ctrl_mode=2 → TC275 PS2 모드, /odom 발행)
  - control_stack: teach_waypoint_recorder                         ← TF 기록
  - rviz2 (waypoint 마커 + 위치 확인)

주행 제어(Nav2 / mecanum_bridge / motion_sequencer)는 띄우지 않는다. 모터는 PS2 가
직접 구동하고 ROS2 는 위치추정 + 기록만 한다.

사용:
  1) ros2 launch capstone_bringup teach_waypoints.launch.py
  2) RViz "2D Pose Estimate" 로 AMCL 초기 위치 수렴
  3) PS2 조이스틱으로 원하는 경로(S자 등)를 천천히 운전
  4) 일정 거리마다 자동 기록됨 (콘솔/파일에 waypoints:="..." 출력)
  5) Ctrl-C 종료 시 최종 목록 + 저장 경로 출력
  6) 출력된 명령으로 continuous launch + s_waypoint_runner 실행해 재생
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    lidar_share = get_package_share_directory('lidar_stack')
    bringup_share = get_package_share_directory('capstone_bringup')
    imu_share = get_package_share_directory('imu_stack')

    control_params = LaunchConfiguration('control_params')
    ekf_params = LaunchConfiguration('ekf_params')
    map_yaml = LaunchConfiguration('map')
    rviz_config = LaunchConfiguration('rviz_config')
    imu_port = LaunchConfiguration('imu_port')
    record_distance = LaunchConfiguration('record_distance_m')
    record_heading = LaunchConfiguration('record_heading_deg')
    max_points = LaunchConfiguration('max_points')

    # can_bridge: PS2 모드(2)로 시작 → TC275 가 0x300 무시하고 PS2 로 구동.
    # /odom 은 0x200/0x201 피드백으로 계속 발행되어 위치추정에 사용됨.
    # use_strafe 등 운동학 파라미터는 재생 모드와 동일하게 continuous 설정을 사용.
    can_node = Node(
        package='control_stack', executable='can_bridge_node',
        name='can_bridge_node', output='screen',
        parameters=[control_params, {'initial_ctrl_mode': 2}],
    )

    imu_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(imu_share, 'launch', 'imu.launch.py')
        ),
        launch_arguments={'port': imu_port}.items(),
    )
    ekf_node = Node(
        package='robot_localization', executable='ekf_node',
        name='ekf_filter_node', output='screen', parameters=[ekf_params],
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(lidar_share, 'launch', 'localization.launch.py')
        ),
        launch_arguments={'map': map_yaml}.items(),
    )

    teach_recorder = Node(
        package='control_stack', executable='teach_waypoint_recorder',
        name='teach_waypoint_recorder', output='screen',
        parameters=[{
            'record_distance_m': ParameterValue(record_distance, value_type=float),
            'record_heading_deg': ParameterValue(record_heading, value_type=float),
            'max_points': ParameterValue(max_points, value_type=int),
        }],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        arguments=['-d', rviz_config],
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'control_params',
            default_value=os.path.expanduser(
                '~/capstone_ws/config/control_params_continuous.yaml'
            ),
        ),
        DeclareLaunchArgument(
            'ekf_params',
            default_value=os.path.expanduser('~/capstone_ws/config/ekf_params.yaml'),
        ),
        DeclareLaunchArgument(
            'map',
            default_value=os.path.expanduser('~/capstone_ws/maps/course_map.yaml'),
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=os.path.join(bringup_share, 'rviz', 'navigation.rviz'),
        ),
        DeclareLaunchArgument('imu_port', default_value='/dev/ttyUSB_IMU'),
        DeclareLaunchArgument('record_distance_m', default_value='0.20'),
        DeclareLaunchArgument('record_heading_deg', default_value='25.0'),
        DeclareLaunchArgument('max_points', default_value='0'),
        can_node,
        imu_launch,
        ekf_node,
        localization_launch,
        teach_recorder,
        rviz,
    ])
