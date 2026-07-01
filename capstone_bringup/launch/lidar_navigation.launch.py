"""LiDAR + IMU + (planner-only Nav2) + segment 기반 자율 주행 launch.

설계:
  bt_navigator / controller_server / behavior_server 를 쓰지 않는다.
  Nav2 는 planner_server (+ global_costmap) 만 띄워 경로 생성기로만 사용하고,
  주행은 segment 파이프라인이 담당한다.

  /goal_pose → goal_to_plan (planner 직접 호출) → /plan
            → path_segmenter → /motion_segments
            → segment_executor → /cmd_vel_raw
            → motion_sequencer → /cmd_vel
            → mecanum_bridge → /wheel_targets → can_bridge → CAN

포함:
  - control_stack: mecanum_bridge, can_bridge, goal_to_plan, path_segmenter,
                   segment_executor, motion_sequencer
  - imu_stack: hfi_a9
  - robot_localization: ekf_node (TF odom→base_link)
  - lidar_stack: localization (rplidar + RSP + map_server + amcl)  ← map→odom
  - nav2_planner: planner_server + 전용 lifecycle_manager
  - rviz2

사용:
  - RViz "2D Pose Estimate" 로 초기 위치 → AMCL 이 map→odom 발행
  - RViz "2D Goal Pose" 로 목표 → goal_to_plan 이 planner 호출 → segment 실행
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
    nav2_params = LaunchConfiguration('nav2_params')
    map_yaml = LaunchConfiguration('map')
    rviz_config = LaunchConfiguration('rviz_config')
    imu_port = LaunchConfiguration('imu_port')

    mecanum_node = Node(
        package='control_stack', executable='mecanum_bridge_node',
        name='mecanum_bridge_node', output='screen', parameters=[control_params],
    )
    can_node = Node(
        package='control_stack', executable='can_bridge_node',
        name='can_bridge_node', output='screen', parameters=[control_params],
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

    # Nav2 planner 만 (경로 생성기). bt_navigator/controller 불필요.
    planner_server = Node(
        package='nav2_planner', executable='planner_server',
        name='planner_server', output='screen', parameters=[nav2_params],
    )
    planner_lifecycle = Node(
        package='nav2_lifecycle_manager', executable='lifecycle_manager',
        name='lifecycle_manager_planner', output='screen',
        parameters=[{
            'use_sim_time': False,
            'autostart': True,
            'node_names': ['planner_server'],
        }],
    )

    goal_to_plan = Node(
        package='control_stack', executable='goal_to_plan_node',
        name='goal_to_plan_node', output='screen', parameters=[control_params],
    )
    path_segmenter = Node(
        package='control_stack', executable='path_segmenter_node',
        name='path_segmenter_node', output='screen', parameters=[control_params],
    )
    segment_executor = Node(
        package='control_stack', executable='segment_executor_node',
        name='segment_executor_node', output='screen', parameters=[control_params],
    )
    motion_sequencer = Node(
        package='control_stack', executable='motion_sequencer_node',
        name='motion_sequencer_node', output='screen', parameters=[control_params],
    )

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
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
            'nav2_params',
            default_value=os.path.expanduser('~/capstone_ws/config/nav2_params.yaml'),
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
        mecanum_node,
        can_node,
        imu_launch,
        ekf_node,
        localization_launch,
        planner_server,
        planner_lifecycle,
        goal_to_plan,
        path_segmenter,
        segment_executor,
        motion_sequencer,
        rviz,
    ])
