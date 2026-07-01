"""LiDAR + IMU + 전체 Nav2 (continuous control) 자율 주행 launch.

설계:
  full Nav2 bringup (planner_server + controller_server(MPPI Omni) + bt_navigator
  + behavior_server + costmaps) 을 사용. 실시간 장애물 회피 + 메카넘 strafe 활용.
  segment 파이프라인(goal_to_plan/path_segmenter/segment_executor)은 미사용.

토픽 흐름:
  /goal_pose → bt_navigator → planner_server → /plan
                            → controller_server → /cmd_vel  (←Nav2 기본)
                                                  ↓ (이 launch 에서 /cmd_vel_nav 로 remap)
                                          motion_sequencer (input=/cmd_vel_nav)
                                                  ↓ (max 클램프)
                                              /cmd_vel  (← motion_sequencer 가 발행)
                                                  ↓
                                        mecanum_bridge (use_strafe=true, full mecanum 역운동학)
                                                  ↓
                                            /wheel_targets
                                                  ↓
                                          can_bridge (use_strafe=true) → CAN 0x300

운영 비교용 — segment 모드는 lidar_navigation.launch.py 그대로 유지.

사용:
  - RViz "2D Pose Estimate" → AMCL 수렴
  - RViz "2D Goal Pose" → Nav2 가 경로 + 제어 모두 담당. 장애물 등장 시 자동 회피/재계획.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetRemap


def generate_launch_description():
    lidar_share = get_package_share_directory('lidar_stack')
    bringup_share = get_package_share_directory('capstone_bringup')
    imu_share = get_package_share_directory('imu_stack')
    nav2_bringup_share = get_package_share_directory('nav2_bringup')
    camera_share = get_package_share_directory('camera_stack')

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

    # Nav2 full bringup — controller_server 가 발행하는 /cmd_vel 을 /cmd_vel_nav 로
    # remap 해 motion_sequencer 에 입력. GroupAction + SetRemap 으로 include 안의
    # 모든 노드에 적용된다.
    nav2_group = GroupAction(actions=[
        SetRemap(src='/cmd_vel', dst='/cmd_vel_nav'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_share, 'launch', 'navigation_launch.py')
            ),
            launch_arguments={
                'use_sim_time': 'false',
                'params_file': nav2_params,
                'autostart': 'true',
            }.items(),
        ),
    ])

    motion_sequencer = Node(
        package='control_stack', executable='motion_sequencer_node',
        name='motion_sequencer_node', output='screen', parameters=[control_params],
    )

    traffic_light_detector = Node(
        package='camera_stack', executable='traffic_light_detector_node',
        name='traffic_light_detector_node', output='screen',
        parameters=[{
            'model_path': os.path.join(camera_share, 'models', 'traffic_light_best.pt'),
            'conf_thresh':        0.50,
            'vote_buffer_size':   10,
            'min_vote_samples':   3,
            'red_vote_ratio':     0.5,
            'green_vote_ratio':   0.5,
            'input_topic':        '/camera/image_raw',
            'show_window':        False,
            'log_csv':            True,
        }],
    )

    nav_logger = Node(
        package='control_stack', executable='navigation_logger_node',
        name='navigation_logger_node', output='screen',
        parameters=[control_params, {'map_yaml_path': map_yaml}],
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
        nav2_group,
        motion_sequencer,
        traffic_light_detector,
        nav_logger,
        rviz,
    ])
