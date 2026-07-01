"""카메라 단독 주행 launch 파일.

twist_mux / Nav2 없이 카메라만으로 차선 추종 주행 + CSV 자동 로깅.

사전 준비 (CAN 인터페이스 — 재부팅 시마다 1회 실행):
    sudo ip link set can0 type can bitrate 500000
    sudo ip link set can0 up

실행:
    ros2 launch camera_stack camera_drive.launch.py

옵션:
    ros2 launch camera_stack camera_drive.launch.py show_window:=true
    ros2 launch camera_stack camera_drive.launch.py base_speed:=0.08 k_steer:=0.4
    ros2 launch camera_stack camera_drive.launch.py log_dir:=/tmp/my_logs

노드 흐름:
    OAK-D
      └─ lane_detector_node  → /lane_offset
                                └─ lane_recovery_node  → /cmd_vel
                                                          └─ mecanum_bridge_node  → /wheel_targets
                                                                                    └─ can_bridge_node  → CAN 0x300 → TC275
    (can_bridge 시작 시 0x301 ControlMode=3 자동 송신)

긴급 정지: launch 터미널에서 SPACE 키
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    log_dir = os.path.expanduser('~/capstone_ws/logs')
    config_dir = os.path.expanduser('~/capstone_ws/config')
    control_params = os.path.join(config_dir, 'control_params.yaml')

    return LaunchDescription([

        # ── launch 인자 ────────────────────────────────────────────────────
        DeclareLaunchArgument('show_window', default_value='false',
                              description='디버그 윈도우 표시 (Jetson=false)'),
        DeclareLaunchArgument('base_speed', default_value='0.10',
                              description='기본 전진 속도 (m/s)'),
        DeclareLaunchArgument('k_steer', default_value='0.5',
                              description='조향 P 게인'),
        DeclareLaunchArgument('log_dir', default_value=log_dir,
                              description='CSV 저장 폴더'),

        LogInfo(msg='[camera_drive] ===== 카메라 단독 주행 시작 ====='),
        LogInfo(msg='[camera_drive] CSV 자동 로깅: ~/capstone_ws/logs/'),
        LogInfo(msg='[camera_drive] 긴급 정지: 이 터미널에서 SPACE 키'),

        # ── 1. 차선 인식 (OAK-D → /lane_offset) ──────────────────────────
        # log_diag=true 기본값 → lane_diag_*.csv 자동 저장
        Node(
            package='camera_stack',
            executable='lane_detector_node',
            name='lane_detector_node',
            output='screen',
            parameters=[{
                'show_window': LaunchConfiguration('show_window'),
                'log_diag':    True,
                'capture_dir': LaunchConfiguration('log_dir'),
            }],
        ),

        # ── 2. 차선 추종 제어 (/lane_offset → /cmd_vel) ───────────────────
        # auto_enable=True  → 시작 즉시 차선 추종 활성화
        # output_topic=/cmd_vel → twist_mux 없이 mecanum_bridge 로 직접 전달
        # log_csv=true 기본값 → lane_recovery_*.csv 자동 저장
        Node(
            package='camera_stack',
            executable='lane_recovery_node',
            name='lane_recovery_node',
            output='screen',
            parameters=[{
                'auto_enable':   True,
                'output_topic':  '/cmd_vel',
                'base_speed':    LaunchConfiguration('base_speed'),
                'k_steer':       LaunchConfiguration('k_steer'),
                'log_csv':       True,
                'capture_dir':   LaunchConfiguration('log_dir'),
                'keyboard_stop': True,
            }],
        ),

        # ── 3. 메카넘 역운동학 (/cmd_vel → /wheel_targets [A,B,C,D km/h]) ─
        # use_strafe=False: 차동구동 전용 (linear.y 무시, 카메라는 wz만 사용)
        Node(
            package='control_stack',
            executable='mecanum_bridge_node',
            name='mecanum_bridge_node',
            output='screen',
            parameters=[control_params],
        ),

        # ── 4. CAN 송신 (/wheel_targets → CAN 0x300 + 0x301) ─────────────
        # 시작 시 0x301 ControlMode=3 (ROS2 Autonomous) 자동 송신
        # cmd_send_rate_hz=20Hz 로 0x300 SpeedCommand 주기 송신
        Node(
            package='control_stack',
            executable='can_bridge_node',
            name='can_bridge_node',
            output='screen',
            parameters=[control_params],
        ),

    ])
