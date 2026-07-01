"""HFI-A9 IMU 단독 launch."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    port = LaunchConfiguration('port')
    baud = LaunchConfiguration('baudrate')
    frame_id = LaunchConfiguration('frame_id')

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyUSB_IMU'),
        DeclareLaunchArgument('baudrate', default_value='921600'),
        DeclareLaunchArgument('frame_id', default_value='imu_link'),

        Node(
            package='imu_stack',
            executable='hfi_a9_node',
            name='hfi_a9_node',
            output='screen',
            parameters=[{
                'port': port,
                'baudrate': baud,
                'frame_id': frame_id,
            }],
        ),
    ])
