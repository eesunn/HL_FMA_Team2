"""RPLidar A1 단일 launch.

토픽: /scan (sensor_msgs/LaserScan)
frame_id: laser
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    serial_port = LaunchConfiguration('serial_port')
    frame_id = LaunchConfiguration('frame_id')
    scan_mode = LaunchConfiguration('scan_mode')

    return LaunchDescription([
        DeclareLaunchArgument('serial_port', default_value='/dev/ttyUSB_LIDAR'),
        DeclareLaunchArgument('frame_id', default_value='laser'),
        DeclareLaunchArgument('scan_mode', default_value='Standard'),

        Node(
            package='rplidar_ros',
            executable='rplidar_composition',
            name='rplidar_node',
            output='screen',
            parameters=[{
                'serial_port': serial_port,
                'serial_baudrate': 115200,
                'frame_id': frame_id,
                'inverted': False,
                'angle_compensate': True,
                'scan_mode': scan_mode,
            }],
        ),
    ])
