import glob
import os

from setuptools import find_packages, setup

package_name = 'camera_stack'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'models'),
            glob.glob('models/*.blob') + glob.glob('models/*.pt')),
        (os.path.join('share', package_name, 'launch'),
            glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='김윤기',
    maintainer_email='kyg100800@gmail.com',
    description='Camera stack: lane / stop_line / traffic_light detection',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'lane_detector_node = camera_stack.lane_detector_node:main',
            'lane_detector_bev_node = camera_stack.lane_detector_bev_node:main',
            'lane_recovery_node = camera_stack.lane_recovery_node:main',
            'traffic_light_detector_node = camera_stack.traffic_light_detector_node:main',
            'debug_view_logger_node = camera_stack.debug_view_logger_node:main',
        ],
    },
)
