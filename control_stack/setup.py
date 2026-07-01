from setuptools import find_packages, setup

package_name = 'control_stack'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='김윤기',
    maintainer_email='kyg100800@gmail.com',
    description='Differential-drive kinematics + CAN bridge + segment-based motion executor.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mecanum_bridge_node = control_stack.mecanum_bridge_node:main',
            'can_bridge_node = control_stack.can_bridge_node:main',
            'path_segmenter_node = control_stack.path_segmenter_node:main',
            'segment_executor_node = control_stack.segment_executor_node:main',
            'motion_sequencer_node = control_stack.motion_sequencer_node:main',
            'goal_to_plan_node = control_stack.goal_to_plan_node:main',
            'navigation_logger_node = control_stack.navigation_logger_node:main',
            's_waypoint_runner = control_stack.s_waypoint_runner_node:main',
            'clicked_waypoint_recorder = control_stack.clicked_waypoint_recorder_node:main',
            'pose_waypoint_recorder = control_stack.pose_waypoint_recorder_node:main',
            'teach_waypoint_recorder = control_stack.teach_waypoint_recorder_node:main',
        ],
    },
)
