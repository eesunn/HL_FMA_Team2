from glob import glob
import os

from setuptools import find_packages, setup

package_name = 'imu_stack'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='김윤기',
    maintainer_email='kyg100800@gmail.com',
    description='IMU stack: HFI-A9 driver + (future) hill_detector.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'hfi_a9_node = imu_stack.hfi_a9_node:main',
        ],
    },
)
