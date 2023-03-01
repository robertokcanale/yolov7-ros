from setuptools import setup
import os, glob
package_name = 'yolov7_ros'
submodules_models = "yolov7_ros/models"
submodules_utils = "yolov7_ros/utils"

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, submodules_models, submodules_utils],
    data_files=[
        ('share/ament_index/resource_index/packages',['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='roberto',
    maintainer_email='canaler@artc.a-star.edu.sg',
    description='Yolov7 ROS2',
    license='MIT Licence',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect_ros2 = yolov7_ros.detect_ros2:main',
            'detect_hpe_ros2 = yolov7_ros.detect_hpe_ros2:main',
        ],
    },
)
