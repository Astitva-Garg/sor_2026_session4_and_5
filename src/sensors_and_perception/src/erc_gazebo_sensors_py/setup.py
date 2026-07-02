from setuptools import setup

package_name = 'erc_gazebo_sensors_py'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Python perception nodes — OpenCV and YOLOv8',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'chase_the_ball = erc_gazebo_sensors_py.chase_the_ball:main',
            'yolo_detection  = erc_gazebo_sensors_py.yolo_detection_node:main',
        ],
    },
)
