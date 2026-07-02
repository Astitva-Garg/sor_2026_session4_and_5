import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    pkg = FindPackageShare('erc_sor_ros_session1')

    gui_arg = DeclareLaunchArgument(
        name='gui', default_value='true', choices=['true', 'false'],
        description='Show joint_state_publisher_gui')
    rviz_arg = DeclareLaunchArgument(
        name='rvizconfig',
        default_value=PathJoinSubstitution([pkg, 'rviz', 'urdf.rviz']),
        description='RViz config path')
    model_arg = DeclareLaunchArgument(
        name='model', default_value='my_robot.xacro',
        description='Xacro file in urdf/')

    urdf_display = IncludeLaunchDescription(
        PathJoinSubstitution([
            FindPackageShare('urdf_launch'), 'launch', 'display.launch.py'
        ]),
        launch_arguments={
            'urdf_package':      'erc_sor_ros_session1',
            'urdf_package_path': PathJoinSubstitution(['urdf', LaunchConfiguration('model')]),
            'rviz_config':       LaunchConfiguration('rvizconfig'),
            'jsp_gui':           LaunchConfiguration('gui'),
        }.items()
    )

    return LaunchDescription([gui_arg, rviz_arg, model_arg, urdf_display])
