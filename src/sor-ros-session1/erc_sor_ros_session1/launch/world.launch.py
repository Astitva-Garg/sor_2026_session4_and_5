import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution

def generate_launch_description():
    pkg_session    = get_package_share_directory('erc_sor_ros_session1')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gazebo_models_path = os.path.expanduser('~/gazebo_models')
    if 'GZ_SIM_RESOURCE_PATH' in os.environ:
        os.environ['GZ_SIM_RESOURCE_PATH'] += os.pathsep + gazebo_models_path
    else:
        os.environ['GZ_SIM_RESOURCE_PATH'] = gazebo_models_path

    world_arg = DeclareLaunchArgument(
        'world', default_value='world.sdf',
        description='SDF world filename inside worlds/')

    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': [
                PathJoinSubstitution([pkg_session, 'worlds', LaunchConfiguration('world')]),
                TextSubstitution(text=' -r -v -v1'),
            ],
            'on_exit_shutdown': 'true',
        }.items()
    )

    return LaunchDescription([world_arg, gazebo_launch])
