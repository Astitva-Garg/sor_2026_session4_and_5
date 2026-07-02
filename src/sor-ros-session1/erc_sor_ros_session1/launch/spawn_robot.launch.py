import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg = get_package_share_directory('erc_sor_ros_session1')

    gazebo_models_path, _ = os.path.split(pkg)
    if 'GZ_SIM_RESOURCE_PATH' in os.environ:
        os.environ['GZ_SIM_RESOURCE_PATH'] += os.pathsep + gazebo_models_path
    else:
        os.environ['GZ_SIM_RESOURCE_PATH'] = gazebo_models_path

    rviz_arg  = DeclareLaunchArgument('rviz',          default_value='true')
    world_arg = DeclareLaunchArgument('world',         default_value='world.sdf')
    model_arg = DeclareLaunchArgument('model',         default_value='my_robot.xacro')
    use_sim_time_arg = DeclareLaunchArgument('use_sim_time', default_value='true')

    urdf_file_path = PathJoinSubstitution([pkg, 'urdf', LaunchConfiguration('model')])

    world_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'world.launch.py')),
        launch_arguments={'world': LaunchConfiguration('world')}.items()
    )

    robot_state_publisher_node = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='robot_state_publisher', output='screen',
        parameters=[{'robot_description': Command(['xacro', ' ', urdf_file_path]),
                     'use_sim_time': LaunchConfiguration('use_sim_time')}],
        remappings=[('/tf', 'tf'), ('/tf_static', 'tf_static')]
    )

    spawn_urdf_node = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-name', 'my_robot', '-topic', 'robot_description',
                   '-x', '0.0', '-y', '0.0', '-z', '0.5', '-Y', '0.0'],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Main bridge — all topics except the raw camera image (handled by image_bridge)
    gz_bridge_node = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        name='parameter_bridge',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/joint_states@sensor_msgs/msg/JointState[gz.msgs.Model',
            # /tf is commented out: EKF publishes the filtered tf instead
            # '/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/camera/depth_image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
            '/scan@sensor_msgs/msg/LaserScan@gz.msgs.LaserScan',
            '/scan/points@sensor_msgs/msg/PointCloud2@gz.msgs.PointCloudPacked',
            '/imu@sensor_msgs/msg/Imu@gz.msgs.IMU',
        ],
        output='screen',
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # Compressed image bridge (replaces raw /camera/image bridge)
    gz_image_bridge_node = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera/image'],
        output='screen',
        parameters=[{
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'camera.image.compressed.jpeg_quality': 75,
        }],
    )

    # Relay: republish /camera/camera_info → /camera/image/camera_info
    # so RViz compressed-image display finds its matching camera_info topic
    relay_camera_info_node = Node(
        package='topic_tools',
        executable='relay',
        name='relay_camera_info',
        output='screen',
        arguments=['camera/camera_info', 'camera/image/camera_info'],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    # EKF node — fuses /odom + /imu → /odometry/filtered and publishes tf
    ekf_node = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[
            os.path.join(pkg, 'config', 'ekf.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')},
        ]
    )

    rviz_node = Node(
        package='rviz2', executable='rviz2', name='rviz2',
        arguments=['-d', os.path.join(pkg, 'rviz', 'rviz.rviz')],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    return LaunchDescription([
        rviz_arg, world_arg, model_arg, use_sim_time_arg,
        world_launch,
        robot_state_publisher_node,
        spawn_urdf_node,
        gz_bridge_node,
        gz_image_bridge_node,
        relay_camera_info_node,
        ekf_node,
        rviz_node,
    ])
