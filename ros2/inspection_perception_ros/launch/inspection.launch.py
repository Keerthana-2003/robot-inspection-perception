from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("model", default_value="/models/best.onnx"),
        DeclareLaunchArgument("num_classes", default_value="1"),
        DeclareLaunchArgument("calibration", default_value="/config/calibration.yaml"),
    ]
    node = Node(
        package="inspection_perception_ros",
        executable="inspection_node",
        name="inspection_perception",
        parameters=[{
            "model": LaunchConfiguration("model"),
            "num_classes": LaunchConfiguration("num_classes"),
            "calibration": LaunchConfiguration("calibration"),
        }],
        remappings=[("image", "/camera/color/image_raw"), ("points", "/lidar/points"),
                    ("thermal", "/thermal/temperature")],
        output="screen",
    )
    return LaunchDescription([*args, node])
