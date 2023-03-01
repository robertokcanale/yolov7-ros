from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution



def generate_launch_description():
    yolov7_ros_ns = LaunchConfiguration('yolov7_ros')

    yolov7_ros2 = Node(
        package='yolov7_ros',
        executable='detect_hpe_ros2.py',
        name='yolov7_hpe_ros',
        parameters=[
            {"weights": PathJoinSubstitution([FindPackageShare('yolov7_ros'), 'weights','yolov7-w6-pose.pt' ])},
            {"img_topic": '/image_raw'},
            {"out_img_topic": 'yolov7_detection'},
            {"skeleton_keypoints_out_topic": 'skeleton_keypoints_out_topic'},
            {"queue_size": 1},
            {"visualize": True},            
            {"device": 'cuda'}],
        output='screen',
    )

    return LaunchDescription([
    yolov7_ros2
    ])