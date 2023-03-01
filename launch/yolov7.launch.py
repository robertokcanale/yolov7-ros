from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution


def generate_launch_description():
    yolov7_ros_ns = LaunchConfiguration('yolov7_ros')
    
    yolov7_ros2 = Node(
        package='yolov7_ros',
        executable='detect_ros2.py',
        name='yolov7_ros',
        parameters=[
            {"weights": PathJoinSubstitution([FindPackageShare('yolov7_ros'), 'weights','yolov7.pt' ])},
            {"img_topic": '/realsense_front/color/image_raw'},
            {"pub_topic": 'yolov7_detection'},
            {"conf_thresh": 0.7},
            {"img_size": 640},
            {"iou_thresh": 0.45},
            {"queue_size":1},
            {"visualize": True},            
            {"device": 'cuda'},            
            {"yaml": PathJoinSubstitution([FindPackageShare('yolov7_ros'), 'conf','coco.yaml'])}],
        output='screen',
    )

    return LaunchDescription([
    yolov7_ros2
    ])