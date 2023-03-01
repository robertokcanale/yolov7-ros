import rclpy
import torch
import yaml
from typing import List
from yolov7_ros.utils.plots import *
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, BoundingBox2D, ObjectHypothesisWithPose
from yolov7_ros.msg import ObjectsStamped, Object, BoundingBox2Di, BoundingBox2Df, Keypoint2Df, HumansStamped, Human
from geometry_msgs.msg import Pose2D

def create_header(node : rclpy.node.Node):
    h = Header()
    h.stamp = node.get_clock().now().to_msg()
    return h

def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    else:
        data_dict = None
    return data_dict

def create_detection_msg(node : rclpy.node.Node, img_msg: Image, detections: torch.Tensor) -> Detection2DArray:
    """
    :param img_msg: original ros image message
    :param detections: torch tensor of shape [num_boxes, 6] where each element is
        [x1, y1, x2, y2, confidence, class_id]
    :returns: detections as a ros message of type Detection2DArray
    """
    detection_array_msg = Detection2DArray()

    # header
    header = create_header(node)
    detection_array_msg.header = header
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        single_detection_msg = Detection2D()
        single_detection_msg.header = header

        # src img
        single_detection_msg.source_img = img_msg

        # bbox
        bbox = BoundingBox2D()
        w = int(round(x2 - x1))
        h = int(round(y2 - y1))
        cx = int(round(x1 + w / 2))
        cy = int(round(y1 + h / 2))
        bbox.size_x = w
        bbox.size_y = h

        bbox.center = Pose2D()
        bbox.center.x = cx
        bbox.center.y = cy

        single_detection_msg.bbox = bbox

        # class id & confidence
        obj_hyp = ObjectHypothesisWithPose()
        obj_hyp.id = int(cls)
        obj_hyp.score = conf
        single_detection_msg.results = [obj_hyp]

        detection_array_msg.detections.append(single_detection_msg)

    return detection_array_msg
   

def create_stamped_detection_msg(node : rclpy.node.Node, detections: torch.Tensor, class_names) -> ObjectsStamped:
    """
    :param img_msg: original ros image message
    :param detections: torch tensor of shape [num_boxes, 6] where each element is
        [x1, y1, x2, y2, confidence, class_id]
    :returns: detections as a ros message of type ObjectsStamped
    """
    detection_array_msg = ObjectsStamped()
    i = 0
    # header
    header = create_header(node)
    detection_array_msg.header = header
    for  idx, detection in enumerate(detections):
        x1, y1, x2, y2, conf, cls = detection.tolist()
        
        single_detection_msg = Object()
        # bbox
        w = float(round(x2 - x1))
        h = float(round(y2 - y1))
        cx = float(round(x1 + w / 2))
        cy = float(round(y1 + h / 2))

        bounding_box_2d = BoundingBox2Df()
        keypoint0 =  Keypoint2Df()
        keypoint1 =  Keypoint2Df()
        keypoint2 =  Keypoint2Df()
        keypoint3 =  Keypoint2Df()

        keypoint0.kp[0] =  x1
        keypoint0.kp[1] =  y1
        bounding_box_2d.corners[0] = keypoint0
        keypoint1.kp[0] =  x1
        keypoint1.kp[1] =  y2
        bounding_box_2d.corners[1] = keypoint1
        keypoint2.kp[0] =  x2
        keypoint2.kp[1] =  y2
        bounding_box_2d.corners[2] = keypoint2
        keypoint3.kp[0] =  x2
        keypoint3.kp[1] =  y1
        bounding_box_2d.corners[3] = keypoint3
        
        single_detection_msg.center = Pose2D()
        single_detection_msg.center.x = cx
        single_detection_msg.center.y = cy

        single_detection_msg.label = class_names[int(cls)]
        single_detection_msg.label_id = i
        single_detection_msg.confidence = conf
        
        single_detection_msg.bounding_box_2d = bounding_box_2d
        
        detection_array_msg.objects.append(single_detection_msg)
        i = i + 1
    
    return detection_array_msg


def create_humans_detection_msg(node : rclpy.node.Node, output) -> HumansStamped:
    keypoints_array_msg = HumansStamped()
    i = 0
    steps = 3
    # header
    keypoints_array_msg.header = create_header(node)
        
    for idx in range(output.shape[0]):
        single_detection_msg = Human()
        single_detection_msg.label_id = i 
        kpts = output[idx, 4:].T

        num_kpts = len(kpts) // steps 
        for kid in range(num_kpts):
            x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
            keypoint =  Keypoint2Df()
            if not (x_coord % 640 == 0 or y_coord % 640 == 0):
                if steps == 3:
                    conf = kpts[steps * kid + 2]
                    if conf < 0.5:
                        keypoint.kp[0] = 0.0
                        keypoint.kp[1] = 0.0
                        single_detection_msg.skeleton_2d.keypoints[kid] = keypoint
                keypoint.kp[0] = x_coord
                keypoint.kp[1] = y_coord
                single_detection_msg.skeleton_2d.keypoints[kid] = keypoint
        keypoints_array_msg.humans.append(single_detection_msg)
        i = i + 1
        
    return keypoints_array_msg