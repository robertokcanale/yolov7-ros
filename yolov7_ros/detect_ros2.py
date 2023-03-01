#!/usr/bin/python3

import torch, cv2, os, string, rclpy
import numpy as np

from rclpy.node import Node
from yolov7_ros.utils.general import non_max_suppression
from yolov7_ros.utils.ros import create_detection_msg, create_stamped_detection_msg
from yolov7_ros.visualizer import draw_detections
from yolov7_ros.utils.ros import load_yaml
from yolov7_ros.models.experimental import attempt_load
from typing import Tuple, Union
from torchvision.transforms import ToTensor

from yolov7_ros.msg import ObjectsStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class YoloV7:
    def __init__(self, weights: string ="../weights/yolov7.pt", conf_thresh: float = 0.5, iou_thresh: float = 0.45,
                 device: str = "cuda"):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = torch.device(device)
        self.model = attempt_load(weights, map_location=self.device)
    
    @torch.no_grad()
    def inference(self, img: torch.Tensor):
        """
        :param img: tensor [c, h, w]
        :returns: tensor of shape [num_boxes, 6], where each item is represented as
            [x1, y1, x2, y2, confidence, class_id]
        """
        img = img.unsqueeze(0)
        pred_results = self.model(img)[0]
        detections = non_max_suppression(
            pred_results, conf_thres=self.conf_thresh, iou_thres=self.iou_thresh
        )
        if detections:
            detections = detections[0]
        return detections


class Yolov7Publisher(Node):

    def __init__(self):              
        """
        :param img_topic: name of the image topic to listen to
        :param weights: path/to/yolo_weights.pt
        :param conf_thresh: confidence threshold
        :param iou_thresh: intersection over union threshold
        :param pub_topic: name of the output topic (will be published under the namespace '/yolov7')
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        :param img_size: (height, width) to which the img is resized before being
            fed into the yolo network. Final output coordinates will be rescaled to
            the original img size.
        """    
        super().__init__('yolo_ros2')

        self.declare_parameter(name = 'img_topic', value = '/realsense_front/color/image_raw')
        self.declare_parameter(name = 'weights', value = '/yolov7_ros/weights/yolov7.pt')
        self.declare_parameter(name = 'conf_thresh', value = 0.75)
        self.declare_parameter(name = 'iou_thresh', value = 0.45)
        self.declare_parameter(name = 'pub_topic', value = 'yolov7_detection')
        self.declare_parameter(name = 'device', value = 'cuda')
        self.declare_parameter(name = 'yaml', value = 'yolov7_ros/conf/coco.yaml')
        self.declare_parameter(name = 'img_size', value = 640)
        self.declare_parameter(name = 'queue_size', value = 1)
        self.declare_parameter(name = 'visualize', value = True)

        self.weights = self.get_parameter("weights").value
        self.img_topic =  self.get_parameter("img_topic").value
        self.conf_thresh = self.get_parameter("conf_thresh").value
        self.iou_thresh = self.get_parameter("iou_thresh").value
        self.pub_topic = self.get_parameter("pub_topic").value
        self.device = self.get_parameter("device").value
        self.yaml = self.get_parameter("yaml").value
        self.img_size = (self.get_parameter("img_size").value, self.get_parameter("img_size").value)
        self.queue_size = self.get_parameter("queue_size").value
        self.visualize = self.get_parameter("visualize").value

        # some sanity checks
        if not os.path.isfile(self.weights):
            raise FileExistsError("Weights not found.")
            
        if not ("cuda" in self.device or "cpu" in self.device):
            raise ValueError("Check your device.")

        self.class_names = load_yaml(self.yaml)['names']
        
        self.tensorize = ToTensor()
        self.model = YoloV7(weights=self.weights, conf_thresh=self.conf_thresh, iou_thresh=self.iou_thresh, device=self.device)
        self.bridge = CvBridge()

        #Topic Name
        self.vis_topic = self.pub_topic + "visualization" if self.pub_topic.endswith("/") else \
            self.pub_topic + "/visualization"
        #Subscribers
        self.img_subscriber = self.create_subscription(Image, self.img_topic, self.process_img_msg, qos_profile=self.queue_size)
        #Publishers
        self.visualization_publisher = self.create_publisher(Image, self.vis_topic, qos_profile=self.queue_size)
        self.detection_publisher = self.create_publisher(ObjectsStamped, self.pub_topic, qos_profile=self.queue_size)
        self.get_logger().info('YOLOv7 initialization complete. Ready to start inference')


    def rescale(self, ori_shape: Tuple[int, int], boxes: Union[torch.Tensor, np.ndarray],target_shape: Tuple[int, int]):
        """Rescale the output to the original image shape
        :param ori_shape: original width and height [width, height].
        :param boxes: original bounding boxes as a torch.Tensor or np.array or shape
            [num_boxes, >=4], where the first 4 entries of each element have to be
            [x1, y1, x2, y2].
        :param target_shape: target width and height [width, height].
        """
        xscale = target_shape[1] / ori_shape[1]
        yscale = target_shape[0] / ori_shape[0]

        boxes[:, [0, 2]] *= xscale
        boxes[:, [1, 3]] *= yscale

        return boxes
    
    def process_img_msg(self, img_msg: Image):
        """ callback function for publisher """
        np_img_orig = self.bridge.imgmsg_to_cv2(
            img_msg, desired_encoding='passthrough'
        )
        # handle possible different img formats
        if len(np_img_orig.shape) == 2:
            np_img_orig = np.stack([np_img_orig] * 3, axis=2)

        h_orig, w_orig, c = np_img_orig.shape
        if c == 1:
            np_img_orig = np.concatenate([np_img_orig] * 3, axis=2)
            c = 3

        # Automatically resize the image to the next smaller possible size
        w_scaled, h_scaled = self.img_size

        # w_scaled = w_orig - (w_orig % 8)
        np_img_resized = cv2.resize(np_img_orig, (w_scaled, h_scaled))

        #Conversion to torch tensor (copied from original yolov7 repo)
        if np_img_resized.shape[2] == 4: #Removing extra channel if RGBA
            np_img_resized = np_img_resized[:,:,:3]
        img = np_img_resized.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(np.ascontiguousarray(img))
        img = img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.
        img = img.to(self.device)

        #Inference & rescaling the output to original img size
        detections = self.model.inference(img)
        detections[:, :4] = self.rescale(
            [h_scaled, w_scaled], detections[:, :4], [h_orig, w_orig])
        detections[:, :4] = detections[:, :4].round()
        
        #Publishing Detections
        if(len(detections) > 0):
            detection_msg = create_stamped_detection_msg(self, detections, self.class_names)
            self.detection_publisher.publish(detection_msg)

        #Publishing Visualization if Required
        if self.visualization_publisher:
            bboxes = [[int(x1), int(y1), int(x2), int(y2)]
                      for x1, y1, x2, y2 in detections[:, :4].tolist()]
            prediction_scores = [float(p) for p in detections[:, 4].tolist()]
            classes = [int(c) for c in detections[:, 5].tolist()]
            vis_img = draw_detections(np_img_orig, bboxes, classes, prediction_scores, self.class_names)
            vis_msg = self.bridge.cv2_to_imgmsg(vis_img)
            self.visualization_publisher.publish(vis_msg)

def main(args=None):
    rclpy.init(args=args)
   
    yolo_publisher =  Yolov7Publisher()
    rclpy.spin(yolo_publisher)

    yolo_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()