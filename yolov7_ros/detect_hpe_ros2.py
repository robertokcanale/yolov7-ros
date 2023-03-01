#!/usr/bin/python3

import torch, cv2, os, string, rclpy, sys
import numpy as np

from rclpy.node import Node
from torchvision import transforms
from yolov7_ros.utils.datasets import letterbox
from yolov7_ros.utils.general import non_max_suppression_kpt
from yolov7_ros.utils.plots import output_to_keypoint, plot_skeleton_kpts
from yolov7_ros.utils.ros import create_humans_detection_msg
from yolov7_ros.models.experimental import attempt_load
from sensor_msgs.msg import Image

from yolov7_ros.msg import HumansStamped, Human
from cv_bridge import CvBridge
from torchvision.transforms import ToTensor

class YoloV7_HPE:
    def __init__(self, weights: string ="../weights/yolov7-w6-pose.pt", device: str = "cuda"):
        self.device = torch.device(device)
        self.weigths = torch.load(weights, map_location=self.device)
        # self.model = self.weigths['model']
        self.model = attempt_load(weights, map_location=self.device)
        _ = self.model.float().eval()
        
        if torch.cuda.is_available():
            self.model.half().to(device)
    

class Yolov7_HPEPublisher(Node):

    def __init__(self,):    
        """
        :param weights: path/to/yolo_weights.pt
        :param img_topic: name of the image topic to listen to
        :param out_img_topic: topic for visualization will be published under the
            namespace '/yolov7')
        :param skeleton_keypoints_out_topic: intersection over union threshold will be published under the
            namespace '/yolov7')
        :param device: device to do inference on (e.g., 'cuda' or 'cpu')
        :param queue_size: queue size for publishers
        :visualize: flag to enable publishing the detections visualized in the image
        """
        super().__init__('yolo_hpe_ros2')

        self.declare_parameter(name = 'img_topic', value = '/realsense_front/color/image_raw')
        self.declare_parameter(name = 'weights', value = '/yolov7_ros/weights/yolov7-w6-pose.pt')
        self.declare_parameter(name = 'out_img_topic', value = 'yolov7_detection')
        self.declare_parameter(name = 'skeleton_keypoints_out_topic', value = 'skeleton_keypoints_out_topic')
        self.declare_parameter(name = 'device', value = 'cuda')
        self.declare_parameter(name = 'queue_size', value = 1)
        self.declare_parameter(name = 'visualize', value = True)        
        
        self.img_topic = self.get_parameter("img_topic").value
        self.weights = self.get_parameter("weights").value
        self.out_img_topic = self.get_parameter("out_img_topic").value
        self.skeleton_keypoints_out_topic = self.get_parameter("skeleton_keypoints_out_topic").value
        self.device = self.get_parameter("device").value
        self.queue_size = self.get_parameter("queue_size").value
        self.visualize = self.get_parameter("visualize").value

        # some sanity checks
        if not os.path.isfile(self.weights):
            raise FileExistsError("Weights not found.")

        if not ("cuda" in self.device or "cpu" in self.device):
            raise ValueError("Check your device.")

        self.tensorize = ToTensor()
        self.model = YoloV7_HPE( weights = self.weights , device = self.device)
        self.bridge = CvBridge()
        
        #Subscribe to Image
        self.img_subscriber = self.create_subscription(Image, self.img_topic, self.process_img_msg, qos_profile=self.queue_size)
        
        #Visualization Publisher
        self.out_img_topic = self.out_img_topic + "visualization" if self.out_img_topic.endswith("/") else self.out_img_topic + "/visualization"
        self.visualization_publisher =  self.create_publisher(Image, self.out_img_topic, qos_profile=self.queue_size) if self.visualize else None
        
        #Keypoints Publisher
        self.skeleton_keypoints_out_topic = self.skeleton_keypoints_out_topic + "visualization" if self.out_img_topic.endswith("/") else self.skeleton_keypoints_out_topic + "/visualization"
        self.skeleton_detection_publisher =  self.create_publisher(HumansStamped, self.skeleton_keypoints_out_topic, qos_profile=self.queue_size)
        self.get_logger().info('YOLOv7HPE initialization complete. Ready to start inference')

    def process_img_msg(self, image: Image):
        """ callback function for publisher """
        image = self.bridge.imgmsg_to_cv2(image, "bgr8")    
        image = letterbox(image, 960, stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))

        if torch.cuda.is_available():
            image = image.half().to(self.device)   
        with torch.no_grad():
            output, _ = self.model.model(image)
            
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=self.model.model.yaml['nc'], nkpt=self.model.model.yaml['nkpt'], kpt_label=True)
        output = output_to_keypoint(output)
        nimg = image[0].permute(1, 2, 0) * 255
        nimg = nimg.cpu().numpy().astype(np.uint8)
        nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
        for idx in range(output.shape[0]):
            plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
       
        #Publishing Keypoints
        if (len(output) > 0):
            keypoints_array_msg = create_humans_detection_msg(self, output)
            self.skeleton_detection_publisher.publish(keypoints_array_msg)
        
        #Publishing Visualization if Required
        if self.visualization_publisher:
            vis_msg = self.bridge.cv2_to_imgmsg(nimg)
            self.visualization_publisher.publish(vis_msg)

def main(args=None):
    rclpy.init(args=args)

    yolo_hpe_publisher =  Yolov7_HPEPublisher()
    rclpy.spin(yolo_hpe_publisher)

    yolo_hpe_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()






