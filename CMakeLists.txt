cmake_minimum_required(VERSION 3.5)
project(yolov7_ros)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate Messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoundingBox2Df.msg"
  "msg/BoundingBox2Di.msg"
  "msg/Keypoint2Df.msg"
  "msg/Keypoint2Di.msg"
  "msg/Object.msg"
  "msg/ObjectsStamped.msg"
  "msg/Human.msg"
  "msg/HumansStamped.msg"
  "msg/Skeleton2D.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs vision_msgs # Add packages that above messages depend on, in this case geometry_msgs for Sphere.msg
)

# Include Cpp "include" directory
#include_directories(include)

# Create Cpp executable
#add_executable(cpp_executable src/cpp_node.cpp)
#ament_target_dependencies(cpp_executable rclcpp)

# Install Cpp executables
#install(TARGETS
#  cpp_executable
#  DESTINATION lib/${PROJECT_NAME}
#)
#Install Directories
install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY weights
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY conf
  DESTINATION share/${PROJECT_NAME}
)
install(
  DIRECTORY yolov7_ros
  DESTINATION share/${PROJECT_NAME}
)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  yolov7_ros/detect_ros2.py
  yolov7_ros/detect_hpe_ros2.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()