#include "pure_pursuit.hpp"

#include <math.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Eigen>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "std_msgs/msg/int8.hpp"  // Deadman 스위치 상태 메시지

int PurePursuit::is_starting_value = 0;  // 정적 멤버 변수 정의 및 초기화

bool first_path_completed = false; // 첫 번째 경로 완료 상태 추적 변수

PurePursuit::PurePursuit() : Node("pure_pursuit_node") {
    // initialise parameters
    this->declare_parameter("waypoints_path", "/home/sm1234/f1tenth_ws/src/waypoint_generator/map_point/smoothed_waypoints.csv");
    this->declare_parameter("second_waypoints_path", "/home/sm1234/f1tenth_ws/src/waypoint_generator/map_point/smoothed_waypoints.csv");
    this->declare_parameter("odom_topic", "/pf/pose/odom");
    this->declare_parameter("car_refFrame", "ego_racecar/base_link");
    this->declare_parameter("drive_topic", "/drive");
    this->declare_parameter("rviz_current_waypoint_topic", "/current_waypoint");
    this->declare_parameter("rviz_lookahead_waypoint_topic", "/lookahead_waypoint");
    this->declare_parameter("global_refFrame", "map");
    this->declare_parameter("min_lookahead", 0.5);
    this->declare_parameter("max_lookahead", 1.0);
    this->declare_parameter("lookahead_ratio", 8.0);
    this->declare_parameter("K_p", 0.5);
    this->declare_parameter("steering_limit", 25.0);
    this->declare_parameter("velocity_percentage", 0.6);
    this->declare_parameter("acceleration_rate", 0.12);

    // Default Values
    acceleration_rate = this->get_parameter("acceleration_rate").as_double();
    
    waypoints_path = this->get_parameter("waypoints_path").as_string();
    second_waypoints_path = this->get_parameter("second_waypoints_path").as_string();
    odom_topic = this->get_parameter("odom_topic").as_string();
    car_refFrame = this->get_parameter("car_refFrame").as_string();
    drive_topic = this->get_parameter("drive_topic").as_string();
    rviz_current_waypoint_topic = this->get_parameter("rviz_current_waypoint_topic").as_string();
    rviz_lookahead_waypoint_topic = this->get_parameter("rviz_lookahead_waypoint_topic").as_string();
    global_refFrame = this->get_parameter("global_refFrame").as_string();
    min_lookahead = this->get_parameter("min_lookahead").as_double();
    max_lookahead = this->get_parameter("max_lookahead").as_double();
    lookahead_ratio = this->get_parameter("lookahead_ratio").as_double();
    K_p = this->get_parameter("K_p").as_double();
    steering_limit = this->get_parameter("steering_limit").as_double();
    velocity_percentage = this->get_parameter("velocity_percentage").as_double();

    subscription_odom = this->create_subscription<nav_msgs::msg::Odometry>(odom_topic, 25, std::bind(&PurePursuit::odom_callback, this, _1));
    timer_ = this->create_wall_timer(2000ms, std::bind(&PurePursuit::timer_callback, this));

    publisher_drive = this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(drive_topic, 25);
    vis_current_point_pub = this->create_publisher<visualization_msgs::msg::Marker>(rviz_current_waypoint_topic, 10);
    vis_lookahead_point_pub = this->create_publisher<visualization_msgs::msg::Marker>(rviz_lookahead_waypoint_topic, 10);

    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
    transform_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Deadman 스위치 상태를 구독하여 가속도 활성화 여부를 제어합니다
    trigger_subscription = this->create_subscription<std_msgs::msg::Int8>(
        "/pure_pursuit/trigger", 10,
        std::bind(&PurePursuit::trigger_callback, this, _1));

    RCLCPP_INFO(this->get_logger(), "Pure pursuit node has been launched");

   load_waypoints();         // 첫 번째 경로 로드
    load_second_waypoints();   // 두 번째 경로 로드
}

// Deadman 스위치 콜백 함수
void PurePursuit::trigger_callback(const std_msgs::msg::Int8::SharedPtr msg) {
    if (msg->data == 0 ) {  // Deadman 스위치가 처음 눌렸을 때
        
        is_starting_value = 1; // 가속도 기능 초기화
        curr_velocity_pure = 0;
        //RCLCPP_INFO(this->get_logger(), "Deadman switch pressed. Autonomous mode enabled with acceleration.");
    } 
}

// 가속도 기능을 통해 목표 속도에 점진적으로 도달하게 하는 함수
double PurePursuit::get_velocity(double steering_angle) {
    double target_velocity = 0.0;
    
      double acceleration_rate = this->get_parameter("acceleration_rate").as_double();
      //RCLCPP_INFO(this->get_logger(), "get_velocity startsatrtcsascasdasdasddas");
      if (waypoints.V[waypoints.velocity_index])
       {
            target_velocity = waypoints.V[waypoints.velocity_index] * velocity_percentage;
        } 
      else 
       {
            if (abs(steering_angle) < to_radians(10.0)) 
            {
                target_velocity = 6.0 * velocity_percentage;
            } 
            else if (abs(steering_angle) <= to_radians(20.0))
             {
                target_velocity = 2.5 * velocity_percentage;
            }
             else 
             {
                target_velocity = 2.0 * velocity_percentage;
             }
        }
        //RCLCPP_INFO(this->get_logger(), "target_velocity: %.2fm/s", target_velocity);
	
	
	//RCLCPP_INFO(this->get_logger(), "acceleration_rate: %2f", acceleration_rate);
	//RCLCPP_INFO(this->get_logger(), "is_starting_value: %d", is_starting_value);
	//RCLCPP_INFO(this->get_logger(), "curr_velocity_pure: %2f", curr_velocity_pure);
	
        // Deadman 스위치가 처음 눌린 상태일 때만 가속도 기능 활성화
       if (is_starting_value) 
        { 
        //RCLCPP_INFO(this->get_logger(), "startstartstartstartstartstart");
            if (curr_velocity_pure < target_velocity) 
            {
            
                curr_velocity_pure = std::min(curr_velocity_pure + acceleration_rate, target_velocity);
                //RCLCPP_INFO(this->get_logger(), "curr_velocity_pure: %.2fm/s", curr_velocity_pure);
            }
             	
            else
            {
                is_starting_value = 0; // 목표 속도에 도달하면 가속도 기능 비활성화
                //RCLCPP_INFO(this->get_logger(), "endendendendendendendendend");
            }
            
        } 
        else 
        {
            curr_velocity_pure = target_velocity; // 가속도 없이 바로 목표 속도로 설정
        }
         return curr_velocity_pure;
    } 

   


// 주행 메시지 발행 함수
void PurePursuit::publish_message(double steering_angle) {
    auto drive_msgObj = ackermann_msgs::msg::AckermannDriveStamped();
    if (steering_angle < 0.0) {
        drive_msgObj.drive.steering_angle = std::max(steering_angle, -to_radians(steering_limit));  // ensure steering angle is dynamically capable
    } else {
        drive_msgObj.drive.steering_angle = std::min(steering_angle, to_radians(steering_limit));  // ensure steering angle is dynamically capable
    }

    curr_velocity = get_velocity(drive_msgObj.drive.steering_angle);
    drive_msgObj.drive.speed = curr_velocity;

    //RCLCPP_INFO(this->get_logger(), "index: %d ... distance: %.2fm ... Speed: %.2fm/s ... Steering Angle: %.2f ... K_p: %.2f ... velocity_percentage: %.2f", waypoints.index, p2pdist(waypoints.X[waypoints.index], x_car_world, waypoints.Y[waypoints.index], y_car_world), drive_msgObj.drive.speed, to_degrees(drive_msgObj.drive.steering_angle), K_p, velocity_percentage);
    //RCLCPP_INFO(this->get_logger(), "Speed: %.2fm/s", drive_msgObj.drive.speed);


    publisher_drive->publish(drive_msgObj);
}

void PurePursuit::odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_submsgObj) {
    x_car_world = odom_submsgObj->pose.pose.position.x;
    y_car_world = odom_submsgObj->pose.pose.position.y;

    get_waypoint();
    transformandinterp_waypoint();

    double steering_angle = p_controller();
    publish_message(steering_angle);
}

void PurePursuit::timer_callback() {
    acceleration_rate =  this->get_parameter("acceleration_rate").as_double();
    K_p = this->get_parameter("K_p").as_double();
    velocity_percentage = this->get_parameter("velocity_percentage").as_double();
    min_lookahead = this->get_parameter("min_lookahead").as_double();
    max_lookahead = this->get_parameter("max_lookahead").as_double();
    lookahead_ratio = this->get_parameter("lookahead_ratio").as_double();
    steering_limit = this->get_parameter("steering_limit").as_double();
}

// 각종 변환 및 거리 계산 함수
double PurePursuit::to_radians(double degrees) {
    return degrees * M_PI / 180.0;
}

double PurePursuit::to_degrees(double radians) {
    return radians * 180.0 / M_PI;
}

double PurePursuit::p2pdist(double &x1, double &x2, double &y1, double &y2) {
    return sqrt(pow((x2 - x1), 2) + pow((y2 - y1), 2));
}

void PurePursuit::load_waypoints() {
    csvFile_waypoints.open(waypoints_path, std::ios::in);

    if (!csvFile_waypoints.is_open()) {
        RCLCPP_ERROR(this->get_logger(), "Cannot Open CSV File: %s", waypoints_path);
        return;
    } else {
        RCLCPP_INFO(this->get_logger(), "CSV111111111 File Opened");
    }

    std::string line, word;
    while (std::getline(csvFile_waypoints, line, '\n')) {
        std::stringstream s(line);
        int j = 0;
        while (getline(s, word, ',')) {
            if (!word.empty()) {
                if (j == 0) {
                    waypoints.X.push_back(std::stod(word));
                } else if (j == 1) {
                    waypoints.Y.push_back(std::stod(word));
                } else if (j == 2) {
                    waypoints.V.push_back(std::stod(word));
                }
            }
            j++;
        }
    }

    csvFile_waypoints.close();
    num_waypoints = waypoints.X.size();
    RCLCPP_INFO(this->get_logger(), "Finished loading %d waypoints from %s", num_waypoints, waypoints_path);

    double average_dist_between_waypoints = 0.0;
    for (int i = 0; i < num_waypoints - 1; i++) {
        average_dist_between_waypoints += p2pdist(waypoints.X[i], waypoints.X[i + 1], waypoints.Y[i], waypoints.Y[i + 1]);
    }
    average_dist_between_waypoints /= num_waypoints;
    RCLCPP_INFO(this->get_logger(), "Average distance between waypoints: %f", average_dist_between_waypoints);
}

void PurePursuit::visualize_lookahead_point(Eigen::Vector3d &point) {
    auto marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = "map";
    marker.header.stamp = rclcpp::Clock().now();
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.25;
    marker.scale.y = 0.25;
    marker.scale.z = 0.25;
    marker.color.a = 1.0;
    marker.color.r = 1.0;

    marker.pose.position.x = point(0);
    marker.pose.position.y = point(1);
    marker.id = 1;
    vis_lookahead_point_pub->publish(marker);
}

void PurePursuit::visualize_current_point(Eigen::Vector3d &point) {
    auto marker = visualization_msgs::msg::Marker();
    marker.header.frame_id = "map";
    marker.header.stamp = rclcpp::Clock().now();
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.25;
    marker.scale.y = 0.25;
    marker.scale.z = 0.25;
    marker.color.a = 1.0;
    marker.color.b = 1.0;

    marker.pose.position.x = point(0);
    marker.pose.position.y = point(1);
    marker.id = 1;
    vis_current_point_pub->publish(marker);
}

void PurePursuit::get_waypoint() {
    if (!first_path_completed && waypoints.index >= num_waypoints - 2) {  // 첫 번째 경로 완료 시 두 번째 경로로 전환
        first_path_completed = true;
        waypoints = second_waypoints;
        num_waypoints = waypoints.X.size();
        waypoints.index = 0;  // 두 번째 경로 시작점으로 인덱스 초기화
        RCLCPP_INFO(this->get_logger(), "First path completed. Switching to second path.");
    }


    // 일반적인 웨이포인트 탐색 로직
    double longest_distance = 0;
    int final_i = -1;
    int start = waypoints.index;
    int end = (waypoints.index + 40) % num_waypoints;

    double lookahead = std::min(std::max(min_lookahead, max_lookahead * curr_velocity / lookahead_ratio), max_lookahead);

    if (end < start) {
        for (int i = start; i < num_waypoints; i++) {
            if (p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) <= lookahead &&
                p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) >= longest_distance) {
                longest_distance = p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
                final_i = i;
            }
        }
        for (int i = 0; i < end; i++) {
            if (p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) <= lookahead &&
                p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) >= longest_distance) {
                longest_distance = p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
                final_i = i;
            }
        }
    } else {
        for (int i = start; i < end; i++) {
            if (p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) <= lookahead &&
                p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) >= longest_distance) {
                longest_distance = p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
                final_i = i;
            }
        }
    }

    if (final_i == -1) {
        final_i = 0;
        for (int i = 0; i < num_waypoints; i++) {
            if (p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) <= lookahead &&
                p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) >= longest_distance) {
                longest_distance = p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
                final_i = i;
            }
        }
    }

    double shortest_distance = p2pdist(waypoints.X[0], x_car_world, waypoints.Y[0], y_car_world);
    int velocity_i = 0;
    for (int i = 0; i < num_waypoints; i++) {
        if (p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world) <= shortest_distance) {
            shortest_distance = p2pdist(waypoints.X[i], x_car_world, waypoints.Y[i], y_car_world);
            velocity_i = i;
        }
    }

    waypoints.index = final_i;
    waypoints.velocity_index = velocity_i;
}

// 두 번째 경로 로드
void PurePursuit::load_second_waypoints() {
    std::ifstream second_csvFile(second_waypoints_path);
    if (!second_csvFile.is_open()) 
    {
        RCLCPP_ERROR(this->get_logger(), "Cannot Open CSV File: %s", second_waypoints_path);
          return;
        }
        else 
        {
        RCLCPP_INFO(this->get_logger(), "CSV22222222 File Opened");
    }

      
    

    std::string line, word;
    while (std::getline(second_csvFile, line)) {
        std::stringstream s(line);
        int j = 0;
        while (getline(s, word, ',')) {
            if (!word.empty()) {
                if (j == 0) {
                    second_waypoints.X.push_back(std::stod(word));
                } else if (j == 1) {
                    second_waypoints.Y.push_back(std::stod(word));
                } else if (j == 2) {
                    second_waypoints.V.push_back(std::stod(word));
                }
            }
            j++;
        }
    }
    second_csvFile.close();
    RCLCPP_INFO(this->get_logger(), "Finished loading %d waypoints from %s", second_waypoints.X.size(), second_waypoints_path);
}

void PurePursuit::quat_to_rot(double q0, double q1, double q2, double q3) {
    double r00 = (double)(2.0 * (q0 * q0 + q1 * q1) - 1.0);
    double r01 = (double)(2.0 * (q1 * q2 - q0 * q3));
    double r02 = (double)(2.0 * (q1 * q3 + q0 * q2));

    double r10 = (double)(2.0 * (q1 * q2 + q0 * q3));
    double r11 = (double)(2.0 * (q0 * q0 + q2 * q2) - 1.0);
    double r12 = (double)(2.0 * (q2 * q3 - q0 * q1));

    double r20 = (double)(2.0 * (q1 * q3 - q0 * q2));
    double r21 = (double)(2.0 * (q2 * q3 + q0 * q1));
    double r22 = (double)(2.0 * (q0 * q0 + q3 * q3) - 1.0);

    rotation_m << r00, r01, r02, r10, r11, r12, r20, r21, r22;
}

void PurePursuit::transformandinterp_waypoint() {
    waypoints.lookahead_point_world << waypoints.X[waypoints.index], waypoints.Y[waypoints.index], 0.0;
    waypoints.current_point_world << waypoints.X[waypoints.velocity_index], waypoints.Y[waypoints.velocity_index], 0.0;

    visualize_lookahead_point(waypoints.lookahead_point_world);
    visualize_current_point(waypoints.current_point_world);

    geometry_msgs::msg::TransformStamped transformStamped;

    try {
        transformStamped = tf_buffer_->lookupTransform(car_refFrame, global_refFrame, tf2::TimePointZero);
    } catch (tf2::TransformException &ex) {
        RCLCPP_INFO(this->get_logger(), "Could not transform. Error: %s", ex.what());
    }

    Eigen::Vector3d translation_v(transformStamped.transform.translation.x, transformStamped.transform.translation.y, transformStamped.transform.translation.z);
    quat_to_rot(transformStamped.transform.rotation.w, transformStamped.transform.rotation.x, transformStamped.transform.rotation.y, transformStamped.transform.rotation.z);

    waypoints.lookahead_point_car = (rotation_m * waypoints.lookahead_point_world) + translation_v;
}

double PurePursuit::p_controller() {
    double r = waypoints.lookahead_point_car.norm();
    double y = waypoints.lookahead_point_car(1);
    double angle = K_p * 2.1 * y / pow(r, 2);
    return angle;
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node_ptr = std::make_shared<PurePursuit>();
    rclcpp::spin(node_ptr);
    rclcpp::shutdown();
    return 0;
}

