/*
Pure Pursuit Implementation in C++. Includes features such as dynamic lookahead and two CSV waypoints loading.
*/
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
#include "std_msgs/msg/int8.hpp"
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

#define _USE_MATH_DEFINES
using std::placeholders::_1;
using namespace std::chrono_literals;

class PurePursuit : public rclcpp::Node {
   public:
    PurePursuit();

   private:
    struct csvFileData {
        std::vector<double> X;
        std::vector<double> Y;
        std::vector<double> V;

        int index = 0;
        int velocity_index = 0;

        Eigen::Vector3d lookahead_point_world;  // from world reference frame (usually `map`)
        Eigen::Vector3d lookahead_point_car;    // from car reference frame
        Eigen::Vector3d current_point_world;    // Locks on to the closest waypoint, which gives a velocity profile
    };

    Eigen::Matrix3d rotation_m;

    double x_car_world = 0.0;
    double y_car_world = 0.0;

    std::string odom_topic;
    std::string car_refFrame;
    std::string drive_topic;
    std::string global_refFrame;
    std::string rviz_current_waypoint_topic;
    std::string rviz_lookahead_waypoint_topic;
    std::string waypoints_path;
    std::string second_waypoints_path; // 두 번째 CSV 파일 경로

    double K_p;
    double min_lookahead;
    double max_lookahead;
    double lookahead_ratio;
    double steering_limit;
    double velocity_percentage;
    double curr_velocity;
    double curr_velocity_pure;  // 가속도 제어를 위한 현재 속도 변수
    double acceleration_rate;  // 가속도 비율

    // 경로 완료 여부와 데이터 구조체
    bool first_path_completed;
    csvFileData waypoints;
    csvFileData second_waypoints;

    int num_waypoints;
 std::fstream csvFile_waypoints; // csvFile_waypoints 선언
    // Timer 및 구독자 초기화
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr subscription_odom;
    rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr trigger_subscription;

    // 퍼블리셔 초기화
    rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr publisher_drive;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vis_current_point_pub;
    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr vis_lookahead_point_pub;

    // tf 및 Deadman 스위치 변수
    std::shared_ptr<tf2_ros::TransformListener> transform_listener_{nullptr};
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    static int is_starting_value; // 정적 변수는 cpp 파일에서 초기화해야 함

    // private functions
    double to_radians(double degrees);
    double to_degrees(double radians);
    double p2pdist(double &x1, double &x2, double &y1, double &y2);

    void trigger_callback(const std_msgs::msg::Int8::SharedPtr msg);  // Deadman 스위치 콜백 함수
    bool is_autonomous_enabled;  // 자율주행 모드 상태 플래그

    void load_waypoints();               // 첫 번째 경로 로드
    void load_second_waypoints();        // 두 번째 경로 로드
    void visualize_lookahead_point(Eigen::Vector3d &point);
    void visualize_current_point(Eigen::Vector3d &point);

    void get_waypoint();
    void quat_to_rot(double q0, double q1, double q2, double q3);
    void transformandinterp_waypoint();

    double p_controller();
    double get_velocity(double steering_angle);
    void publish_message(double steering_angle);

    void odom_callback(const nav_msgs::msg::Odometry::ConstSharedPtr odom_submsgObj);
    void timer_callback();
};
