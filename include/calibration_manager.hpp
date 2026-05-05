#pragma once

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <string>
#include <filesystem>
#include <memory>

namespace BikeGuard {

// Calibration configuration structure
struct CalibrationConfig {
    double exclusion_zone_percentage = 0.15;  // Default 15% from bottom
    int exclusion_y_coordinate = 0;            // Calculated Y coordinate
    bool is_configured = false;
    
    // Camera information
    int camera_width = 640;
    int camera_height = 480;
    
    // Calibration UI settings
    std::string window_name = "BikeGuard - Calibration Mode";
    cv::Scalar exclusion_color = cv::Scalar(0, 0, 255, 128);  // Semi-transparent red
    cv::Scalar line_color = cv::Scalar(0, 255, 0);           // Green line
    int line_thickness = 2;
    
    // Control settings
    double adjustment_step = 0.01;  // 1% adjustment per key press
    double min_exclusion = 0.05;   // Minimum 5% exclusion
    double max_exclusion = 0.40;   // Maximum 40% exclusion
    
    // File paths
    std::string config_file = "config/calibration_config.json";
};

// Interactive calibration manager for dynamic handlebar/mirror exclusion zone
class CalibrationManager {
public:
    explicit CalibrationManager(const CalibrationConfig& config = {});
    ~CalibrationManager();
    
    // Core functionality
    auto initialize_camera(int camera_id = 0) -> bool;
    auto run_calibration() -> void;
    auto is_configured() const -> bool;
    
    // Configuration methods
    auto load_config() -> bool;
    auto save_config() -> bool;
    auto get_config() const -> CalibrationConfig;
    auto set_config(const CalibrationConfig& config) -> void;
    
    // Exclusion zone methods
    auto get_exclusion_zone() const -> cv::Rect;
    auto apply_exclusion_zone(cv::Mat& frame) const -> void;
    auto get_exclusion_y_coordinate() const -> int;

private:
    // Core calibration loop
    auto calibration_loop() -> void;
    
    // UI and interaction
    auto draw_interface(cv::Mat& frame) -> void;
    auto handle_key_input(int key) -> bool;
    auto update_exclusion_zone() -> void;
    
    // Camera management
    auto capture_frame(cv::Mat& frame) -> bool;
    auto release_camera() -> void;
    
    // Configuration serialization
    auto serialize_config() const -> nlohmann::json;
    auto deserialize_config(const nlohmann::json& json) -> bool;
    
    // Utility methods
    auto calculate_y_coordinate() const -> int;
    auto clamp_exclusion_percentage(double percentage) const -> double;
    auto create_instructions_text() const -> std::string;

    // Member variables
    CalibrationConfig config_;
    cv::VideoCapture camera_;
    bool camera_initialized_{false};
    bool should_exit_{false};
    
    // UI state
    cv::Mat display_frame_;
    cv::Mat overlay_frame_;
    
    // Statistics
    size_t frame_count_{0};
    double average_fps_{0.0};
    std::chrono::steady_clock::time_point last_fps_update_;
};

// Factory function
auto create_calibration_manager(const CalibrationConfig& config = {}) 
    -> std::unique_ptr<CalibrationManager>;

} // namespace BikeGuard
