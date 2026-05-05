#include "calibration_manager.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>

namespace BikeGuard {

CalibrationManager::CalibrationManager(const CalibrationConfig& config) 
    : config_(config), last_fps_update_(std::chrono::steady_clock::now()) {
    
    // Ensure config directory exists
    std::filesystem::create_directories("config");
}

CalibrationManager::~CalibrationManager() {
    release_camera();
}

auto CalibrationManager::initialize_camera(int camera_id) -> bool {
    if (camera_.isOpened()) {
        camera_.release();
    }
    
    camera_.open(camera_id, cv::CAP_MSMF);
    if (!camera_.isOpened()) {
        std::cerr << "Failed to open camera with ID: " << camera_id << '\n';
        return false;
    }
    
    // Get camera properties
    config_.camera_width = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_WIDTH));
    config_.camera_height = static_cast<int>(camera_.get(cv::CAP_PROP_FRAME_HEIGHT));
    
    // Set camera properties for optimal performance
    camera_.set(cv::CAP_PROP_FPS, 30);
    camera_.set(cv::CAP_PROP_BUFFERSIZE, 1);  // Minimize latency
    
    // Calculate initial Y coordinate
    update_exclusion_zone();
    
    camera_initialized_ = true;
    std::cout << "Camera initialized: " << config_.camera_width << "x" 
              << config_.camera_height << '\n';
    
    return true;
}

auto CalibrationManager::run_calibration() -> void {
    if (!camera_initialized_) {
        std::cerr << "Camera not initialized. Call initialize_camera() first.\n";
        return;
    }
    
    std::cout << "Starting interactive calibration...\n";
    std::cout << create_instructions_text() << '\n';
    
    calibration_loop();
}

auto CalibrationManager::is_configured() const -> bool {
    return config_.is_configured;
}

auto CalibrationManager::load_config() -> bool {
    try {
        std::ifstream file(config_.config_file);
        if (!file.is_open()) {
            std::cout << "No calibration config found, using defaults.\n";
            return false;
        }
        
        nlohmann::json json;
        file >> json;
        
        return deserialize_config(json);
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load calibration config: " << e.what() << '\n';
        return false;
    }
}

auto CalibrationManager::save_config() -> bool {
    try {
        nlohmann::json json = serialize_config();
        
        std::ofstream file(config_.config_file);
        if (!file.is_open()) {
            std::cerr << "Failed to create calibration config file.\n";
            return false;
        }
        
        file << json.dump(4);  // Pretty print with 4 spaces
        file.close();
        
        config_.is_configured = true;
        std::cout << "Calibration saved to: " << config_.config_file << '\n';
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to save calibration config: " << e.what() << '\n';
        return false;
    }
}

auto CalibrationManager::get_config() const -> CalibrationConfig {
    return config_;
}

auto CalibrationManager::set_config(const CalibrationConfig& config) -> void {
    config_ = config;
    update_exclusion_zone();
}

auto CalibrationManager::get_exclusion_zone() const -> cv::Rect {
    int y_start = calculate_y_coordinate();
    return cv::Rect(0, y_start, config_.camera_width, 
                   config_.camera_height - y_start);
}

auto CalibrationManager::apply_exclusion_zone(cv::Mat& frame) const -> void {
    if (!config_.is_configured) {
        return;
    }
    
    cv::Rect exclusion = get_exclusion_zone();
    cv::Mat mask = cv::Mat::zeros(frame.size(), frame.type());
    cv::rectangle(mask, exclusion, cv::Scalar(255), cv::FILLED);
    
    // Apply exclusion zone (set to black or blur)
    cv::bitwise_and(frame, cv::Scalar(0), mask, frame);
}

auto CalibrationManager::get_exclusion_y_coordinate() const -> int {
    return calculate_y_coordinate();
}

auto CalibrationManager::calibration_loop() -> void {
    should_exit_ = false;
    
    while (!should_exit_) {
        cv::Mat frame;
        if (!capture_frame(frame)) {
            std::cerr << "Failed to capture frame\n";
            break;
        }
        
        // Create display frame with overlay
        display_frame_ = frame.clone();
        draw_interface(display_frame_);
        
        // Show the calibration interface
        cv::imshow(config_.window_name, display_frame_);
        
        // Handle keyboard input
        int key = cv::waitKey(1) & 0xFF;
        if (handle_key_input(key)) {
            break;
        }
        
        // Update FPS counter
        frame_count_++;
        auto now = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration<double>(now - last_fps_update_).count();
        
        if (duration >= 1.0) {
            average_fps_ = frame_count_ / duration;
            frame_count_ = 0;
            last_fps_update_ = now;
        }
    }
    
    cv::destroyWindow(config_.window_name);
}

auto CalibrationManager::draw_interface(cv::Mat& frame) -> void {
    // Draw exclusion zone overlay
    cv::Rect exclusion = get_exclusion_zone();
    
    // Create semi-transparent overlay
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, exclusion, config_.exclusion_color, cv::FILLED);
    
    // Blend with original frame
    cv::addWeighted(overlay, 0.3, frame, 0.7, 0, frame);
    
    // Draw exclusion line (green)
    cv::line(frame, cv::Point(0, exclusion.y), 
             cv::Point(config_.camera_width, exclusion.y), 
             config_.line_color, config_.line_thickness);
    
    // Draw UI text
    cv::putText(frame, "Calibration Mode", cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);
    
    cv::putText(frame, "Exclusion: " + std::to_string(static_cast<int>(config_.exclusion_zone_percentage * 100)) + "%", 
                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
    cv::putText(frame, "FPS: " + std::to_string(static_cast<int>(average_fps_)), 
                cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
    // Draw instructions
    std::string instructions = "W/↑: Up | S/↓: Down | Enter: Save & Exit | ESC: Cancel";
    cv::putText(frame, instructions, cv::Point(10, frame.rows - 20), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
}

auto CalibrationManager::handle_key_input(int key) -> bool {
    switch (key) {
        case 'w':
        case 'W':
        case cv::K_UP:  // Up arrow
            config_.exclusion_zone_percentage = clamp_exclusion_percentage(
                config_.exclusion_zone_percentage - config_.adjustment_step);
            update_exclusion_zone();
            break;
            
        case 's':
        case 'S':
        case cv::K_DOWN:  // Down arrow
            config_.exclusion_zone_percentage = clamp_exclusion_percentage(
                config_.exclusion_zone_percentage + config_.adjustment_step);
            update_exclusion_zone();
            break;
            
        case 13:  // Enter key
            std::cout << "Saving calibration configuration...\n";
            if (save_config()) {
                std::cout << "Calibration saved successfully!\n";
            }
            should_exit_ = true;
            return true;
            
        case 27:  // ESC key
            std::cout << "Calibration cancelled.\n";
            should_exit_ = true;
            return true;
    }
    
    return false;
}

auto CalibrationManager::update_exclusion_zone() -> void {
    config_.exclusion_y_coordinate = calculate_y_coordinate();
}

auto CalibrationManager::capture_frame(cv::Mat& frame) -> void {
    if (!camera_.read(frame)) {
        throw std::runtime_error("Failed to capture frame from camera");
    }
    
    if (frame.empty()) {
        throw std::runtime_error("Captured frame is empty");
    }
}

auto CalibrationManager::release_camera() -> void {
    if (camera_.isOpened()) {
        camera_.release();
        camera_initialized_ = false;
    }
}

auto CalibrationManager::serialize_config() const -> nlohmann::json {
    nlohmann::json json;
    json["exclusion_zone_percentage"] = config_.exclusion_zone_percentage;
    json["exclusion_y_coordinate"] = config_.exclusion_y_coordinate;
    json["is_configured"] = config_.is_configured;
    json["camera_width"] = config_.camera_width;
    json["camera_height"] = config_.camera_height;
    json["adjustment_step"] = config_.adjustment_step;
    json["min_exclusion"] = config_.min_exclusion;
    json["max_exclusion"] = config_.max_exclusion;
    json["timestamp"] = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    return json;
}

auto CalibrationManager::deserialize_config(const nlohmann::json& json) -> bool {
    try {
        if (json.contains("exclusion_zone_percentage")) {
            config_.exclusion_zone_percentage = json["exclusion_zone_percentage"];
        }
        
        if (json.contains("exclusion_y_coordinate")) {
            config_.exclusion_y_coordinate = json["exclusion_y_coordinate"];
        }
        
        if (json.contains("is_configured")) {
            config_.is_configured = json["is_configured"];
        }
        
        if (json.contains("camera_width")) {
            config_.camera_width = json["camera_width"];
        }
        
        if (json.contains("camera_height")) {
            config_.camera_height = json["camera_height"];
        }
        
        if (json.contains("adjustment_step")) {
            config_.adjustment_step = json["adjustment_step"];
        }
        
        if (json.contains("min_exclusion")) {
            config_.min_exclusion = json["min_exclusion"];
        }
        
        if (json.contains("max_exclusion")) {
            config_.max_exclusion = json["max_exclusion"];
        }
        
        // Validate and clamp values
        config_.exclusion_zone_percentage = clamp_exclusion_percentage(config_.exclusion_zone_percentage);
        update_exclusion_zone();
        
        std::cout << "Calibration config loaded successfully.\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing calibration config: " << e.what() << '\n';
        return false;
    }
}

auto CalibrationManager::calculate_y_coordinate() const -> int {
    return static_cast<int>((1.0 - config_.exclusion_zone_percentage) * config_.camera_height);
}

auto CalibrationManager::clamp_exclusion_percentage(double percentage) const -> double {
    return std::clamp(percentage, config_.min_exclusion, config_.max_exclusion);
}

auto CalibrationManager::create_instructions_text() const -> std::string {
    std::ostringstream oss;
    oss << "=== BikeGuard Calibration Instructions ===\n";
    oss << "Adjust the green line to set the exclusion zone boundary.\n";
    oss << "Everything below the green line will be excluded from detection.\n";
    oss << "\nControls:\n";
    oss << "  W/↑ : Move exclusion line UP (reduce exclusion zone)\n";
    oss << "  S/↓ : Move exclusion line DOWN (increase exclusion zone)\n";
    oss << "  Enter : Save configuration and exit\n";
    oss << "  ESC  : Cancel calibration\n";
    oss << "\nCurrent exclusion zone: " << static_cast<int>(config_.exclusion_zone_percentage * 100) << "%";
    
    return oss.str();
}

// Factory function implementation
auto create_calibration_manager(const CalibrationConfig& config) 
    -> std::unique_ptr<CalibrationManager> {
    
    return std::make_unique<CalibrationManager>(config);
}

} // namespace BikeGuard
