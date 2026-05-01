#include "bikeguard_road_engine.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <iomanip>

namespace BikeGuard {

// Royal Enfield Classic 350 Calibration Implementation
class BaronCalibrationEngine {
public:
    struct CalibrationResult {
        bool success = false;
        BaronProfile profile;
        std::string error_message;
        float calibration_confidence = 0.0f;
        cv::Mat calibration_overlay;
    };
    
    struct DetectionPoint {
        cv::Point2f position;
        float confidence;
        std::string description;
    };
    
    auto initialize() -> bool {
        try {
            // Initialize calibration parameters for Royal Enfield Classic 350
            default_profile_.enabled = true;
            default_profile_.handlebar_exclusion_zone = 0.15f;
            default_profile_.auto_calibrate = true;
            
            // Set default geometry for Royal Enfield Classic 350
            default_profile_.geometry.handlebar_height_ratio = 0.65f;
            default_profile_.geometry.mirror_width_ratio = 0.12f;
            default_profile_.geometry.mirror_height_ratio = 0.08f;
            default_profile_.geometry.mirror_left_x_ratio = 0.15f;
            default_profile_.geometry.mirror_right_x_ratio = 0.75f;
            default_profile_.geometry.mirror_y_ratio = 0.25f;
            
            initialized_ = true;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Baron calibration engine initialization failed: {}\n", e.what());
            return false;
        }
    }
    
    auto calibrate_from_frame(const cv::Mat& frame) -> CalibrationResult {
        CalibrationResult result;
        
        if (!initialized_) {
            result.error_message = "Calibration engine not initialized";
            return result;
        }
        
        try {
            // Convert to grayscale for feature detection
            cv::Mat gray_frame;
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
            
            // Detect motorcycle features
            auto detection_points = detect_royal_enfield_features(gray_frame);
            
            if (detection_points.empty()) {
                result.error_message = "No Royal Enfield features detected";
                return result;
            }
            
            // Calculate calibration profile
            result.profile = calculate_calibration_profile(detection_points, frame.size());
            result.success = true;
            result.calibration_confidence = calculate_calibration_confidence(detection_points);
            
            // Create calibration overlay
            result.calibration_overlay = create_calibration_overlay(frame, result.profile);
            
            // Save calibration profile
            save_calibration_profile(result.profile);
            
            return result;
            
        } catch (const std::exception& e) {
            result.error_message = std::format("Calibration failed: {}", e.what());
            return result;
        }
    }
    
    auto load_calibration_profile(const std::string& profile_path) -> BaronProfile {
        BaronProfile profile = default_profile_;
        
        try {
            std::ifstream file(profile_path);
            if (!file.is_open()) {
                std::cerr << std::format("Failed to open calibration profile: {}\n", profile_path);
                return profile;
            }
            
            // Load profile parameters
            std::string line;
            while (std::getline(file, line)) {
                if (line.find("handlebar_exclusion_zone") != std::string::npos) {
                    profile.handlebar_exclusion_zone = std::stof(line.substr(line.find('=') + 1));
                } else if (line.find("handlebar_height_ratio") != std::string::npos) {
                    profile.geometry.handlebar_height_ratio = std::stof(line.substr(line.find('=') + 1));
                } else if (line.find("mirror_left_x_ratio") != std::string::npos) {
                    profile.geometry.mirror_left_x_ratio = std::stof(line.substr(line.find('=') + 1));
                } else if (line.find("mirror_right_x_ratio") != std::string::npos) {
                    profile.geometry.mirror_right_x_ratio = std::stof(line.substr(line.find('=') + 1));
                }
            }
            
            profile.enabled = true;
            return profile;
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Failed to load calibration profile: {}\n", e.what());
            return profile;
        }
    }
    
    auto apply_calibration(cv::Mat& frame, const BaronProfile& profile) -> cv::Mat {
        if (!profile.enabled) {
            return frame;
        }
        
        try {
            cv::Mat calibrated_frame = frame.clone();
            
            // Apply handlebar exclusion zone
            apply_handlebar_exclusion(calibrated_frame, profile);
            
            // Apply mirror exclusion zones
            apply_mirror_exclusion(calibrated_frame, profile);
            
            // Apply computational optimization zones
            apply_optimization_zones(calibrated_frame, profile);
            
            return calibrated_frame;
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Calibration application failed: {}\n", e.what());
            return frame;
        }
    }

private:
    bool initialized_ = false;
    BaronProfile default_profile_;
    
    auto detect_royal_enfield_features(const cv::Mat& gray_frame) -> std::vector<DetectionPoint> {
        std::vector<DetectionPoint> detection_points;
        
        try {
            // Detect handlebar region using template matching
            cv::Mat handlebar_template = create_handlebar_template(gray_frame.size());
            cv::Mat result;
            cv::matchTemplate(gray_frame, handlebar_template, result, cv::TM_CCOEFF_NORMED);
            
            double min_val, max_val;
            cv::Point min_loc, max_loc;
            cv::minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
            
            if (max_val > 0.6f) {
                detection_points.push_back({
                    cv::Point2f(max_loc.x + handlebar_template.cols / 2.0f, 
                              max_loc.y + handlebar_template.rows / 2.0f),
                    max_val,
                    "Handlebar"
                });
            }
            
            // Detect mirrors using circular feature detection
            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(gray_frame, circles, cv::HOUGH_GRADIENT, 1, 50, 100, 30, 10, 30);
            
            for (const auto& circle : circles) {
                cv::Point center(cvRound(circle[0]), cvRound(circle[1]));
                int radius = cvRound(circle[2]);
                
                // Check if circle is in expected mirror position
                float x_ratio = static_cast<float>(center.x) / gray_frame.cols;
                float y_ratio = static_cast<float>(center.y) / gray_frame.rows;
                
                if ((x_ratio < 0.3f || x_ratio > 0.7f) && y_ratio < 0.5f) {
                    std::string mirror_type = x_ratio < 0.5f ? "Left Mirror" : "Right Mirror";
                    detection_points.push_back({
                        cv::Point2f(center.x, center.y),
                        circle[2] / 30.0f, // Normalize confidence
                        mirror_type
                    });
                }
            }
            
            // Detect motorcycle body using edge detection
            cv::Mat edges;
            cv::Canny(gray_frame, edges, 50, 150);
            
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            
            for (const auto& contour : contours) {
                double area = cv::contourArea(contour);
                if (area > gray_frame.rows * gray_frame.cols * 0.1f) { // Large contour
                    cv::Rect bounding_rect = cv::boundingRect(contour);
                    detection_points.push_back({
                        cv::Point2f(bounding_rect.x + bounding_rect.width / 2.0f,
                                  bounding_rect.y + bounding_rect.height / 2.0f),
                        std::min(area / (gray_frame.rows * gray_frame.cols), 1.0f),
                        "Motorcycle Body"
                    });
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Royal Enfield feature detection failed: {}\n", e.what());
        }
        
        return detection_points;
    }
    
    auto create_handlebar_template(const cv::Size& frame_size) -> cv::Mat {
        // Create a synthetic handlebar template
        cv::Mat template_img = cv::Mat::zeros(frame_size.height / 4, frame_size.width / 2, CV_8UC1);
        
        // Draw handlebar shape (simplified)
        cv::Point center(template_img.cols / 2, template_img.rows / 2);
        cv::ellipse(template_img, center, cv::Size(template_img.cols / 3, template_img.rows / 6), 
                   0, 0, 360, cv::Scalar(255), -1);
        
        return template_img;
    }
    
    auto calculate_calibration_profile(const std::vector<DetectionPoint>& detection_points, 
                                    const cv::Size& frame_size) -> BaronProfile {
        BaronProfile profile = default_profile_;
        
        try {
            // Find handlebar detection
            auto handlebar_it = std::find_if(detection_points.begin(), detection_points.end(),
                [](const DetectionPoint& point) { return point.description == "Handlebar"; });
            
            if (handlebar_it != detection_points.end()) {
                // Calculate handlebar exclusion zone based on detected position
                profile.geometry.handlebar_height_ratio = 
                    static_cast<float>(handlebar_it->position.y) / frame_size.height;
                profile.handlebar_exclusion_zone = 1.0f - profile.geometry.handlebar_height_ratio;
            }
            
            // Find mirror detections
            auto left_mirror_it = std::find_if(detection_points.begin(), detection_points.end(),
                [](const DetectionPoint& point) { return point.description == "Left Mirror"; });
            auto right_mirror_it = std::find_if(detection_points.begin(), detection_points.end(),
                [](const DetectionPoint& point) { return point.description == "Right Mirror"; });
            
            if (left_mirror_it != detection_points.end()) {
                profile.geometry.mirror_left_x_ratio = 
                    static_cast<float>(left_mirror_it->position.x) / frame_size.width;
                profile.geometry.mirror_y_ratio = 
                    static_cast<float>(left_mirror_it->position.y) / frame_size.height;
            }
            
            if (right_mirror_it != detection_points.end()) {
                profile.geometry.mirror_right_x_ratio = 
                    static_cast<float>(right_mirror_it->position.x) / frame_size.width;
            }
            
            // Calculate mirror exclusion zones
            int mirror_width = static_cast<int>(frame_size.width * profile.geometry.mirror_width_ratio);
            int mirror_height = static_cast<int>(frame_size.height * profile.geometry.mirror_height_ratio);
            
            // Left mirror zone
            profile.mirror_exclusion_zones[0] = cv::Rect(
                static_cast<int>(frame_size.width * profile.geometry.mirror_left_x_ratio) - mirror_width / 2,
                static_cast<int>(frame_size.height * profile.geometry.mirror_y_ratio) - mirror_height / 2,
                mirror_width, mirror_height
            );
            
            // Right mirror zone
            profile.mirror_exclusion_zones[1] = cv::Rect(
                static_cast<int>(frame_size.width * profile.geometry.mirror_right_x_ratio) - mirror_width / 2,
                static_cast<int>(frame_size.height * profile.geometry.mirror_y_ratio) - mirror_height / 2,
                mirror_width, mirror_height
            );
            
            // Calculate handlebar ROI
            profile.handlebar_roi = cv::Rect(
                0,
                static_cast<int>(frame_size.height * profile.geometry.handlebar_height_ratio),
                frame_size.width,
                static_cast<int>(frame_size.height * profile.handlebar_exclusion_zone)
            );
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Calibration profile calculation failed: {}\n", e.what());
        }
        
        return profile;
    }
    
    auto calculate_calibration_confidence(const std::vector<DetectionPoint>& detection_points) -> float {
        if (detection_points.empty()) {
            return 0.0f;
        }
        
        float total_confidence = 0.0f;
        for (const auto& point : detection_points) {
            total_confidence += point.confidence;
        }
        
        return total_confidence / detection_points.size();
    }
    
    auto create_calibration_overlay(const cv::Mat& frame, const BaronProfile& profile) -> cv::Mat {
        cv::Mat overlay = frame.clone();
        
        try {
            // Draw handlebar exclusion zone
            cv::Rect handlebar_zone = profile.handlebar_roi;
            cv::rectangle(overlay, handlebar_zone, cv::Scalar(0, 0, 255), 2);
            cv::putText(overlay, "Handlebar Exclusion", 
                       cv::Point(handlebar_zone.x, handlebar_zone.y - 10),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
            
            // Draw mirror exclusion zones
            for (int i = 0; i < 2; ++i) {
                cv::Rect mirror_zone = profile.mirror_exclusion_zones[i];
                if (mirror_area > 0) {
                    cv::rectangle(overlay, mirror_zone, cv::Scalar(255, 0, 0), 2);
                    std::string label = i == 0 ? "Left Mirror" : "Right Mirror";
                    cv::putText(overlay, label,
                               cv::Point(mirror_zone.x, mirror_zone.y - 10),
                               cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
                }
            }
            
            // Add calibration info
            std::string info_text = std::format("Baron Calibration Active - Exclusion: {:.1f}%", 
                                             profile.handlebar_exclusion_zone * 100.0f);
            cv::putText(overlay, info_text, cv::Point(10, 30),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Calibration overlay creation failed: {}\n", e.what());
        }
        
        return overlay;
    }
    
    auto save_calibration_profile(const BaronProfile& profile) -> void {
        try {
            std::ofstream file("baron_calibration.profile");
            if (!file.is_open()) {
                std::cerr << "Failed to save calibration profile\n";
                return;
            }
            
            file << std::fixed << std::setprecision(4);
            file << "# Royal Enfield Classic 350 Calibration Profile\n";
            file << "enabled=" << (profile.enabled ? 1 : 0) << "\n";
            file << "handlebar_exclusion_zone=" << profile.handlebar_exclusion_zone << "\n";
            file << "handlebar_height_ratio=" << profile.geometry.handlebar_height_ratio << "\n";
            file << "mirror_left_x_ratio=" << profile.geometry.mirror_left_x_ratio << "\n";
            file << "mirror_right_x_ratio=" << profile.geometry.mirror_right_x_ratio << "\n";
            file << "mirror_y_ratio=" << profile.geometry.mirror_y_ratio << "\n";
            file << "mirror_width_ratio=" << profile.geometry.mirror_width_ratio << "\n";
            file << "mirror_height_ratio=" << profile.geometry.mirror_height_ratio << "\n";
            
            file.close();
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Failed to save calibration profile: {}\n", e.what());
        }
    }
    
    auto apply_handlebar_exclusion(cv::Mat& frame, const BaronProfile& profile) -> void {
        cv::Rect handlebar_zone = profile.handlebar_roi;
        if (handlebar_zone.area() > 0) {
            cv::Mat handlebar_roi = frame(handlebar_zone);
            handlebar_roi *= 0.2f; // Darken by 80%
        }
    }
    
    auto apply_mirror_exclusion(cv::Mat& frame, const BaronProfile& profile) -> void {
        for (int i = 0; i < 2; ++i) {
            cv::Rect mirror_zone = profile.mirror_exclusion_zones[i];
            if (mirror_zone.area() > 0) {
                cv::Mat mirror_roi = frame(mirror_zone);
                mirror_roi *= 0.3f; // Darken by 70%
            }
        }
    }
    
    auto apply_optimization_zones(cv::Mat& frame, const BaronProfile& profile) -> void {
        // Apply computational optimization by reducing processing in excluded zones
        // This is handled at the inference level, but we can visualize it here
        
        cv::Mat optimization_mask = cv::Mat::zeros(frame.size(), CV_8UC1);
        
        // Create mask for areas to process (inverse of exclusion zones)
        cv::rectangle(optimization_mask, cv::Point(0, 0), frame.size() - cv::Point(1, 1), cv::Scalar(255), -1);
        
        // Subtract handlebar exclusion zone
        cv::rectangle(optimization_mask, profile.handlebar_roi, cv::Scalar(0), -1);
        
        // Subtract mirror exclusion zones
        for (int i = 0; i < 2; ++i) {
            cv::Rect mirror_zone = profile.mirror_exclusion_zones[i];
            if (mirror_zone.area() > 0) {
                cv::rectangle(optimization_mask, mirror_zone, cv::Scalar(0), -1);
            }
        }
        
        // Apply subtle overlay to show optimization zones
        cv::Mat colored_mask;
        cv::applyColorMap(optimization_mask, colored_mask, cv::COLORMAP_JET);
        cv::addWeighted(frame, 0.9, colored_mask, 0.1, 0, frame);
    }
};

// Factory function implementation
auto create_baron_calibration_engine() -> std::unique_ptr<BaronCalibrationEngine> {
    auto engine = std::make_unique<BaronCalibrationEngine>();
    if (engine->initialize()) {
        return engine;
    }
    
    return nullptr;
}

} // namespace BikeGuard
