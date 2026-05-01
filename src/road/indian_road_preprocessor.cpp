#include "bikeguard_road_engine.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>

namespace BikeGuard {

// Indian Road Preprocessor Implementation
auto IndianRoadPreprocessor::initialize(const PreprocessingConfig& config) -> bool {
    try {
        config_ = config;
        
        // Initialize CLAHE for adaptive histogram equalization
        clahe_ = cv::createCLAHE();
        clahe_->setClipLimit(3.0);
        clahe_->setTilesGridSize(cv::Size(8, 8));
        
        // Initialize masks for dust and glare detection
        dust_mask_ = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
        glare_mask_ = cv::Mat::zeros(cv::Size(640, 480), CV_8UC1);
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Failed to initialize Indian road preprocessor: {}\n", e.what());
        return false;
    }
}

auto IndianRoadPreprocessor::preprocess_frame(const cv::Mat& input_frame, const BaronProfile& baron_profile) -> cv::Mat {
    if (!initialized_) {
        return input_frame.clone();
    }
    
    cv::Mat processed_frame = input_frame.clone();
    
    // Apply lighting corrections for Indian road conditions
    if (config_.adaptive_contrast) {
        apply_lighting_correction(processed_frame);
    }
    
    // Apply dust compensation for Indian roads
    if (config_.dust_compensation) {
        apply_dust_compensation(processed_frame);
    }
    
    // Apply glare reduction for bright sunlight
    if (config_.glare_reduction) {
        apply_glare_reduction(processed_frame);
    }
    
    // Exclude handlebar region if Baron profile is enabled
    if (baron_profile.enabled) {
        processed_frame = exclude_handlebar_region(processed_frame, baron_profile);
    }
    
    return processed_frame;
}

auto IndianRoadPreprocessor::apply_lighting_correction(cv::Mat& frame) -> void {
    try {
        // Convert to LAB color space for better lighting correction
        cv::Mat lab_frame;
        cv::cvtColor(frame, lab_frame, cv::COLOR_BGR2LAB);
        
        // Split channels
        std::vector<cv::Mat> lab_channels;
        cv::split(lab_frame, lab_channels);
        
        // Apply CLAHE to L channel (lightness)
        clahe_->apply(lab_channels[0], lab_channels[0]);
        
        // Merge channels back
        cv::merge(lab_channels, lab_frame);
        
        // Convert back to BGR
        cv::cvtColor(lab_frame, frame, cv::COLOR_LAB2BGR);
        
        // Additional brightness/contrast adjustment based on lighting conditions
        cv::Scalar mean_brightness = cv::mean(frame);
        float avg_brightness = (mean_brightness[0] + mean_brightness[1] + mean_brightness[2]) / 3.0f;
        
        if (avg_brightness < config_.low_light_threshold) {
            // Low light conditions - increase brightness
            frame.convertTo(frame, -1, 1.2, 20);
        } else if (avg_brightness > config_.high_light_threshold) {
            // Bright conditions - reduce brightness and increase contrast
            frame.convertTo(frame, -1, 1.1, -10);
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Lighting correction failed: {}\n", e.what());
    }
}

auto IndianRoadPreprocessor::apply_dust_compensation(cv::Mat& frame) -> void {
    try {
        // Convert to HSV for better dust detection
        cv::Mat hsv_frame;
        cv::cvtColor(frame, hsv_frame, cv::COLOR_BGR2HSV);
        
        // Create dust mask based on saturation and value
        cv::Mat dust_mask;
        cv::inRange(hsv_frame, 
                   cv::Scalar(0, 0, 100),    // Low saturation, low value (dust)
                   cv::Scalar(180, 50, 255), // High saturation, high value
                   dust_mask);
        
        // Apply morphological operations to clean mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(dust_mask, dust_mask, cv::MORPH_CLOSE, kernel);
        
        // Apply dust compensation using inpainting
        if (cv::countNonZero(dust_mask) > 0) {
            cv::inpaint(frame, dust_mask, frame, 3, cv::INPAINT_TELEA);
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Dust compensation failed: {}\n", e.what());
    }
}

auto IndianRoadPreprocessor::apply_glare_reduction(cv::Mat& frame) -> void {
    try {
        // Detect glare regions (bright spots)
        cv::Mat gray_frame;
        cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);
        
        // Threshold for bright regions
        cv::Mat glare_mask;
        cv::threshold(gray_frame, glare_mask, 200, 255, cv::THRESH_BINARY);
        
        // Apply morphological operations to clean mask
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(glare_mask, glare_mask, cv::MORPH_CLOSE, kernel);
        
        // Reduce glare by blending with surrounding pixels
        if (cv::countNonZero(glare_mask) > 0) {
            cv::Mat blurred_frame;
            cv::GaussianBlur(frame, blurred_frame, cv::Size(15, 15), 0);
            
            // Blend original and blurred frame based on glare mask
            cv::Mat mask_float;
            glare_mask.convertTo(mask_float, CV_32F, 1.0/255.0);
            
            for (int i = 0; i < 3; ++i) {
                cv::Mat original_channel = frame.col(i);
                cv::Mat blurred_channel = blurred_frame.col(i);
                cv::Mat result_channel = frame.col(i);
                
                for (int y = 0; y < frame.rows; ++y) {
                    for (int x = 0; x < frame.cols; ++x) {
                        if (glare_mask.at<uchar>(y, x) > 0) {
                            float blend_factor = mask_float.at<float>(y, x);
                            result_channel.at<uchar>(y, x) = 
                                static_cast<uchar>((1.0f - blend_factor) * original_channel.at<uchar>(y, x) + 
                                                 blend_factor * blurred_channel.at<uchar>(y, x));
                        }
                    }
                }
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Glare reduction failed: {}\n", e.what());
    }
}

auto IndianRoadPreprocessor::exclude_handlebar_region(const cv::Mat& frame, const BaronProfile& profile) -> void {
    try {
        if (!profile.enabled) {
            return;
        }
        
        // Create exclusion mask for handlebar region (lower 15% of frame)
        int exclusion_height = static_cast<int>(frame.rows * profile.handlebar_exclusion_zone);
        cv::Rect handlebar_region(0, frame.rows - exclusion_height, frame.cols, exclusion_height);
        
        // Darken the handlebar region to reduce false detections
        cv::Mat handlebar_roi = frame(handlebar_region);
        handlebar_roi *= 0.3f; // Reduce brightness by 70%
        
        // Add exclusion zones for mirrors
        for (int i = 0; i < 2; ++i) {
            cv::Rect mirror_zone = profile.mirror_exclusion_zones[i];
            if (mirror_zone.x >= 0 && mirror_zone.y >= 0 && 
                mirror_zone.x + mirror_zone.width < frame.cols &&
                mirror_zone.y + mirror_zone.height < frame.rows) {
                
                cv::Mat mirror_roi = frame(mirror_zone);
                mirror_roi *= 0.2f; // Reduce brightness by 80%
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Handlebar exclusion failed: {}\n", e.what());
    }
}

// Indian Helmet Classifier Implementation
auto IndianHelmetClassifier::initialize() -> bool {
    try {
        // Initialize typical colors for Sikh turbans
        turban_features_.typical_colors = {
            cv::Scalar(255, 140, 0),   // Orange
            cv::Scalar(255, 69, 0),    // Red-Orange
            cv::Scalar(255, 215, 0),   // Gold
            cv::Scalar(128, 0, 128),   // Purple
            cv::Scalar(255, 255, 255), // White
            cv::Scalar(0, 0, 0)        // Black
        };
        
        // Initialize typical colors for construction helmets
        construction_helmet_features_.typical_colors = {
            cv::Scalar(255, 255, 0),   // Yellow
            cv::Scalar(255, 165, 0),   // Orange
            cv::Scalar(255, 0, 0),     // Red
            cv::Scalar(255, 255, 255), // White
            cv::Scalar(128, 128, 128)  // Gray
        };
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Failed to initialize Indian helmet classifier: {}\n", e.what());
        return false;
    }
}

auto IndianHelmetClassifier::classify_helmet(const cv::Mat& roi, const DetectionResult& detection) -> RoadDetectionResult {
    RoadDetectionResult result(detection.confidence, detection.class_id, detection.bbox, detection.class_name);
    
    if (!initialized_) {
        result.helmet_type = RoadDetectionResult::HelmetType::UNKNOWN;
        result.compliance_status = RoadDetectionResult::ComplianceStatus::UNKNOWN;
        return result;
    }
    
    try {
        // Check for Sikh turban first (exempt category)
        if (classify_sikh_turban(roi)) {
            result.helmet_type = RoadDetectionResult::HelmetType::SIKH_TURBAN;
            result.compliance_status = RoadDetectionResult::ComplianceStatus::EXEMPT;
            return result;
        }
        
        // Check for construction helmet (non-compliant for road)
        if (classify_construction_helmet(roi)) {
            result.helmet_type = RoadDetectionResult::HelmetType::CONSTRUCTION_HELMET;
            result.compliance_status = RoadDetectionResult::ComplianceStatus::NON_COMPLIANT;
            return result;
        }
        
        // Classify standard helmets
        float shape_confidence = calculate_shape_confidence(roi, RoadDetectionResult::HelmetType::STANDARD_FULL_FACE);
        if (shape_confidence > 0.6f) {
            result.helmet_type = RoadDetectionResult::HelmetType::STANDARD_FULL_FACE;
            result.compliance_status = RoadDetectionResult::ComplianceStatus::COMPLIANT;
        } else {
            shape_confidence = calculate_shape_confidence(roi, RoadDetectionResult::HelmetType::STANDARD_HALF_FACE);
            if (shape_confidence > 0.6f) {
                result.helmet_type = RoadDetectionResult::HelmetType::STANDARD_HALF_FACE;
                result.compliance_status = RoadDetectionResult::ComplianceStatus::COMPLIANT;
            } else {
                result.helmet_type = RoadDetectionResult::HelmetType::NO_HELMET;
                result.compliance_status = RoadDetectionResult::ComplianceStatus::NON_COMPLIANT;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Helmet classification failed: {}\n", e.what());
        result.helmet_type = RoadDetectionResult::HelmetType::UNKNOWN;
        result.compliance_status = RoadDetectionResult::ComplianceStatus::UNKNOWN;
    }
    
    return result;
}

auto IndianHelmetClassifier::classify_sikh_turban(const cv::Mat& roi) -> bool {
    try {
        // Convert to HSV for better color analysis
        cv::Mat hsv_roi;
        cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
        
        // Calculate aspect ratio
        float aspect_ratio = static_cast<float>(roi.cols) / roi.rows;
        if (aspect_ratio < turban_features_.aspect_ratio_min || 
            aspect_ratio > turban_features_.aspect_ratio_max) {
            return false;
        }
        
        // Check color uniformity (turbans typically have uniform color)
        cv::Scalar mean_color, stddev_color;
        cv::meanStdDev(hsv_roi, mean_color, stddev_color);
        
        float color_uniformity = 1.0f - (stddev_color[0] + stddev_color[1] + stddev_color[2]) / (3.0f * 255.0f);
        if (color_uniformity < turban_features_.color_uniformity_threshold) {
            return false;
        }
        
        // Check if color matches typical turban colors
        for (const auto& typical_color : turban_features_.typical_colors) {
            cv::Scalar hsv_typical;
            cv::Mat color_mat(1, 1, CV_8UC3, typical_color);
            cv::cvtColor(color_mat, color_mat, cv::COLOR_BGR2HSV);
            hsv_typical = color_mat.at<cv::Vec3b>(0, 0);
            
            float color_distance = std::sqrt(
                std::pow(mean_color[0] - hsv_typical[0], 2) +
                std::pow(mean_color[1] - hsv_typical[1], 2) +
                std::pow(mean_color[2] - hsv_typical[2], 2)
            );
            
            if (color_distance < 30.0f) { // Threshold for color matching
                return true;
            }
        }
        
        return false;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Sikh turban classification failed: {}\n", e.what());
        return false;
    }
}

auto IndianHelmetClassifier::classify_construction_helmet(const cv::Mat& roi) -> bool {
    try {
        // Convert to HSV for color analysis
        cv::Mat hsv_roi;
        cv::cvtColor(roi, hsv_roi, cv::COLOR_BGR2HSV);
        
        // Calculate aspect ratio (construction helmets are typically more rounded)
        float aspect_ratio = static_cast<float>(roi.cols) / roi.rows;
        if (aspect_ratio < construction_helmet_features_.aspect_ratio_min || 
            aspect_ratio > construction_helmet_features_.aspect_ratio_max) {
            return false;
        }
        
        // Check for rib patterns (common in construction helmets)
        cv::Mat gray_roi;
        cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
        
        // Apply edge detection to find rib patterns
        cv::Mat edges;
        cv::Canny(gray_roi, edges, 50, 150);
        
        // Look for horizontal lines (rib patterns)
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI/180, 50, 50, 10);
        
        int horizontal_lines = 0;
        for (const auto& line : lines) {
            float angle = std::atan2(line[3] - line[1], line[2] - line[0]) * 180.0f / CV_PI;
            if (std::abs(angle) < 15.0f || std::abs(angle - 180.0f) < 15.0f) {
                horizontal_lines++;
            }
        }
        
        construction_helmet_features_.has_rib_pattern = horizontal_lines > 2;
        
        // Check if color matches typical construction helmet colors
        cv::Scalar mean_color = cv::mean(hsv_roi);
        for (const auto& typical_color : construction_helmet_features_.typical_colors) {
            cv::Scalar hsv_typical;
            cv::Mat color_mat(1, 1, CV_8UC3, typical_color);
            cv::cvtColor(color_mat, color_mat, cv::COLOR_BGR2HSV);
            hsv_typical = color_mat.at<cv::Vec3b>(0, 0);
            
            float color_distance = std::sqrt(
                std::pow(mean_color[0] - hsv_typical[0], 2) +
                std::pow(mean_color[1] - hsv_typical[1], 2) +
                std::pow(mean_color[2] - hsv_typical[2], 2)
            );
            
            if (color_distance < 40.0f) { // Threshold for color matching
                return true;
            }
        }
        
        return false;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Construction helmet classification failed: {}\n", e.what());
        return false;
    }
}

auto IndianHelmetClassifier::calculate_shape_confidence(const cv::Mat& roi, RoadDetectionResult::HelmetType type) -> float {
    try {
        // Convert to grayscale for shape analysis
        cv::Mat gray_roi;
        cv::cvtColor(roi, gray_roi, cv::COLOR_BGR2GRAY);
        
        // Apply threshold to get binary image
        cv::Mat binary_roi;
        cv::threshold(gray_roi, binary_roi, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        
        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_roi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        
        if (contours.empty()) {
            return 0.0f;
        }
        
        // Get the largest contour (assumed to be the helmet)
        auto largest_contour = std::max_element(contours.begin(), contours.end(),
            [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                return cv::contourArea(a) < cv::contourArea(b);
            });
        
        // Calculate shape features
        double area = cv::contourArea(*largest_contour);
        cv::Rect bounding_rect = cv::boundingRect(*largest_contour);
        double perimeter = cv::arcLength(*largest_contour, true);
        
        // Calculate circularity
        double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        
        // Calculate aspect ratio
        double aspect_ratio = static_cast<double>(bounding_rect.width) / bounding_rect.height;
        
        // Calculate solidity
        std::vector<cv::Point> hull;
        cv::convexHull(*largest_contour, hull);
        double hull_area = cv::contourArea(hull);
        double solidity = area / hull_area;
        
        // Calculate confidence based on helmet type
        float confidence = 0.0f;
        
        switch (type) {
            case RoadDetectionResult::HelmetType::STANDARD_FULL_FACE:
                // Full-face helmets: high circularity, aspect ratio ~1.0
                confidence = static_cast<float>((circularity * 0.4f) + 
                                               (1.0f - std::abs(aspect_ratio - 1.0f)) * 0.3f + 
                                               solidity * 0.3f);
                break;
                
            case RoadDetectionResult::HelmetType::STANDARD_HALF_FACE:
                // Half-face helmets: moderate circularity, aspect ratio ~0.8-1.2
                confidence = static_cast<float>((circularity * 0.3f) + 
                                               (1.0f - std::abs(aspect_ratio - 1.0f)) * 0.4f + 
                                               solidity * 0.3f);
                break;
                
            default:
                confidence = 0.0f;
                break;
        }
        
        return std::clamp(confidence, 0.0f, 1.0f);
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Shape confidence calculation failed: {}\n", e.what());
        return 0.0f;
    }
}

auto IndianHelmetClassifier::determine_compliance(const RoadDetectionResult& result) -> ComplianceStatus {
    // Compliance logic for Indian road conditions
    switch (result.helmet_type) {
        case RoadDetectionResult::HelmetType::STANDARD_FULL_FACE:
        case RoadDetectionResult::HelmetType::STANDARD_HALF_FACE:
            return RoadDetectionResult::ComplianceStatus::COMPLIANT;
            
        case RoadDetectionResult::HelmetType::SIKH_TURBAN:
            return RoadDetectionResult::ComplianceStatus::EXEMPT;
            
        case RoadDetectionResult::HelmetType::CONSTRUCTION_HELMET:
        case RoadDetectionResult::HelmetType::NO_HELMET:
            return RoadDetectionResult::ComplianceStatus::NON_COMPLIANT;
            
        default:
            return RoadDetectionResult::ComplianceStatus::UNKNOWN;
    }
}

// Factory function implementations
auto create_indian_road_preprocessor() -> std::unique_ptr<IndianRoadPreprocessor> {
    auto preprocessor = std::make_unique<IndianRoadPreprocessor>();
    IndianRoadPreprocessor::PreprocessingConfig config;
    config.adaptive_contrast = true;
    config.dust_compensation = true;
    config.glare_reduction = true;
    config.pillion_detection = true;
    
    if (preprocessor->initialize(config)) {
        return preprocessor;
    }
    
    return nullptr;
}

auto create_indian_helmet_classifier() -> std::unique_ptr<IndianHelmetClassifier> {
    auto classifier = std::make_unique<IndianHelmetClassifier>();
    if (classifier->initialize()) {
        return classifier;
    }
    
    return nullptr;
}

} // namespace BikeGuard
