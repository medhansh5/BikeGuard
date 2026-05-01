#include "bikeguard_road_engine.hpp"
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <iostream>

namespace BikeGuard {

// Pillion Rider Detector Implementation
auto PillionRiderDetector::initialize() -> bool {
    try {
        // Initialize geometry parameters for motorcycle rider detection
        geometry_.max_rider_distance_ratio = 0.3f;  // Maximum distance between riders
        geometry_.pillion_position_x_ratio = 0.6f;  // Typical pillion X position (right side)
        geometry_.pillion_position_y_ratio = 0.4f;  // Typical pillion Y position (slightly higher)
        geometry_.motorcycle_width_ratio = 0.4f;     // Expected motorcycle width relative to frame
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Pillion rider detector initialization failed: {}\n", e.what());
        return false;
    }
}

auto PillionRiderDetector::detect_pillion_riders(std::span<const DetectionResult> person_detections, 
                                               const cv::Size& frame_size) -> std::vector<RoadDetectionResult> {
    std::vector<RoadDetectionResult> pillion_detections;
    
    if (!initialized_ || person_detections.empty()) {
        return pillion_detections;
    }
    
    try {
        // Filter person detections to find potential riders
        std::vector<DetectionResult> rider_candidates;
        for (const auto& detection : person_detections) {
            // Check if detection is a person (class_id typically 0 for person in YOLO models)
            if (detection.class_id == 0 && detection.confidence > 0.5f) {
                rider_candidates.push_back(detection);
            }
        }
        
        if (rider_candidates.size() < 2) {
            return pillion_detections; // Need at least 2 people for driver + pillion
        }
        
        // Analyze rider positions to identify driver and pillion
        auto rider_analysis = analyze_rider_positions(rider_candidates);
        
        // Convert to RoadDetectionResult with pillion-specific information
        for (const auto& analysis : rider_analysis) {
            RoadDetectionResult road_result(
                analysis.confidence, 
                analysis.class_id, 
                analysis.bbox, 
                analysis.class_name
            );
            
            road_result.rider_type = analysis.rider_type;
            road_result.compliance_status = check_pillion_compliance(road_result);
            
            pillion_detections.push_back(road_result);
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Pillion rider detection failed: {}\n", e.what());
    }
    
    return pillion_detections;
}

auto PillionRiderDetector::analyze_rider_positions(std::span<const DetectionResult> detections) -> std::vector<RoadDetectionResult> {
    std::vector<RoadDetectionResult> analyzed_detections;
    
    try {
        if (detections.empty()) {
            return analyzed_detections;
        }
        
        // Calculate center points for each detection
        std::vector<std::pair<cv::Point2f, size_t>> rider_centers;
        for (size_t i = 0; i < detections.size(); ++i) {
            cv::Point2f center(
                detections[i].bbox.x + detections[i].bbox.width / 2.0f,
                detections[i].bbox.y + detections[i].bbox.height / 2.0f
            );
            rider_centers.emplace_back(center, i);
        }
        
        // Sort riders by X position (left to right)
        std::sort(rider_centers.begin(), rider_centers.end(),
                  [](const auto& a, const auto& b) { return a.first.x < b.first.x; });
        
        // Analyze each rider's position to determine type
        for (size_t i = 0; i < rider_centers.size(); ++i) {
            const auto& [center, original_index] = rider_centers[i];
            const auto& detection = detections[original_index];
            
            RoadDetectionResult analyzed_result(
                detection.confidence,
                detection.class_id,
                detection.bbox,
                detection.class_name
            );
            
            // Calculate normalized position
            float normalized_x = center.x / 640.0f;  // Assuming 640px width, normalize to [0,1]
            float normalized_y = center.y / 480.0f;  // Assuming 480px height, normalize to [0,1]
            
            // Determine rider type based on position
            if (i == 0) {
                // Leftmost rider is typically the driver
                analyzed_result.rider_type = RoadDetectionResult::RiderType::DRIVER;
            } else if (i == 1 && rider_centers.size() >= 2) {
                // Second rider is typically the pillion
                analyzed_result.rider_type = RoadDetectionResult::RiderType::PILLION;
                
                // Validate pillion position
                if (is_valid_pillion_position(normalized_x, normalized_y)) {
                    analyzed_result.rider_type = RoadDetectionResult::RiderType::PILLION;
                } else {
                    // Might be a different scenario (multiple motorcycles, etc.)
                    analyzed_result.rider_type = RoadDetectionResult::RiderType::UNKNOWN;
                }
            } else {
                // Additional riders (rare case)
                analyzed_result.rider_type = RoadDetectionResult::RiderType::UNKNOWN;
            }
            
            // Calculate rider distance for validation
            if (i > 0) {
                const auto& prev_center = rider_centers[i-1].first;
                float distance = cv::norm(center - prev_center);
                float normalized_distance = distance / 640.0f;  // Normalize by frame width
                
                // Check if riders are too far apart (might be different motorcycles)
                if (normalized_distance > geometry_.max_rider_distance_ratio) {
                    analyzed_result.rider_type = RoadDetectionResult::RiderType::UNKNOWN;
                }
            }
            
            analyzed_detections.push_back(analyzed_result);
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Rider position analysis failed: {}\n", e.what());
    }
    
    return analyzed_detections;
}

auto PillionRiderDetector::check_pillion_compliance(const RoadDetectionResult& pillion) -> bool {
    if (pillion.rider_type != RoadDetectionResult::RiderType::PILLION) {
        return true; // Not a pillion, so no pillion-specific compliance needed
    }
    
    try {
        // Pillion compliance logic for Indian road conditions
        bool is_compliant = true;
        
        // Check if pillion has proper head protection
        if (pillion.helmet_type == RoadDetectionResult::HelmetType::NO_HELMET) {
            is_compliant = false;
        } else if (pillion.helmet_type == RoadDetectionResult::HelmetType::CONSTRUCTION_HELMET) {
            // Construction helmets are non-compliant for road use
            is_compliant = false;
        } else if (pillion.helmet_type == RoadDetectionResult::HelmetType::SIKH_TURBAN) {
            // Sikh turbans are exempt
            is_compliant = true;
        } else if (pillion.helmet_type == RoadDetectionResult::HelmetType::STANDARD_FULL_FACE ||
                   pillion.helmet_type == RoadDetectionResult::HelmetType::STANDARD_HALF_FACE) {
            // Standard helmets are compliant
            is_compliant = true;
        }
        
        // Additional pillion-specific checks
        // Check if pillion is in a safe position (not too far back, etc.)
        if (!is_safe_pillion_position(pillion.bbox)) {
            is_compliant = false;
        }
        
        // Check confidence level - low confidence detections might be unreliable
        if (pillion.confidence < 0.6f) {
            is_compliant = false;
        }
        
        return is_compliant;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Pillion compliance check failed: {}\n", e.what());
        return false;
    }
}

auto PillionRiderDetector::is_valid_pillion_position(float normalized_x, float normalized_y) -> bool {
    try {
        // Check if position is within expected pillion position range
        float x_tolerance = 0.2f;  // 20% tolerance
        float y_tolerance = 0.15f; // 15% tolerance
        
        bool x_valid = std::abs(normalized_x - geometry_.pillion_position_x_ratio) <= x_tolerance;
        bool y_valid = std::abs(normalized_y - geometry_.pillion_position_y_ratio) <= y_tolerance;
        
        return x_valid && y_valid;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Pillion position validation failed: {}\n", e.what());
        return false;
    }
}

auto PillionRiderDetector::is_safe_pillion_position(const cv::Rect& bbox) -> bool {
    try {
        // Check if pillion is in a safe position on the motorcycle
        // This is a simplified check - in practice, you'd want more sophisticated analysis
        
        // Check if bbox is reasonable size (not too small or too large)
        int min_area = 5000;   // Minimum area for a person
        int max_area = 50000;  // Maximum area for a person
        
        int area = bbox.area();
        if (area < min_area || area > max_area) {
            return false;
        }
        
        // Check aspect ratio (people typically have certain aspect ratios)
        float aspect_ratio = static_cast<float>(bbox.width) / bbox.height;
        if (aspect_ratio < 0.3f || aspect_ratio > 1.5f) {
            return false;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Safe pillion position check failed: {}\n", e.what());
        return false;
    }
}

// Advanced pillion detection with motorcycle context
class AdvancedPillionDetector : public PillionRiderDetector {
public:
    struct MotorcycleContext {
        cv::Rect motorcycle_bbox;
        float confidence;
        cv::Point2f center;
        float angle;  // Motorcycle orientation angle
        bool detected = false;
    };
    
    struct RiderContext {
        cv::Rect rider_bbox;
        cv::Point2f center;
        float relative_position[2]; // x, y relative to motorcycle
        RoadDetectionResult::RiderType type;
        bool is_valid = false;
    };
    
    auto detect_with_motorcycle_context(std::span<const DetectionResult> all_detections,
                                       const cv::Size& frame_size) -> std::vector<RoadDetectionResult> {
        std::vector<RoadDetectionResult> enhanced_results;
        
        try {
            // Detect motorcycle first
            auto motorcycle_context = detect_motorcycle(all_detections, frame_size);
            
            if (!motorcycle_context.detected) {
                // Fallback to basic detection if no motorcycle detected
                return detect_pillion_riders(all_detections, frame_size);
            }
            
            // Detect riders in motorcycle context
            auto rider_contexts = detect_riders_with_context(all_detections, motorcycle_context);
            
            // Analyze rider positions and determine types
            for (auto& rider_ctx : rider_contexts) {
                if (rider_ctx.is_valid) {
                    RoadDetectionResult result;
                    result.bbox = rider_ctx.rider_bbox;
                    result.rider_type = rider_ctx.type;
                    result.compliance_status = check_pillion_compliance(result);
                    
                    enhanced_results.push_back(result);
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Advanced pillion detection failed: {}\n", e.what());
        }
        
        return enhanced_results;
    }

private:
    auto detect_motorcycle(std::span<const DetectionResult> detections, 
                          const cv::Size& frame_size) -> MotorcycleContext {
        MotorcycleContext context;
        
        try {
            // Look for motorcycle detections (class_id typically 2 or 3 for motorcycle/bicycle)
            for (const auto& detection : detections) {
                if ((detection.class_id == 2 || detection.class_id == 3) && 
                    detection.confidence > 0.6f) {
                    
                    context.motorcycle_bbox = detection.bbox;
                    context.confidence = detection.confidence;
                    context.center = cv::Point2f(
                        detection.bbox.x + detection.bbox.width / 2.0f,
                        detection.bbox.y + detection.bbox.height / 2.0f
                    );
                    context.detected = true;
                    break;
                }
            }
            
            // Calculate motorcycle orientation if detected
            if (context.detected) {
                context.angle = calculate_motorcycle_angle(context.motorcycle_bbox);
            }
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Motorcycle detection failed: {}\n", e.what());
        }
        
        return context;
    }
    
    auto detect_riders_with_context(std::span<const DetectionResult> detections,
                                   const MotorcycleContext& motorcycle) -> std::vector<RiderContext> {
        std::vector<RiderContext> rider_contexts;
        
        try {
            if (!motorcycle.detected) {
                return rider_contexts;
            }
            
            // Find person detections
            std::vector<DetectionResult> person_detections;
            for (const auto& detection : detections) {
                if (detection.class_id == 0 && detection.confidence > 0.5f) {
                    person_detections.push_back(detection);
                }
            }
            
            // Analyze each person in motorcycle context
            for (const auto& person : person_detections) {
                RiderContext rider_ctx;
                rider_ctx.rider_bbox = person.bbox;
                rider_ctx.center = cv::Point2f(
                    person.bbox.x + person.bbox.width / 2.0f,
                    person.bbox.y + person.bbox.height / 2.0f
                );
                
                // Calculate relative position to motorcycle
                rider_ctx.relative_position[0] = (rider_ctx.center.x - motorcycle.center.x) / motorcycle.motorcycle_bbox.width;
                rider_ctx.relative_position[1] = (rider_ctx.center.y - motorcycle.center.y) / motorcycle.motorcycle_bbox.height;
                
                // Check if rider is on the motorcycle
                if (is_rider_on_motorcycle(rider_ctx, motorcycle)) {
                    rider_ctx.is_valid = true;
                    rider_ctx.type = classify_rider_type(rider_ctx, motorcycle);
                    rider_contexts.push_back(rider_ctx);
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << std::format("Rider context detection failed: {}\n", e.what());
        }
        
        return rider_contexts;
    }
    
    auto calculate_motorcycle_angle(const cv::Rect& bbox) -> float {
        // Simplified angle calculation - in practice, you'd use more sophisticated methods
        return 0.0f; // Assuming motorcycle is mostly horizontal
    }
    
    auto is_rider_on_motorcycle(const RiderContext& rider, const MotorcycleContext& motorcycle) -> bool {
        // Check if rider is within reasonable distance of motorcycle
        float distance = cv::norm(rider.center - motorcycle.center);
        float max_distance = motorcycle.motorcycle_bbox.width * 0.8f; // Within 80% of motorcycle width
        
        return distance <= max_distance;
    }
    
    auto classify_rider_type(const RiderContext& rider, const MotorcycleContext& motorcycle) -> RoadDetectionResult::RiderType {
        // Classify based on relative position to motorcycle center
        float relative_x = rider.relative_position[0];
        
        if (relative_x < -0.1f) {
            // Left side - typically driver (in right-hand traffic countries)
            return RoadDetectionResult::RiderType::DRIVER;
        } else if (relative_x > 0.1f) {
            // Right side - typically pillion
            return RoadDetectionResult::RiderType::PILLION;
        } else {
            // Center - ambiguous
            return RoadDetectionResult::RiderType::UNKNOWN;
        }
    }
};

// Factory function implementations
auto create_pillion_rider_detector() -> std::unique_ptr<PillionRiderDetector> {
    auto detector = std::make_unique<PillionRiderDetector>();
    if (detector->initialize()) {
        return detector;
    }
    
    return nullptr;
}

auto create_advanced_pillion_detector() -> std::unique_ptr<AdvancedPillionDetector> {
    auto detector = std::make_unique<AdvancedPillionDetector>();
    if (detector->initialize()) {
        return detector;
    }
    
    return nullptr;
}

} // namespace BikeGuard
