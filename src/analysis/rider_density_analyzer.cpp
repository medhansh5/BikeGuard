#include "rider_density_analyzer.hpp"
#include "bikeguard_engine.hpp"

namespace BikeGuard {

auto RiderDensityAnalyzer::analyze_rider_density(const std::vector<DetectionResult>& detections) 
    -> DensityAnalysis {
    
    // Initialize analysis result
    DensityAnalysis result;
    
    // Get motorcycle and person detections
    auto motorcycles = get_motorcycle_detections(detections);
    auto persons = get_person_detections(detections);
    
    result.total_motorcycles = motorcycles.size();
    result.total_persons = persons.size();
    
    // Analyze each motorcycle for rider density
    for (const auto* motorcycle : motorcycles) {
        if (!motorcycle) continue;
        
        size_t rider_count = 0;
        std::vector<const DetectionResult*> associated_persons;
        float max_ioa = 0.0f;
        
        // Check each person for association with this motorcycle
        for (const auto* person : persons) {
            if (!person) continue;
            
            // Check X-axis overlap and IoA threshold
            if (is_person_overlapping_motorcycle_x(*person, *motorcycle)) {
                float ioa = calculate_ioa(*person, *motorcycle);
                
                if (ioa > MIN_IOA_THRESHOLD) {
                    rider_count++;
                    associated_persons.push_back(person);
                    max_ioa = std::max(max_ioa, ioa);
                }
            }
        }
        
        // Update maximum riders found on any motorcycle
        if (rider_count > result.max_riders_on_motorcycle) {
            result.max_riders_on_motorcycle = rider_count;
            result.highest_ioa = max_ioa;
            
            // Store violation details if threshold exceeded
            if (rider_count >= TRIPLE_RIDING_THRESHOLD) {
                result.state = DensityState::TRIPLE_RIDING_VIOLATION;
                result.violation_motorcycles.clear();
                result.violation_persons.clear();
                
                result.violation_motorcycles.push_back(motorcycle);
                result.violation_persons = associated_persons;
            }
        }
    }
    
    // Set final state based on maximum riders found
    if (result.state == DensityState::NO_VIOLATION) {
        if (result.max_riders_on_motorcycle >= 3) {
            result.state = DensityState::TRIPLE_RIDING_VIOLATION;
        } else if (result.max_riders_on_motorcycle == 2) {
            result.state = DensityState::DOUBLE_RIDING;
        } else if (result.max_riders_on_motorcycle == 1) {
            result.state = DensityState::SINGLE_RIDER;
        }
    }
    
    return result;
}

auto RiderDensityAnalyzer::calculate_ioa(const DetectionResult& person, const DetectionResult& motorcycle) noexcept 
    -> float {
    
    // Calculate intersection area
    float intersection_area = calculate_intersection_area(person, motorcycle);
    
    // Return Intersection over Area (IoA)
    float person_area = static_cast<float>(person.bbox.area());
    return (person_area > 0.0f) ? (intersection_area / person_area) : 0.0f;
}

auto RiderDensityAnalyzer::is_person_overlapping_motorcycle_x(const DetectionResult& person, 
                                                  const DetectionResult& motorcycle) noexcept 
    -> bool {
    
    // Get person's center point
    auto person_center = get_bbox_center(person);
    
    // Check if person's X center is within motorcycle's X boundaries
    return (person_center.first >= motorcycle.bbox.x && 
            person_center.first <= (motorcycle.bbox.x + motorcycle.bbox.width));
}

auto RiderDensityAnalyzer::get_bbox_center(const DetectionResult& detection) noexcept 
    -> std::pair<float, float> {
    
    float center_x = detection.bbox.x + (detection.bbox.width * 0.5f);
    float center_y = detection.bbox.y + (detection.bbox.height * 0.5f);
    
    return {center_x, center_y};
}

auto RiderDensityAnalyzer::get_motorcycle_detections(const std::vector<DetectionResult>& detections) 
    -> std::vector<const DetectionResult*> {
    
    std::vector<const DetectionResult*> motorcycles;
    motorcycles.reserve(detections.size());
    
    for (const auto& detection : detections) {
        if (detection.class_id == 3) { // Motorcycle class ID
            motorcycles.push_back(&detection);
        }
    }
    
    return motorcycles;
}

auto RiderDensityAnalyzer::get_person_detections(const std::vector<DetectionResult>& detections) 
    -> std::vector<const DetectionResult*> {
    
    std::vector<const DetectionResult*> persons;
    persons.reserve(detections.size());
    
    for (const auto& detection : detections) {
        if (detection.class_id == 0) { // Person class ID
            persons.push_back(&detection);
        }
    }
    
    return persons;
}

auto RiderDensityAnalyzer::calculate_intersection_area(const DetectionResult& a, const DetectionResult& b) noexcept 
    -> float {
    
    // Calculate intersection rectangle
    int x1 = std::max(a.bbox.x, b.bbox.x);
    int y1 = std::max(a.bbox.y, b.bbox.y);
    int x2 = std::min(a.bbox.x + a.bbox.width, b.bbox.x + b.bbox.width);
    int y2 = std::min(a.bbox.y + a.bbox.height, b.bbox.y + b.bbox.height);
    
    // Calculate intersection area
    if (x2 > x1 && y2 > y1) {
        return static_cast<float>((x2 - x1) * (y2 - y1));
    }
    
    return 0.0f; // No intersection
}

} // namespace BikeGuard
