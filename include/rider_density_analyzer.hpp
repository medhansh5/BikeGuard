#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace BikeGuard {

// Forward declaration
struct DetectionResult;

// Rider density analysis states
enum class DensityState : uint8_t {
    NO_VIOLATION = 0,
    SINGLE_RIDER = 1,
    DOUBLE_RIDING = 2,
    TRIPLE_RIDING_VIOLATION = 3
};

// Rider density analysis result
struct DensityAnalysis {
    DensityState state = DensityState::NO_VIOLATION;
    size_t max_riders_on_motorcycle = 0;
    size_t total_motorcycles = 0;
    size_t total_persons = 0;
    float highest_ioa = 0.0f;
    
    // Bounding boxes involved in violation (if any)
    std::vector<const DetectionResult*> violation_motorcycles;
    std::vector<const DetectionResult*> violation_persons;
    
    constexpr DensityAnalysis() noexcept = default;
    constexpr DensityAnalysis(DensityState s, size_t max_riders, size_t motorcycles, 
                           size_t persons, float max_ioa) noexcept
        : state(s), max_riders_on_motorcycle(max_riders), 
          total_motorcycles(motorcycles), total_persons(persons), highest_ioa(max_ioa) {}
};

// Stateless utility class for rider density analysis
class RiderDensityAnalyzer {
public:
    // Configuration constants (constexpr for performance)
    static constexpr float MIN_IOA_THRESHOLD = 0.3f;  // 30% overlap threshold
    static constexpr size_t TRIPLE_RIDING_THRESHOLD = 3;  // 3+ riders = violation
    
    // Main analysis function
    static auto analyze_rider_density(const std::vector<DetectionResult>& detections) 
        -> DensityAnalysis;
    
    // Utility functions
    static auto calculate_ioa(const DetectionResult& person, const DetectionResult& motorcycle) noexcept 
        -> float;
    static auto is_person_overlapping_motorcycle_x(const DetectionResult& person, 
                                                  const DetectionResult& motorcycle) noexcept 
        -> bool;
    static auto get_bbox_center(const DetectionResult& detection) noexcept 
        -> std::pair<float, float>;

private:
    // Internal helper functions
    static auto get_motorcycle_detections(const std::vector<DetectionResult>& detections) 
        -> std::vector<const DetectionResult*>;
    static auto get_person_detections(const std::vector<DetectionResult>& detections) 
        -> std::vector<const DetectionResult*>;
    static auto calculate_intersection_area(const DetectionResult& a, const DetectionResult& b) noexcept 
        -> float;
};

} // namespace BikeGuard
