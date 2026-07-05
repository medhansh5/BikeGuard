#pragma once

#include "bikeguard_core.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <unordered_map>
#include <memory>
#include <chrono>
#include <cmath>

namespace BikeGuard {

// Metrics representing speed and trajectory instability (reckless driving)
struct WeavingMetrics {
    int vehicle_id{-1};
    float speed_kmh{0.0f};
    float angular_variance{0.0f};
    float weaving_instability_index{0.0f};
    bool is_reckless_weaving{false};
    bool is_speeding{false};
    std::chrono::steady_clock::time_point timestamp;
};

// Optical Flow Speed Estimation & Trajectory Analyzer for Indian Traffic
class TrajectoryAnalyzer {
public:
    struct TrajectoryConfig {
        size_t history_window_size{30};         // Rolling window of frames
        float speed_limit_kmh{60.0f};           // Urban Indian road speed limit
        float weaving_threshold_index{1.8f};    // Angular variance threshold for reckless weaving
        float pixels_per_meter{45.0f};          // Camera height calibration factor
        float fps_default{30.0f};
    };

    TrajectoryAnalyzer() = default;
    ~TrajectoryAnalyzer() = default;

    auto initialize(const TrajectoryConfig& config = TrajectoryConfig{}) -> bool;
    auto track_vehicle(int vehicle_id, const cv::Point2f& centroid, double timestamp_sec = 0.0) -> WeavingMetrics;
    auto estimate_speed_kmh(int vehicle_id) -> float;
    auto analyze_weaving_instability(int vehicle_id) -> float;
    auto is_reckless_driving(int vehicle_id) -> bool;
    auto reset_tracker() -> void;
    auto remove_stale_tracks(double current_timestamp_sec, double max_age_sec = 5.0) -> void;

private:
    bool initialized_{false};
    TrajectoryConfig config_;
    
    struct VehicleHistory {
        std::deque<cv::Point2f> centroids;
        std::deque<double> timestamps;
        float current_speed_kmh{0.0f};
        float instability_index{0.0f};
        double last_seen_timestamp{0.0};
    };
    
    std::unordered_map<int, VehicleHistory> vehicle_tracks_;
    
    // Internal computation methods
    auto calculate_velocity_vector(const cv::Point2f& p1, const cv::Point2f& p2, double dt_sec) -> cv::Point2f;
    auto compute_angular_variance(const std::deque<cv::Point2f>& points) -> float;
};

auto create_trajectory_analyzer() -> std::unique_ptr<TrajectoryAnalyzer>;

} // namespace BikeGuard
