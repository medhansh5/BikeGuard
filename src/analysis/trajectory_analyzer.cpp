#include "trajectory_analyzer.hpp"
#include <numeric>
#include <algorithm>
#include <iostream>
#include <format>

namespace BikeGuard {

auto TrajectoryAnalyzer::initialize(const TrajectoryConfig& config) -> bool {
    try {
        config_ = config;
        vehicle_tracks_.clear();
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << std::format("Trajectory Analyzer init error: {}\n", e.what());
        return false;
    }
}

auto TrajectoryAnalyzer::track_vehicle(int vehicle_id, const cv::Point2f& centroid, double timestamp_sec) -> WeavingMetrics {
    WeavingMetrics metrics;
    metrics.vehicle_id = vehicle_id;
    metrics.timestamp = std::chrono::steady_clock::now();
    
    if (!initialized_ || vehicle_id < 0) {
        return metrics;
    }

    try {
        auto& history = vehicle_tracks_[vehicle_id];
        history.centroids.push_back(centroid);
        history.timestamps.push_back(timestamp_sec);
        history.last_seen_timestamp = timestamp_sec;
        
        while (history.centroids.size() > config_.history_window_size) {
            history.centroids.pop_front();
            history.timestamps.pop_front();
        }

        if (history.centroids.size() >= 5) {
            metrics.speed_kmh = estimate_speed_kmh(vehicle_id);
            metrics.weaving_instability_index = analyze_weaving_instability(vehicle_id);
            metrics.is_speeding = (metrics.speed_kmh > config_.speed_limit_kmh);
            metrics.is_reckless_weaving = (metrics.weaving_instability_index > config_.weaving_threshold_index);
            
            history.current_speed_kmh = metrics.speed_kmh;
            history.instability_index = metrics.weaving_instability_index;
        }
    } catch (const std::exception& e) {
        std::cerr << std::format("Vehicle tracking error for ID {}: {}\n", vehicle_id, e.what());
    }

    return metrics;
}

auto TrajectoryAnalyzer::estimate_speed_kmh(int vehicle_id) -> float {
    auto it = vehicle_tracks_.find(vehicle_id);
    if (it == vehicle_tracks_.end() || it->second.centroids.size() < 2) {
        return 0.0f;
    }

    const auto& points = it->second.centroids;
    const auto& times = it->second.timestamps;
    
    // Calculate Euclidean displacement over the recent window
    float total_dist_pixels = 0.0f;
    double total_time_sec = 0.0;
    
    for (size_t i = 1; i < points.size(); ++i) {
        float dx = points[i].x - points[i - 1].x;
        float dy = points[i].y - points[i - 1].y;
        total_dist_pixels += std::sqrt(dx * dx + dy * dy);
        total_time_sec += (times[i] - times[i - 1]);
    }

    if (total_time_sec <= 0.001) {
        // Fallback to default FPS if timestamp delta is zero
        total_time_sec = static_cast<double>(points.size() - 1) / config_.fps_default;
    }

    float speed_mps = (total_dist_pixels / config_.pixels_per_meter) / static_cast<float>(total_time_sec);
    float speed_kmh = speed_mps * 3.6f;
    
    // Clamp to realistic motorcycle speed range (0 to 180 km/h)
    return std::clamp(speed_kmh, 0.0f, 180.0f);
}

auto TrajectoryAnalyzer::analyze_weaving_instability(int vehicle_id) -> float {
    auto it = vehicle_tracks_.find(vehicle_id);
    if (it == vehicle_tracks_.end() || it->second.centroids.size() < 5) {
        return 0.0f;
    }

    return compute_angular_variance(it->second.centroids);
}

auto TrajectoryAnalyzer::compute_angular_variance(const std::deque<cv::Point2f>& points) -> float {
    std::vector<float> angles;
    angles.reserve(points.size() - 1);
    
    for (size_t i = 1; i < points.size(); ++i) {
        float dx = points[i].x - points[i - 1].x;
        float dy = points[i].y - points[i - 1].y;
        if (std::abs(dx) > 0.1f || std::abs(dy) > 0.1f) {
            float angle = std::atan2(dy, dx);
            angles.push_back(angle);
        }
    }

    if (angles.size() < 3) {
        return 0.0f;
    }

    // Compute mean angle and variance (instability index)
    float mean_angle = std::accumulate(angles.begin(), angles.end(), 0.0f) / static_cast<float>(angles.size());
    float variance = 0.0f;
    for (float angle : angles) {
        float diff = angle - mean_angle;
        // Normalize angle difference between -pi and pi
        while (diff > static_cast<float>(M_PI)) diff -= static_cast<float>(2.0 * M_PI);
        while (diff < -static_cast<float>(M_PI)) diff += static_cast<float>(2.0 * M_PI);
        variance += diff * diff;
    }

    variance /= static_cast<float>(angles.size());
    return variance * 10.0f; // Scale factor for readability
}

auto TrajectoryAnalyzer::is_reckless_driving(int vehicle_id) -> bool {
    auto it = vehicle_tracks_.find(vehicle_id);
    if (it != vehicle_tracks_.end()) {
        return (it->second.current_speed_kmh > config_.speed_limit_kmh &&
                it->second.instability_index > config_.weaving_threshold_index);
    }
    return false;
}

auto TrajectoryAnalyzer::reset_tracker() -> void {
    vehicle_tracks_.clear();
}

auto TrajectoryAnalyzer::remove_stale_tracks(double current_timestamp_sec, double max_age_sec) -> void {
    for (auto it = vehicle_tracks_.begin(); it != vehicle_tracks_.end(); ) {
        if ((current_timestamp_sec - it->second.last_seen_timestamp) > max_age_sec) {
            it = vehicle_tracks_.erase(it);
        } else {
            ++it;
        }
    }
}

auto create_trajectory_analyzer() -> std::unique_ptr<TrajectoryAnalyzer> {
    auto analyzer = std::make_unique<TrajectoryAnalyzer>();
    if (analyzer->initialize()) {
        return analyzer;
    }
    return nullptr;
}

} // namespace BikeGuard
