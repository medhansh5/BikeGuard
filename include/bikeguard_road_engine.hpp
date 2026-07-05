#pragma once

#include "bikeguard_engine.hpp"
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <memory>
#include <vector>
#include <span>
#include <string>
#include <string_view>
#include <chrono>
#include <mutex>
#include <atomic>
#include <concepts>
#include <fstream>
#include <sstream>
#include <iomanip>

// Use JSON library if available, otherwise provide basic JSON functionality
#ifdef USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
using json = nlohmann::json;
#else
// Basic JSON implementation for compatibility
namespace nlohmann {
    class json {
    public:
        static json object() { return json(); }
        static json array() { return json(); }
        
        template<typename T>
        json& operator[](const T& key) { return *this; }
        
        template<typename T>
        const json& operator[](const T& key) const { return *this; }
        
        template<typename T>
        void push_back(const T& value) {}
        
        std::string dump() const { return "{}"; }
    };
}
using json = nlohmann::json;
#endif

namespace BikeGuard {

using Microsoft::WRL::ComPtr;
using json = nlohmann::json;

// Enhanced detection result for Indian road compliance
struct RoadDetectionResult : public DetectionResult {
    enum class HelmetType {
        UNKNOWN = 0,
        STANDARD_FULL_FACE,
        STANDARD_HALF_FACE,
        SIKH_TURBAN,
        CONSTRUCTION_HELMET,
        NO_HELMET
    };
    
    enum class ComplianceStatus {
        UNKNOWN = 0,
        COMPLIANT,
        NON_COMPLIANT,
        EXEMPT
    };
    
    enum class RiderType {
        UNKNOWN = 0,
        DRIVER,
        PILLION,
        PEDIATRIC_PILLION
    };
    
    HelmetType helmet_type = HelmetType::UNKNOWN;
    ComplianceStatus compliance_status = ComplianceStatus::UNKNOWN;
    RiderType rider_type = RiderType::UNKNOWN;
    float vibration_level = 0.0f;
    bool frame_dropped = false;
    bool in_roi = true; // Region of Interest
    
    RoadDetectionResult() = default;
    RoadDetectionResult(float conf, int cls, const cv::Rect& box, std::string_view name)
        : DetectionResult(conf, cls, box, name) {}
};

// Royal Enfield Classic 350 calibration profile
struct BaronProfile {
    bool enabled = false;
    float handlebar_exclusion_zone = 0.15f; // Lower 15% of frame
    cv::Rect mirror_exclusion_zones[2];    // Left and right mirror zones
    cv::Rect handlebar_roi;                // Handlebar region of interest
    bool auto_calibrate = true;
    
    // Calibration parameters for Royal Enfield Classic 350
    struct {
        float handlebar_height_ratio = 0.65f;  // Handlebar position from top
        float mirror_width_ratio = 0.12f;      // Mirror width relative to frame
        float mirror_height_ratio = 0.08f;    // Mirror height relative to frame
        float mirror_left_x_ratio = 0.15f;    // Left mirror X position
        float mirror_right_x_ratio = 0.75f;   // Right mirror X position
        float mirror_y_ratio = 0.25f;         // Mirror Y position from top
    } geometry;
};

// Vibration analysis settings for ShadowMap v1.3.0 compatibility
struct VibrationAnalysis {
    float high_intensity_threshold = 15.0f;  // Hz threshold for frame dropping
    float amplitude_threshold = 0.7f;        // Amplitude threshold
    bool frame_dropping_enabled = true;
    size_t consecutive_dropped_frames = 0;
    size_t max_consecutive_drops = 3;
    
    // Frequency bands for Indian road conditions
    struct {
        float engine_vibration_min = 10.0f;  // Hz
        float engine_vibration_max = 25.0f;  // Hz
        float road_vibration_min = 2.0f;     // Hz
        float road_vibration_max = 8.0f;     // Hz
        float shock_absorber_min = 15.0f;     // Hz
        float shock_absorber_max = 40.0f;     // Hz
    } frequency_bands;
};

// Enhanced pre-processing pipeline for Indian road variables
class IndianRoadPreprocessor {
public:
    struct PreprocessingConfig {
        // Lighting conditions for Indian roads
        float low_light_threshold = 50.0f;
        float high_light_threshold = 200.0f;
        bool adaptive_contrast = true;
        
        // Environmental factors
        bool dust_compensation = true;
        bool glare_reduction = true;
        bool motion_compensation = true;
        
        // Indian road specific adjustments
        float vehicle_detection_sensitivity = 0.6f;
        float helmet_detection_sensitivity = 0.5f;
        bool pillion_detection = true;
    };
    
    auto initialize(const PreprocessingConfig& config) -> bool;
    auto preprocess_frame(const cv::Mat& input_frame, const BaronProfile& baron_profile) -> cv::Mat;
    auto apply_lighting_correction(cv::Mat& frame) -> void;
    auto apply_dust_compensation(cv::Mat& frame) -> void;
    auto apply_glare_reduction(cv::Mat& frame) -> void;
    auto exclude_handlebar_region(const cv::Mat& frame, const BaronProfile& profile) -> cv::Mat;

private:
    PreprocessingConfig config_;
    bool initialized_ = false;
    
    // Adaptive histogram equalization for varying lighting
    cv::Ptr<cv::CLAHE> clahe_;
    
    // Dust and glare detection
    cv::Mat dust_mask_;
    cv::Mat glare_mask_;
};

// Enhanced helmet classification for Indian road context
class IndianHelmetClassifier {
public:
    enum class ClassificationMethod {
        VISUAL_FEATURES = 0,
        SHAPE_ANALYSIS,
        COLOR_PATTERN,
        DEEP_LEARNING
    };
    
    auto initialize() -> bool;
    auto classify_helmet(const cv::Mat& roi, const DetectionResult& detection) -> RoadDetectionResult;
    auto classify_sikh_turban(const cv::Mat& roi) -> bool;
    auto classify_construction_helmet(const cv::Mat& roi) -> bool;
    auto determine_compliance(const RoadDetectionResult& result) -> ComplianceStatus;

private:
    bool initialized_ = false;
    
    // Feature extraction for helmet types
    std::vector<cv::KeyPoint> extract_helmet_features(const cv::Mat& roi);
    float calculate_shape_confidence(const cv::Mat& roi, RoadDetectionResult::HelmetType type);
    float analyze_color_pattern(const cv::Mat& roi);
    
    // Sikh turban detection patterns
    struct {
        float aspect_ratio_min = 0.8f;
        float aspect_ratio_max = 1.2f;
        float color_uniformity_threshold = 0.7f;
        std::vector<cv::Scalar> typical_colors;
    } turban_features_;
    
    // Construction helmet detection patterns
    struct {
        float aspect_ratio_min = 0.9f;
        float aspect_ratio_max = 1.1f;
        bool has_rib_pattern = false;
        std::vector<cv::Scalar> typical_colors;
    } construction_helmet_features_;
};

// Pillion rider detection and compliance checking
class PillionRiderDetector {
public:
    auto initialize() -> bool;
    auto detect_pillion_riders(std::span<const DetectionResult> person_detections, 
                              const cv::Size& frame_size) -> std::vector<RoadDetectionResult>;
    auto check_pillion_compliance(const RoadDetectionResult& pillion) -> bool;
    auto analyze_rider_positions(std::span<const DetectionResult> detections) -> std::vector<RoadDetectionResult>;

private:
    bool initialized_ = false;
    
    // Position analysis for motorcycle rider detection
    struct {
        float max_rider_distance_ratio = 0.3f;  // Maximum distance between riders
        float pillion_position_x_ratio = 0.6f;  // Typical pillion X position
        float pillion_position_y_ratio = 0.4f;  // Typical pillion Y position
        float motorcycle_width_ratio = 0.4f;     // Expected motorcycle width
        float pediatric_height_ratio_threshold = 0.65f; // Child stature threshold vs adult
        float pediatric_area_ratio_threshold = 0.45f;   // Child area threshold vs adult
    } geometry_;
};

// Enhanced FFT vibration filter with frame dropping logic
class EnhancedVibrationFilter : public IVibrationFilter {
public:
    auto initialize(float sample_rate, size_t fft_size = 1024) -> bool override;
    auto filter_frame(std::span<float> motion_data) -> std::span<float> override;
    auto reset() -> void override;
    auto is_stable() const -> bool override;
    
    // ShadowMap v1.3.0 compatible vibration analysis
    auto analyze_vibration_intensity(std::span<float> motion_data) -> float;
    auto should_drop_frame() -> bool;
    auto get_vibration_metrics() -> VibrationAnalysis;

private:
    bool fft_initialized_ = false;
    size_t fft_size_ = 1024;
    float sample_rate_ = 30.0f;
    
    // FFT buffers
    std::vector<float> fft_input_;
    std::vector<std::complex<float>> fft_output_;
    std::vector<float> frequency_spectrum_;
    
    // Vibration analysis
    VibrationAnalysis vibration_config_;
    float current_vibration_intensity_ = 0.0f;
    std::atomic<bool> frame_should_be_dropped_{false};
    
    // Frequency band analysis
    auto analyze_frequency_bands() -> std::vector<float>;
    auto detect_engine_vibration() -> float;
    auto detect_road_vibration() -> float;
    auto detect_shock_absorber_response() -> float;
};

// Comprehensive telemetry logging system
class TelemetryLogger {
public:
    struct InferenceMetrics {
        std::chrono::system_clock::time_point timestamp;
        double inference_latency_ms;
        double preprocessing_time_ms;
        double postprocessing_time_ms;
        float confidence_score;
        float vibration_level;
        bool frame_dropped;
        size_t detection_count;
        std::vector<RoadDetectionResult> detections;
    };
    
    struct SystemMetrics {
        std::chrono::system_clock::time_point timestamp;
        double cpu_usage_percent;
        size_t memory_usage_mb;
        size_t gpu_memory_usage_mb;
        double gpu_utilization_percent;
        float camera_fps;
        bool camera_connected;
    };
    
    auto initialize(const std::string& log_directory = "telemetry_logs") -> bool;
    auto log_inference_metrics(const InferenceMetrics& metrics) -> void;
    auto log_system_metrics(const SystemMetrics& metrics) -> void;
    auto log_road_event(const std::string& event_type, const json& event_data) -> void;
    auto generate_sha256_signature(std::string_view payload) -> std::string;
    auto flush_logs() -> void;
    auto export_session_summary() -> std::string;
    bool enable_cryptographic_audit = true;

private:
    bool initialized_ = false;
    std::string log_directory_;
    std::ofstream inference_log_;
    std::ofstream system_log_;
    std::ofstream event_log_;
    
    std::mutex log_mutex_;
    std::atomic<size_t> inference_count_{0};
    std::atomic<size_t> total_frames_processed_{0};
    std::atomic<size_t> total_frames_dropped_{0};
    
    // Session statistics
    struct {
        std::chrono::system_clock::time_point session_start;
        double avg_inference_latency = 0.0;
        double max_inference_latency = 0.0;
        double min_inference_latency = std::numeric_limits<double>::max();
        float avg_confidence_score = 0.0f;
        size_t total_detections = 0;
        size_t compliance_violations = 0;
    } session_stats_;
};

// Hardware disconnect handling and recovery
class HardwareManager {
public:
    enum class HardwareStatus {
        CONNECTED = 0,
        DISCONNECTED,
        RECOVERING,
        FAILED
    };
    
    struct HardwareState {
        HardwareStatus camera_status = HardwareStatus::CONNECTED;
        HardwareStatus gpu_status = HardwareStatus::CONNECTED;
        std::chrono::system_clock::time_point last_camera_frame;
        std::chrono::system_clock::time_point last_gpu_operation;
        size_t consecutive_camera_failures = 0;
        size_t consecutive_gpu_failures = 0;
    };
    
    auto initialize() -> bool;
    auto monitor_hardware_health() -> HardwareState;
    auto handle_camera_disconnect() -> bool;
    auto handle_gpu_failure() -> bool;
    auto attempt_hardware_recovery() -> bool;

private:
    bool initialized_ = false;
    HardwareState hardware_state_;
    std::mutex hardware_mutex_;
    
    // Recovery parameters
    struct {
        std::chrono::milliseconds camera_timeout{5000};
        std::chrono::milliseconds gpu_timeout{1000};
        size_t max_camera_retries = 3;
        size_t max_gpu_retries = 3;
        std::chrono::milliseconds recovery_interval{1000};
    } recovery_config_;
};

// Forward declarations for new enterprise modules
class ALPREngine;
class TrajectoryAnalyzer;
class LiveStreamer;

// Main Road Engine for on-road testing
class BikeGuardRoadEngine : public BikeGuardEngine {
public:
    auto initialize_road_mode(const EngineConfig& config, const BaronProfile& baron_profile) -> bool;
    auto process_road_frame(const cv::Mat& frame) -> std::vector<RoadDetectionResult>;
    auto enable_calibration_mode(bool enable) -> void;
    auto enable_live_streaming(int port = 8080) -> bool;
    auto get_road_metrics() -> RoadDetectionResult;
    
    // Royal Enfield specific features
    auto calibrate_baron_profile(const cv::Mat& reference_frame) -> BaronProfile;
    auto update_baron_profile(const BaronProfile& profile) -> void;
    
    // Compliance reporting
    auto generate_compliance_report() -> json;

protected:
    // Enhanced processing pipeline
    auto preprocess_road_frame(const cv::Mat& frame) -> cv::Mat;
    auto classify_detections(std::span<const DetectionResult> detections) -> std::vector<RoadDetectionResult>;
    auto apply_vibration_filtering(const cv::Mat& frame) -> bool;

private:
    // Road-specific components
    std::unique_ptr<IndianRoadPreprocessor> preprocessor_;
    std::unique_ptr<IndianHelmetClassifier> helmet_classifier_;
    std::unique_ptr<PillionRiderDetector> pillion_detector_;
    std::unique_ptr<EnhancedVibrationFilter> vibration_filter_;
    std::unique_ptr<TelemetryLogger> telemetry_logger_;
    std::unique_ptr<HardwareManager> hardware_manager_;
    std::unique_ptr<ALPREngine> alpr_engine_;
    std::unique_ptr<TrajectoryAnalyzer> trajectory_analyzer_;
    std::unique_ptr<LiveStreamer> live_streamer_;
    
    // Configuration
    BaronProfile baron_profile_;
    bool calibration_mode_ = false;
    bool road_mode_initialized_ = false;
    
    // Road-specific metrics
    std::atomic<size_t> compliance_violations_{0};
    std::atomic<size_t> exempt_detections_{0};
    std::atomic<float> current_vibration_level_{0.0f};
};

// Factory functions for road-specific components
auto create_indian_road_preprocessor() -> std::unique_ptr<IndianRoadPreprocessor>;
auto create_indian_helmet_classifier() -> std::unique_ptr<IndianHelmetClassifier>;
auto create_pillion_rider_detector() -> std::unique_ptr<PillionRiderDetector>;
auto create_enhanced_vibration_filter() -> std::unique_ptr<EnhancedVibrationFilter>;
auto create_telemetry_logger() -> std::unique_ptr<TelemetryLogger>;
auto create_hardware_manager() -> std::unique_ptr<HardwareManager>;

// Utility functions for road testing
namespace road_utils {
    auto calculate_compliance_rate(const std::vector<RoadDetectionResult>& detections) -> float;
    auto detect_royal_enframe(const cv::Mat& frame) -> bool;
    auto estimate_road_conditions(const cv::Mat& frame) -> std::string;
    auto validate_detection_quality(const RoadDetectionResult& detection) -> bool;
}

} // namespace BikeGuard
