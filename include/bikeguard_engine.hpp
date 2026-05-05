#pragma once

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
#include <chrono>
#include <mutex>
#include <atomic>
#include <concepts>

namespace BikeGuard {

using Microsoft::WRL::ComPtr;

// Modern C++20 concepts for type safety
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;

template<typename T>
concept FloatingPoint = std::floating_point<T>;

// High-performance detection result with zero-copy design
struct DetectionResult {
    float confidence;
    int class_id;
    cv::Rect bbox;
    std::string_view class_name;  // Use string_view to avoid allocations
    
    constexpr DetectionResult() noexcept = default;
    constexpr DetectionResult(float conf, int cls, const cv::Rect& box, std::string_view name) noexcept
        : confidence(conf), class_id(cls), bbox(box), class_name(name) {}
};

// Temporal smoothing state machine enums
enum class ComplianceState : uint8_t {
    STATE_COMPLIANT = 0,
    STATE_VIOLATION = 1,
    STATE_TRANSITIONING = 2
};

enum class DetectionPresence : uint8_t {
    NO_DETECTION = 0,
    RIDER_DETECTED = 1,
    HELMET_DETECTED = 2
};

// Zero-allocation frame history for temporal smoothing
struct FrameHistory {
    static constexpr size_t HISTORY_SIZE = 30;  // ~1 second at 30 FPS
    
    std::array<DetectionPresence, HISTORY_SIZE> detections{};
    std::array<float, HISTORY_SIZE> confidences{};
    std::array<uint64_t, HISTORY_SIZE> timestamps{};
    size_t current_index = 0;
    size_t frame_count = 0;
    
    constexpr FrameHistory() noexcept = default;
    
    auto add_frame(DetectionPresence detection, float confidence, uint64_t timestamp) noexcept -> void {
        detections[current_index] = detection;
        confidences[current_index] = confidence;
        timestamps[current_index] = timestamp;
        
        current_index = (current_index + 1) % HISTORY_SIZE;
        if (frame_count < HISTORY_SIZE) frame_count++;
    }
    
    auto get_recent_count(DetectionPresence detection, size_t count) const noexcept -> size_t {
        size_t result = 0;
        size_t check_count = std::min(count, frame_count);
        
        for (size_t i = 0; i < check_count; ++i) {
            size_t index = (current_index - 1 - i + HISTORY_SIZE) % HISTORY_SIZE;
            if (detections[index] == detection) result++;
        }
        
        return result;
    }
    
    auto reset() noexcept -> void {
        detections.fill(NO_DETECTION);
        confidences.fill(0.0f);
        timestamps.fill(0);
        current_index = 0;
        frame_count = 0;
    }
};

// Configuration structure with modern defaults
struct EngineConfig {
    std::string model_path = "models/helmet_detector.onnx";
    cv::Size input_size{640, 640};
    float confidence_threshold = 0.5f;
    float nms_threshold = 0.4f;
    int max_detections = 100;
    bool use_gpu = true;
    int gpu_device_id = 0;
    
    // Performance tuning parameters
    int intra_op_num_threads = 1;
    int inter_op_num_threads = 1;
    bool enable_cpu_mem_arena = true;
    bool enable_gpu_mem_arena = true;
    
    // Temporal smoothing configuration
    bool enable_temporal_smoothing = true;
    size_t violation_threshold_frames = 5;  // N consecutive frames for violation
    size_t compliance_threshold_frames = 2; // M consecutive frames for compliance
    float vibration_override_threshold_hz = 15.0f;  // Vibration threshold for state freeze
};

// Performance metrics with atomic updates for thread safety
struct PerformanceMetrics {
    std::atomic<double> fps{0.0};
    std::atomic<double> inference_time_ms{0.0};
    std::atomic<double> preprocessing_time_ms{0.0};
    std::atomic<double> postprocessing_time_ms{0.0};
    std::atomic<size_t> frame_count{0};
    std::atomic<size_t> total_detections{0};
    
    auto get_snapshot() const noexcept -> std::tuple<double, double, double, double, size_t, size_t> {
        return {
            fps.load(),
            inference_time_ms.load(),
            preprocessing_time_ms.load(),
            postprocessing_time_ms.load(),
            frame_count.load(),
            total_detections.load()
        };
    }
};

// Abstract camera interface for modern C++ design
class ICameraCapture {
public:
    virtual ~ICameraCapture() = default;
    virtual auto initialize(int camera_id = 0) -> bool = 0;
    virtual auto capture_frame(cv::Mat& frame) -> bool = 0;
    virtual auto release() -> void = 0;
    virtual auto get_frame_size() const -> cv::Size = 0;
    virtual auto get_fps() const -> double = 0;
    virtual auto is_initialized() const -> bool = 0;
};

// Modern FFT vibration filtering interface
class IVibrationFilter {
public:
    virtual ~IVibrationFilter() = default;
    virtual auto initialize(float sample_rate, size_t fft_size = 1024) -> bool = 0;
    virtual auto filter_frame(std::span<float> motion_data) -> std::span<float> = 0;
    virtual auto reset() -> void = 0;
    virtual auto is_stable() const -> bool = 0;
};

// Main BikeGuard Engine with DirectML and zero-allocation design
class BikeGuardEngine {
public:
    BikeGuardEngine();
    ~BikeGuardEngine() noexcept;

    // Core functionality
    auto initialize(const EngineConfig& config) -> bool;
    auto load_model(const std::string& model_path) -> bool;
    auto run_inference(const cv::Mat& input_frame) -> std::vector<DetectionResult>;
    auto cleanup() -> void;

    // Modern configuration methods
    auto set_confidence_threshold(float threshold) noexcept -> void;
    auto set_nms_threshold(float threshold) noexcept -> void;
    auto set_input_size(const cv::Size& size) -> void;
    auto get_config() const noexcept -> const EngineConfig&;

    // Performance monitoring
    auto get_metrics() const noexcept -> PerformanceMetrics;
    auto reset_metrics() noexcept -> void;

    // Camera management
    auto set_camera(std::unique_ptr<ICameraCapture> camera) -> void;
    auto get_camera() const noexcept -> ICameraCapture*;

    // Vibration filtering
    auto set_vibration_filter(std::unique_ptr<IVibrationFilter> filter) -> void;
    auto enable_vibration_filtering(bool enable) noexcept -> void;

    // Temporal smoothing state machine
    auto update_state(const DetectionResult& current_frame, float current_vibration_hz) -> ComplianceState;
    auto get_compliance_state() const noexcept -> ComplianceState;
    auto get_stable_detections() const noexcept -> std::vector<DetectionResult>;
    auto reset_state_machine() noexcept -> void;

    // Telemetry dispatcher integration
    auto set_telemetry_dispatcher(std::unique_ptr<TelemetryDispatcher> dispatcher) -> void;
    auto enable_telemetry(bool enable) noexcept -> void;
    auto is_telemetry_enabled() const noexcept -> bool;

    // Calibration integration
    auto load_calibration_config() -> bool;
    auto apply_calibration_exclusion(cv::Mat& frame) const -> void;
    auto get_exclusion_zone() const -> cv::Rect;

    // DirectML/GPU management
    auto is_gpu_available() const noexcept -> bool;
    auto get_gpu_info() const -> std::string;

private:
    // Core ONNX Runtime components
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<Ort::MemoryInfo> memory_info_;

    // DirectML components
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
    ComPtr<IDXGIAdapter> dxgi_adapter_;

    // Model metadata
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    // Zero-allocation buffers
    cv::Mat preprocessed_buffer_;
    std::vector<float> input_tensor_data_;
    std::vector<float> output_tensor_data_;
    
    // Configuration and state
    EngineConfig config_;
    PerformanceMetrics metrics_;
    mutable std::mutex metrics_mutex_;
    std::atomic<bool> initialized_{false};
    std::atomic<bool> gpu_available_{false};

    // Camera and filtering
    std::unique_ptr<ICameraCapture> camera_;
    std::unique_ptr<IVibrationFilter> vibration_filter_;
    std::atomic<bool> vibration_filtering_enabled_{false};

    // Temporal smoothing state machine
    FrameHistory frame_history_;
    std::atomic<ComplianceState> current_state_{ComplianceState::STATE_COMPLIANT};
    std::vector<DetectionResult> stable_detections_;
    std::atomic<bool> state_frozen_{false};
    std::atomic<uint64_t> last_state_change_{0};
    mutable std::mutex state_mutex_;

    // Telemetry dispatcher
    std::unique_ptr<TelemetryDispatcher> telemetry_dispatcher_;
    std::atomic<bool> telemetry_enabled_{false};

    // Calibration configuration
    cv::Rect exclusion_zone_{0, 0, 0, 0};
    bool calibration_loaded_{false};

    // Internal methods
    auto initialize_directml() -> bool;
    auto setup_session_options() -> void;
    auto load_model_metadata() -> bool;
    auto preprocess_frame(const cv::Mat& frame) -> std::span<float>;
    auto postprocess_results(std::span<const float> output_data, 
                           const std::vector<int64_t>& output_shape) 
        -> std::vector<DetectionResult>;
    auto update_performance_metrics(double inference_time, 
                                 double preprocess_time, 
                                 double postprocess_time,
                                 size_t detection_count) noexcept -> void;
    
    // Utility methods
    static auto apply_nms(std::vector<DetectionResult>& detections, 
                         float nms_threshold) -> std::vector<DetectionResult>;
    static auto get_class_name(int class_id) noexcept -> std::string_view;
};

// Factory functions for modern C++ design
auto create_msmf_camera() -> std::unique_ptr<ICameraCapture>;
auto create_fft_vibration_filter() -> std::unique_ptr<IVibrationFilter>;

// Utility functions with modern C++20 features
namespace utils {
    template<Numeric T>
    auto letterbox_resize(const cv::Mat& image, const cv::Size& target_size, 
                         T scale_factor = 1.0f) -> cv::Mat;
    
    auto draw_detections(cv::Mat& image, 
                        std::span<const DetectionResult> detections) -> void;
    
    template<FloatingPoint T>
    auto calculate_iou(const cv::Rect& box1, const cv::Rect& box2) noexcept -> T;
    
    auto get_gpu_memory_info() -> std::pair<size_t, size_t>; // used, total
}

// Exception hierarchy for better error handling
class BikeGuardException : public std::exception {
public:
    enum class ErrorCode {
        SUCCESS = 0,
        MODEL_LOAD_FAILED,
        INFERENCE_FAILED,
        CAMERA_INIT_FAILED,
        GPU_INIT_FAILED,
        INVALID_CONFIG,
        MEMORY_ALLOCATION_FAILED,
        DIRECTML_INIT_FAILED
    };

    BikeGuardException(ErrorCode code, std::string message);
    auto what() const noexcept -> const char* override;
    auto get_error_code() const noexcept -> ErrorCode;

private:
    ErrorCode error_code_;
    std::string message_;
};

// RAII helpers for Windows COM objects
template<typename T>
class ComPtrGuard {
public:
    ComPtrGuard() = default;
    explicit ComPtrGuard(T* ptr) : ptr_(ptr) {}
    ~ComPtrGuard() { if (ptr_) ptr_->Release(); }
    
    auto get() const noexcept -> T* { return ptr_; }
    auto release() noexcept -> T* { 
        T* temp = ptr_; 
        ptr_ = nullptr; 
        return temp; 
    }
    
    // Move-only type
    ComPtrGuard(const ComPtrGuard&) = delete;
    auto operator=(const ComPtrGuard&) -> ComPtrGuard& = delete;
    ComPtrGuard(ComPtrGuard&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    auto operator=(ComPtrGuard&& other) noexcept -> ComPtrGuard& {
        if (this != &other) {
            if (ptr_) ptr_->Release();
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

private:
    T* ptr_ = nullptr;
};

} // namespace BikeGuard
