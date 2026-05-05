#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <mutex>

#ifdef _WIN32
    #include <windows.h>
    #include <dxgi1_2.h>
    #include <d3d11.h>
    #include <wrl/client.h>
    using Microsoft::WRL::ComPtr;
#endif

namespace BikeGuard {

// Forward declarations
class InferenceEngine;
class CameraCapture;
class HardwareAccelerator;

// Core data structures
struct DetectionResult {
    float confidence;
    int class_id;
    cv::Rect bbox;
    std::string class_name;
    
    DetectionResult() = default;
    DetectionResult(float conf, int cls, const cv::Rect& box, const std::string& name)
        : confidence(conf), class_id(cls), bbox(box), class_name(name) {}
};

struct InferenceConfig {
    std::string model_path;
    cv::Size input_size{640, 640};
    float confidence_threshold{0.5f};
    float nms_threshold{0.4f};
    int max_detections{100};
    bool use_gpu{false};
    
    #ifdef _WIN32
    int gpu_device_id{0};
    #endif
};

struct PerformanceMetrics {
    double fps{0.0};
    double inference_time_ms{0.0};
    double preprocessing_time_ms{0.0};
    double postprocessing_time_ms{0.0};
    size_t frame_count{0};
    std::chrono::steady_clock::time_point last_update;
};

// Abstract interface for camera capture
class CameraCapture {
public:
    virtual ~CameraCapture() = default;
    virtual bool initialize(int camera_id = 0) = 0;
    virtual bool capture_frame(cv::Mat& frame) = 0;
    virtual void release() = 0;
    virtual cv::Size get_frame_size() const = 0;
    virtual double get_fps() const = 0;
    
    #ifdef _WIN32
    virtual bool initialize_directx() = 0;
    virtual bool capture_frame_dx(cv::Mat& frame) = 0;
    #endif
};

// Hardware abstraction layer
class HardwareAccelerator {
public:
    virtual ~HardwareAccelerator() = default;
    virtual bool initialize(const InferenceConfig& config) = 0;
    virtual void* get_execution_provider() = 0;
    virtual bool is_available() const = 0;
    virtual std::string get_provider_name() const = 0;
    
    #ifdef _WIN32
    virtual bool initialize_d3d(int device_id) = 0;
    virtual ID3D11Device* get_d3d_device() = 0;
    #endif
};

// Main inference engine
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine();
    
    // Core functionality
    bool initialize(const InferenceConfig& config);
    bool load_model(const std::string& model_path);
    std::vector<DetectionResult> run_inference(const cv::Mat& input_frame);
    void cleanup();
    
    // Performance monitoring
    PerformanceMetrics get_metrics() const;
    void reset_metrics();
    
    // Configuration
    void set_confidence_threshold(float threshold);
    void set_nms_threshold(float threshold);
    void set_input_size(const cv::Size& size);
    
private:
    // Internal state
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    std::unique_ptr<HardwareAccelerator> hardware_accel_;
    
    // Model information
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    
    // Preprocessing buffers (zero-allocation design)
    cv::Mat preprocessed_buffer_;
    std::vector<float> input_tensor_data_;
    
    // Configuration
    InferenceConfig config_;
    
    // Performance tracking
    mutable std::mutex metrics_mutex_;
    PerformanceMetrics metrics_;
    
    // Internal methods
    bool setup_session();
    void preprocess_frame(const cv::Mat& frame);
    std::vector<DetectionResult> postprocess_results(const std::vector<Ort::Value>& outputs);
    void update_performance_metrics(double inference_time, double preprocess_time, double postprocess_time);
};

// Factory functions for platform-specific implementations
#ifdef _WIN32
std::unique_ptr<CameraCapture> create_directx_camera();
std::unique_ptr<HardwareAccelerator> create_d3d_accelerator();
#endif

std::unique_ptr<CameraCapture> create_opencv_camera();
std::unique_ptr<HardwareAccelerator> create_cpu_accelerator();

// Utility functions
namespace utils {
    cv::Mat letterbox_resize(const cv::Mat& image, const cv::Size& target_size, cv::Scalar fill_color = cv::Scalar(114, 114, 114));
    std::vector<DetectionResult> apply_nms(const std::vector<DetectionResult>& detections, float nms_threshold);
    void draw_detections(cv::Mat& image, const std::vector<DetectionResult>& detections);
    std::string get_class_name(int class_id);
}

// Error handling
enum class ErrorCode {
    SUCCESS = 0,
    MODEL_LOAD_FAILED,
    INFERENCE_FAILED,
    CAMERA_INIT_FAILED,
    HARDWARE_ACCEL_FAILED,
    INVALID_CONFIG
};

class BikeGuardException : public std::exception {
public:
    BikeGuardException(ErrorCode code, const std::string& message);
    const char* what() const noexcept override;
    ErrorCode get_error_code() const noexcept;
    
private:
    ErrorCode error_code_;
    std::string message_;
};

} // namespace BikeGuard
