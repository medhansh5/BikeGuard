#include "bikeguard_core.hpp"
#include <algorithm>
#include <numeric>

namespace BikeGuard {

// Class name mappings for helmet detection
static const std::vector<std::string> CLASS_NAMES = {
    "person", "bicycle", "motorcycle", "helmet", "no_helmet"
};

InferenceEngine::InferenceEngine() 
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BikeGuard")),
      session_options_(std::make_unique<Ort::SessionOptions>()),
      metrics_{} {
    
    // Initialize metrics
    metrics_.last_update = std::chrono::steady_clock::now();
    
    // Set up session options for performance
    session_options_->SetIntraOpNumThreads(1);
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
#ifdef _WIN32
    // Enable memory arena on Windows for better performance
    session_options_->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
#endif
}

InferenceEngine::~InferenceEngine() {
    cleanup();
}

bool InferenceEngine::initialize(const InferenceConfig& config) {
    config_ = config;
    
    // Initialize hardware accelerator if requested
    if (config.use_gpu) {
#ifdef _WIN32
        hardware_accel_ = create_d3d_accelerator();
        if (!hardware_accel_ || !hardware_accel_->initialize(config)) {
            hardware_accel_ = create_cpu_accelerator();
            config_.use_gpu = false;
        }
#else
        hardware_accel_ = create_cpu_accelerator();
        config_.use_gpu = false;
#endif
    } else {
        hardware_accel_ = create_cpu_accelerator();
    }
    
    if (!hardware_accel_ || !hardware_accel_->is_available()) {
        throw BikeGuardException(ErrorCode::HARDWARE_ACCEL_FAILED, 
                                "Failed to initialize hardware accelerator");
    }
    
    // Pre-allocate buffers for zero-allocation design
    const size_t input_size = config_.input_size.width * config_.input_size.height * 3;
    input_tensor_data_.reserve(input_size);
    preprocessed_buffer_.create(config_.input_size.height, config_.input_size.width, CV_32FC3);
    
    return load_model(config_.model_path);
}

bool InferenceEngine::load_model(const std::string& model_path) {
    try {
        // Apply hardware accelerator settings
        if (hardware_accel_) {
            auto* provider = hardware_accel_->get_execution_provider();
            if (provider) {
                session_options_->AppendExecutionProvider(provider);
            }
        }
        
        // Create session
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        
        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        size_t num_input_nodes = session_->GetInputCount();
        input_names_.resize(num_input_nodes);
        input_shapes_.resize(num_input_nodes);
        
        for (size_t i = 0; i < num_input_nodes; ++i) {
            char* input_name = session_->GetInputName(i, allocator);
            input_names_[i] = input_name;
            
            Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(i);
            auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
            auto input_dims = input_tensor_info.GetShape();
            input_shapes_[i] = std::vector<int64_t>(input_dims.begin(), input_dims.end());
        }
        
        // Output info
        size_t num_output_nodes = session_->GetOutputCount();
        output_names_.resize(num_output_nodes);
        output_shapes_.resize(num_output_nodes);
        
        for (size_t i = 0; i < num_output_nodes; ++i) {
            char* output_name = session_->GetOutputName(i, allocator);
            output_names_[i] = output_name;
            
            Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(i);
            auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
            auto output_dims = output_tensor_info.GetShape();
            output_shapes_[i] = std::vector<int64_t>(output_dims.begin(), output_dims.end());
        }
        
        return true;
    }
    catch (const std::exception& e) {
        throw BikeGuardException(ErrorCode::MODEL_LOAD_FAILED, 
                               std::string("Failed to load model: ") + e.what());
    }
}

std::vector<DetectionResult> InferenceEngine::run_inference(const cv::Mat& input_frame) {
    if (!session_) {
        throw BikeGuardException(ErrorCode::INFERENCE_FAILED, "Session not initialized");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Preprocessing (zero-allocation)
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    preprocess_frame(input_frame);
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    
    // Create input tensor (reuse pre-allocated data)
    std::vector<int64_t> input_shape = {1, 3, config_.input_size.height, config_.input_size.width};
    
    // Ensure input_tensor_data_ has correct size
    const size_t required_size = config_.input_size.width * config_.input_size.height * 3;
    if (input_tensor_data_.size() != required_size) {
        input_tensor_data_.resize(required_size);
    }
    
    // Convert BGR to RGB and normalize to [0,1] in one pass
    cv::Mat rgb_image;
    cv::cvtColor(preprocessed_buffer_, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(rgb_image, CV_32F, 1.0/255.0);
    
    // Copy data to tensor buffer (zero-allocation)
    float* tensor_data = input_tensor_data_.data();
    for (int y = 0; y < rgb_image.rows; ++y) {
        const float* row_ptr = rgb_image.ptr<float>(y);
        for (int x = 0; x < rgb_image.cols; ++x) {
            for (int c = 0; c < 3; ++c) {
                tensor_data[c * rgb_image.rows * rgb_image.cols + y * rgb_image.cols + x] = 
                    row_ptr[x * 3 + c];
            }
        }
    }
    
    // Create input tensor
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Value::CreateTensor<float>(
        memory_info, tensor_data, required_size, input_shape.data(), input_shape.size()));
    
    // Run inference
    auto inference_start = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, input_names_.data(), input_tensors.data(), 
        input_names_.size(), output_names_.data(), output_names_.size());
    auto inference_end = std::chrono::high_resolution_clock::now();
    
    // Postprocessing
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    auto results = postprocess_results(output_tensors);
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    
    // Update performance metrics
    double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    
    update_performance_metrics(inference_time, preprocess_time, postprocess_time);
    
    return results;
}

void InferenceEngine::preprocess_frame(const cv::Mat& frame) {
    // Zero-allocation preprocessing using pre-allocated buffer
    cv::resize(frame, preprocessed_buffer_, config_.input_size, 0, 0, cv::INTER_LINEAR);
}

std::vector<DetectionResult> InferenceEngine::postprocess_results(const std::vector<Ort::Value>& outputs) {
    std::vector<DetectionResult> detections;
    
    if (outputs.empty()) return detections;
    
    // Assuming YOLOv8 output format: [batch, num_detections, 85] where 85 = 4(bbox) + 1(conf) + 80(classes)
    float* output_data = outputs[0].GetTensorMutableData<float>();
    auto output_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    
    const int num_detections = output_shape[1];
    const int num_classes = output_shape[2] - 5; // Subtract bbox and confidence
    
    for (int i = 0; i < num_detections; ++i) {
        const float* detection = output_data + i * output_shape[2];
        
        // Extract bbox (center_x, center_y, width, height)
        float center_x = detection[0];
        float center_y = detection[1];
        float width = detection[2];
        float height = detection[3];
        
        // Convert to corner format
        int x1 = static_cast<int>(center_x - width / 2);
        int y1 = static_cast<int>(center_y - height / 2);
        int x2 = static_cast<int>(center_x + width / 2);
        int y2 = static_cast<int>(center_y + height / 2);
        
        // Clamp to image bounds
        x1 = std::max(0, std::min(x1, config_.input_size.width - 1));
        y1 = std::max(0, std::min(y1, config_.input_size.height - 1));
        x2 = std::max(0, std::min(x2, config_.input_size.width - 1));
        y2 = std::max(0, std::min(y2, config_.input_size.height - 1));
        
        // Find best class
        float max_conf = 0.0f;
        int best_class = -1;
        
        for (int c = 0; c < num_classes; ++c) {
            float class_conf = detection[5 + c];
            if (class_conf > max_conf) {
                max_conf = class_conf;
                best_class = c;
            }
        }
        
        // Apply confidence threshold
        if (max_conf >= config_.confidence_threshold && best_class >= 0) {
            cv::Rect bbox(x1, y1, x2 - x1, y2 - y1);
            std::string class_name = (best_class < CLASS_NAMES.size()) ? 
                                    CLASS_NAMES[best_class] : "unknown";
            
            detections.emplace_back(max_conf, best_class, bbox, class_name);
        }
    }
    
    // Apply Non-Maximum Suppression
    return utils::apply_nms(detections, config_.nms_threshold);
}

void InferenceEngine::update_performance_metrics(double inference_time, double preprocess_time, double postprocess_time) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    
    metrics_.inference_time_ms = inference_time;
    metrics_.preprocessing_time_ms = preprocess_time;
    metrics_.postprocessing_time_ms = postprocess_time;
    metrics_.frame_count++;
    
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(now - metrics_.last_update).count();
    
    if (duration >= 1.0) { // Update FPS every second
        metrics_.fps = metrics_.frame_count / duration;
        metrics_.frame_count = 0;
        metrics_.last_update = now;
    }
}

PerformanceMetrics InferenceEngine::get_metrics() const {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    return metrics_;
}

void InferenceEngine::reset_metrics() {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    metrics_ = PerformanceMetrics{};
    metrics_.last_update = std::chrono::steady_clock::now();
}

void InferenceEngine::set_confidence_threshold(float threshold) {
    config_.confidence_threshold = std::clamp(threshold, 0.0f, 1.0f);
}

void InferenceEngine::set_nms_threshold(float threshold) {
    config_.nms_threshold = std::clamp(threshold, 0.0f, 1.0f);
}

void InferenceEngine::set_input_size(const cv::Size& size) {
    config_.input_size = size;
    // Reallocate buffers if size changed
    const size_t input_size = size.width * size.height * 3;
    input_tensor_data_.reserve(input_size);
    preprocessed_buffer_.create(size.height, size.width, CV_32FC3);
}

void InferenceEngine::cleanup() {
    session_.reset();
    hardware_accel_.reset();
    input_names_.clear();
    output_names_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
    input_tensor_data_.clear();
    preprocessed_buffer_.release();
}

// Utility function implementations
namespace utils {

cv::Mat letterbox_resize(const cv::Mat& image, const cv::Size& target_size, cv::Scalar fill_color) {
    float scale = std::min(static_cast<float>(target_size.width) / image.cols,
                          static_cast<float>(target_size.height) / image.rows);
    
    cv::Size new_size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale));
    cv::Mat resized;
    cv::resize(image, resized, new_size);
    
    cv::Mat letterboxed(target_size, image.type(), fill_color);
    
    int top = (target_size.height - new_size.height) / 2;
    int left = (target_size.width - new_size.width) / 2;
    
    resized.copyTo(letterboxed(cv::Rect(left, top, new_size.width, new_size.height)));
    
    return letterboxed;
}

std::vector<DetectionResult> apply_nms(const std::vector<DetectionResult>& detections, float nms_threshold) {
    if (detections.empty()) return {};
    
    // Sort by confidence
    std::vector<DetectionResult> sorted_detections = detections;
    std::sort(sorted_detections.begin(), sorted_detections.end(),
              [](const DetectionResult& a, const DetectionResult& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> suppressed(sorted_detections.size(), false);
    std::vector<DetectionResult> results;
    
    for (size_t i = 0; i < sorted_detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        results.push_back(sorted_detections[i]);
        
        for (size_t j = i + 1; j < sorted_detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            // Calculate IoU
            cv::Rect intersection = sorted_detections[i].bbox & sorted_detections[j].bbox;
            float intersection_area = intersection.area();
            
            float union_area = sorted_detections[i].bbox.area() + 
                              sorted_detections[j].bbox.area() - intersection_area;
            
            float iou = intersection_area / union_area;
            
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return results;
}

void draw_detections(cv::Mat& image, const std::vector<DetectionResult>& detections) {
    for (const auto& detection : detections) {
        // Draw bounding box
        cv::rectangle(image, detection.bbox, cv::Scalar(0, 255, 0), 2);
        
        // Draw label
        std::string label = detection.class_name + " " + 
                           std::to_string(static_cast<int>(detection.confidence * 100)) + "%";
        
        int baseline;
        cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        
        cv::Point label_pos(detection.bbox.x, detection.bbox.y - 10);
        cv::rectangle(image, label_pos, 
                     cv::Point(label_pos.x + text_size.width, label_pos.y - text_size.height - baseline),
                     cv::Scalar(0, 255, 0), cv::FILLED);
        
        cv::putText(image, label, label_pos, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                   cv::Scalar(0, 0, 0), 1);
    }
}

std::string get_class_name(int class_id) {
    if (class_id >= 0 && class_id < CLASS_NAMES.size()) {
        return CLASS_NAMES[class_id];
    }
    return "unknown";
}

} // namespace utils

// Exception implementations
BikeGuardException::BikeGuardException(ErrorCode code, const std::string& message)
    : error_code_(code), message_(message) {}

const char* BikeGuardException::what() const noexcept {
    return message_.c_str();
}

ErrorCode BikeGuardException::get_error_code() const noexcept {
    return error_code_;
}

} // namespace BikeGuard
