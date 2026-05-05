#include "bikeguard_engine.hpp"
#include <algorithm>
#include <numeric>
#include <iostream>
#include <format>

namespace BikeGuard {

// Static class name mappings for helmet detection
constexpr std::array<std::string_view, 5> CLASS_NAMES = {
    "person", "bicycle", "motorcycle", "helmet", "no_helmet"
};

BikeGuardEngine::BikeGuardEngine() 
    : env_(std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BikeGuard")),
      session_options_(std::make_unique<Ort::SessionOptions>()),
      memory_info_(std::make_unique<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(
          OrtArenaAllocator, OrtMemTypeDefault))) {
    
    // Initialize performance metrics
    metrics_.fps.store(0.0);
    metrics_.inference_time_ms.store(0.0);
    metrics_.preprocessing_time_ms.store(0.0);
    metrics_.postprocessing_time_ms.store(0.0);
    metrics_.frame_count.store(0);
    metrics_.total_detections.store(0);
}

BikeGuardEngine::~BikeGuardEngine() noexcept {
    try {
        cleanup();
    } catch (...) {
        // Destructor should not throw
    }
}

bool BikeGuardEngine::initialize(const EngineConfig& config) {
    config_ = config;
    
    try {
        // Initialize DirectML if GPU is requested
        if (config_.use_gpu) {
            if (!initialize_directml()) {
                std::cout << "DirectML initialization failed, falling back to CPU\n";
                config_.use_gpu = false;
                gpu_available_.store(false);
            } else {
                gpu_available_.store(true);
                std::cout << "DirectML GPU acceleration enabled\n";
            }
        } else {
            gpu_available_.store(false);
        }
        
        // Setup session options
        setup_session_options();
        
        // Load model
        if (!load_model(config_.model_path)) {
            return false;
        }
        
        // Pre-allocate buffers for zero-allocation design
        const size_t input_size = config_.input_size.width * config_.input_size.height * 3;
        input_tensor_data_.reserve(input_size);
        preprocessed_buffer_.create(config_.input_size.height, config_.input_size.width, CV_32FC3);
        
        initialized_.store(true);
        return true;
        
    } catch (const std::exception& e) {
        throw BikeGuardException(ErrorCode::INFERENCE_FAILED, 
                               std::format("Engine initialization failed: {}", e.what()));
    }
}

bool BikeGuardEngine::initialize_directml() {
    try {
        // Create D3D11 device for DirectML
        D3D_FEATURE_LEVEL feature_levels[] = { D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_1 };
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            feature_levels, 2, D3D11_SDK_VERSION,
            &d3d_device_, nullptr, &d3d_context_);
        
        if (FAILED(hr)) {
            return false;
        }
        
        // Get DXGI adapter for GPU info
        hr = d3d_device_->GetAdapter(&dxgi_adapter_);
        if (FAILED(hr)) {
            return false;
        }
        
        // Get adapter description
        DXGI_ADAPTER_DESC desc;
        hr = dxgi_adapter_->GetDesc(&desc);
        if (FAILED(hr)) {
            return false;
        }
        
        std::wcout << L"Using GPU: " << desc.Description << L'\n';
        return true;
        
    } catch (...) {
        return false;
    }
}

void BikeGuardEngine::setup_session_options() {
    // Set thread configuration
    session_options_->SetIntraOpNumThreads(config_.intra_op_num_threads);
    session_options_->SetInterOpNumThreads(config_.inter_op_num_threads);
    
    // Enable graph optimization
    session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    
    // Memory arena settings
    session_options_->SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    
    // Add DirectML execution provider if available
    if (gpu_available_.load() && d3d_device_) {
        try {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(*session_options_, 0));
        } catch (const Ort::Exception& e) {
            std::cerr << "Failed to add DirectML provider: " << e.what() << '\n';
            gpu_available_.store(false);
        }
    }
}

bool BikeGuardEngine::load_model(const std::string& model_path) {
    try {
        session_ = std::make_unique<Ort::Session>(*env_, model_path.c_str(), *session_options_);
        
        // Load model metadata
        return load_model_metadata();
        
    } catch (const Ort::Exception& e) {
        throw BikeGuardException(ErrorCode::MODEL_LOAD_FAILED, 
                               std::format("Failed to load model '{}': {}", model_path, e.what()));
    }
}

bool BikeGuardEngine::load_model_metadata() {
    try {
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Get input information
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
        
        // Get output information
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
        
    } catch (const std::exception& e) {
        throw BikeGuardException(ErrorCode::MODEL_LOAD_FAILED, 
                               std::format("Failed to load model metadata: {}", e.what()));
    }
}

std::vector<DetectionResult> BikeGuardEngine::run_inference(const cv::Mat& input_frame) {
    if (!initialized_.load()) {
        throw BikeGuardException(ErrorCode::INFERENCE_FAILED, "Engine not initialized");
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Preprocessing (zero-allocation)
    auto preprocess_start = std::chrono::high_resolution_clock::now();
    auto input_span = preprocess_frame(input_frame);
    auto preprocess_end = std::chrono::high_resolution_clock::now();
    
    // Create input tensor (reuse pre-allocated data)
    std::vector<int64_t> input_shape = {1, 3, config_.input_size.height, config_.input_size.width};
    
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        *memory_info_, input_span.data(), input_span.size(), 
        input_shape.data(), input_shape.size());
    
    // Run inference
    auto inference_start = std::chrono::high_resolution_clock::now();
    std::vector<Ort::Value> output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 
        input_names_.size(), output_names_.data(), output_names_.size());
    auto inference_end = std::chrono::high_resolution_clock::now();
    
    // Postprocessing
    auto postprocess_start = std::chrono::high_resolution_clock::now();
    std::vector<DetectionResult> results;
    
    if (!output_tensors.empty()) {
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        std::span<const float> output_span(output_data, 
            std::accumulate(output_shape.begin(), output_shape.end(), 1LL, std::multiplies<>()));
        
        results = postprocess_results(output_span, output_shape);
    }
    
    auto postprocess_end = std::chrono::high_resolution_clock::now();
    
    // Apply vibration filtering if enabled
    if (vibration_filtering_enabled_.load() && vibration_filter_) {
        // Extract motion data from detections (simplified example)
        std::vector<float> motion_data;
        motion_data.reserve(results.size() * 4); // x, y, w, h for each detection
        
        for (const auto& detection : results) {
            motion_data.push_back(static_cast<float>(detection.bbox.x));
            motion_data.push_back(static_cast<float>(detection.bbox.y));
            motion_data.push_back(static_cast<float>(detection.bbox.width));
            motion_data.push_back(static_cast<float>(detection.bbox.height));
        }
        
        if (!motion_data.empty()) {
            auto filtered_motion = vibration_filter_->filter_frame(std::span<float>(motion_data));
            // Update detection results with filtered data (simplified)
            for (size_t i = 0; i < results.size() && i * 4 + 3 < filtered_motion.size(); ++i) {
                results[i].bbox.x = static_cast<int>(filtered_motion[i * 4]);
                results[i].bbox.y = static_cast<int>(filtered_motion[i * 4 + 1]);
                results[i].bbox.width = static_cast<int>(filtered_motion[i * 4 + 2]);
                results[i].bbox.height = static_cast<int>(filtered_motion[i * 4 + 3]);
            }
        }
    }
    
    // Update performance metrics
    double preprocess_time = std::chrono::duration<double, std::milli>(preprocess_end - preprocess_start).count();
    double inference_time = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();
    double postprocess_time = std::chrono::duration<double, std::milli>(postprocess_end - postprocess_start).count();
    
    update_performance_metrics(inference_time, preprocess_time, postprocess_time, results.size());
    
    return results;
}

std::span<float> BikeGuardEngine::preprocess_frame(const cv::Mat& frame) {
    // Zero-allocation preprocessing using pre-allocated buffer
    cv::resize(frame, preprocessed_buffer_, config_.input_size, 0, 0, cv::INTER_LINEAR);
    
    // Convert BGR to RGB and normalize to [0,1] in one pass
    cv::Mat rgb_image;
    cv::cvtColor(preprocessed_buffer_, rgb_image, cv::COLOR_BGR2RGB);
    rgb_image.convertTo(rgb_image, CV_32F, 1.0f/255.0f);
    
    // Ensure input_tensor_data_ has correct size
    const size_t required_size = config_.input_size.width * config_.input_size.height * 3;
    if (input_tensor_data_.size() != required_size) {
        input_tensor_data_.resize(required_size);
    }
    
    // Copy data to tensor buffer in NCHW format (zero-allocation)
    float* tensor_data = input_tensor_data_.data();
    const float* image_data = rgb_image.ptr<float>();
    
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < rgb_image.rows; ++h) {
            for (int w = 0; w < rgb_image.cols; ++w) {
                tensor_data[c * rgb_image.rows * rgb_image.cols + h * rgb_image.cols + w] = 
                    image_data[h * rgb_image.cols * 3 + w * 3 + c];
            }
        }
    }
    
    return std::span<float>(input_tensor_data_);
}

std::vector<DetectionResult> BikeGuardEngine::postprocess_results(
    std::span<const float> output_data, 
    const std::vector<int64_t>& output_shape) {
    
    std::vector<DetectionResult> detections;
    
    if (output_shape.size() < 3) return detections;
    
    // Assuming YOLOv8 output format: [batch, num_detections, 85]
    const int num_detections = static_cast<int>(output_shape[1]);
    const int num_classes = static_cast<int>(output_shape[2]) - 5;
    
    for (int i = 0; i < num_detections; ++i) {
        const float* detection = output_data.data() + i * output_shape[2];
        
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
            std::string_view class_name = get_class_name(best_class);
            
            detections.emplace_back(max_conf, best_class, bbox, class_name);
        }
    }
    
    // Apply Non-Maximum Suppression
    return apply_nms(detections, config_.nms_threshold);
}

void BikeGuardEngine::update_performance_metrics(
    double inference_time, 
    double preprocess_time, 
    double postprocess_time,
    size_t detection_count) noexcept {
    
    // Update atomic values
    metrics_.inference_time_ms.store(inference_time);
    metrics_.preprocessing_time_ms.store(preprocess_time);
    metrics_.postprocessing_time_ms.store(postprocess_time);
    
    size_t frame_count = metrics_.frame_count.fetch_add(1) + 1;
    metrics_.total_detections.fetch_add(detection_count);
    
    // Calculate FPS every 30 frames
    static thread_local auto last_fps_update = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration<double>(now - last_fps_update).count();
    
    if (duration >= 1.0) {
        double fps = frame_count / duration;
        metrics_.fps.store(fps);
        metrics_.frame_count.store(0);
        last_fps_update = now;
    }
}

std::vector<DetectionResult> BikeGuardEngine::apply_nms(
    std::vector<DetectionResult>& detections, 
    float nms_threshold) {
    
    if (detections.empty()) return {};
    
    // Sort by confidence
    std::sort(detections.begin(), detections.end(),
              [](const DetectionResult& a, const DetectionResult& b) {
                  return a.confidence > b.confidence;
              });
    
    std::vector<bool> suppressed(detections.size(), false);
    std::vector<DetectionResult> results;
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (suppressed[i]) continue;
        
        results.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            
            // Calculate IoU
            cv::Rect intersection = detections[i].bbox & detections[j].bbox;
            float intersection_area = static_cast<float>(intersection.area());
            
            float union_area = static_cast<float>(detections[i].bbox.area()) + 
                              static_cast<float>(detections[j].bbox.area()) - intersection_area;
            
            float iou = intersection_area / union_area;
            
            if (iou > nms_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return results;
}

std::string_view BikeGuardEngine::get_class_name(int class_id) noexcept {
    if (class_id >= 0 && class_id < static_cast<int>(CLASS_NAMES.size())) {
        return CLASS_NAMES[class_id];
    }
    return "unknown";
}

// Configuration methods
void BikeGuardEngine::set_confidence_threshold(float threshold) noexcept {
    config_.confidence_threshold = std::clamp(threshold, 0.0f, 1.0f);
}

void BikeGuardEngine::set_nms_threshold(float threshold) noexcept {
    config_.nms_threshold = std::clamp(threshold, 0.0f, 1.0f);
}

void BikeGuardEngine::set_input_size(const cv::Size& size) {
    config_.input_size = size;
    // Reallocate buffers if size changed
    const size_t input_size = size.width * size.height * 3;
    input_tensor_data_.reserve(input_size);
    preprocessed_buffer_.create(size.height, size.width, CV_32FC3);
}

const EngineConfig& BikeGuardEngine::get_config() const noexcept {
    return config_;
}

// Performance monitoring
PerformanceMetrics BikeGuardEngine::get_metrics() const noexcept {
    return metrics_;
}

void BikeGuardEngine::reset_metrics() noexcept {
    metrics_.fps.store(0.0);
    metrics_.inference_time_ms.store(0.0);
    metrics_.preprocessing_time_ms.store(0.0);
    metrics_.postprocessing_time_ms.store(0.0);
    metrics_.frame_count.store(0);
    metrics_.total_detections.store(0);
}

// Camera management
void BikeGuardEngine::set_camera(std::unique_ptr<ICameraCapture> camera) {
    camera_ = std::move(camera);
}

ICameraCapture* BikeGuardEngine::get_camera() const noexcept {
    return camera_.get();
}

// Vibration filtering
void BikeGuardEngine::set_vibration_filter(std::unique_ptr<IVibrationFilter> filter) {
    vibration_filter_ = std::move(filter);
}

void BikeGuardEngine::enable_vibration_filtering(bool enable) noexcept {
    vibration_filtering_enabled_.store(enable);
}

// Temporal smoothing state machine implementation
ComplianceState BikeGuardEngine::update_state(const DetectionResult& current_frame, float current_vibration_hz) {
    if (!config_.enable_temporal_smoothing) {
        return current_state_.load();
    }
    
    std::lock_guard<std::mutex> lock(state_mutex_);
    
    // Get current timestamp
    auto now = std::chrono::steady_clock::now();
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    
    // Vibration override: freeze state if vibration exceeds threshold
    if (current_vibration_hz > config_.vibration_override_threshold_hz) {
        state_frozen_.store(true);
        return current_state_.load();
    } else {
        state_frozen_.store(false);
    }
    
    // Determine detection presence for current frame
    DetectionPresence current_presence = NO_DETECTION;
    if (current_frame.confidence > config_.confidence_threshold) {
        if (current_frame.class_id == 3) { // helmet class
            current_presence = HELMET_DETECTED;
        } else if (current_frame.class_id == 0) { // person class
            current_presence = RIDER_DETECTED;
        }
    }
    
    // Add frame to history
    frame_history_.add_frame(current_presence, current_frame.confidence, timestamp);
    
    ComplianceState new_state = current_state_.load();
    
    // State transition logic with hysteresis
    switch (current_state_.load()) {
        case ComplianceState::STATE_COMPLIANT:
            // Transition to VIOLATION only after N consecutive frames with NO_HELMET while RIDER present
            if (frame_history_.get_recent_count(RIDER_DETECTED, config_.violation_threshold_frames) >= config_.violation_threshold_frames &&
                frame_history_.get_recent_count(HELMET_DETECTED, config_.violation_threshold_frames) == 0) {
                new_state = ComplianceState::STATE_VIOLATION;
                last_state_change_.store(timestamp);
            }
            break;
            
        case ComplianceState::STATE_VIOLATION:
            // Transition back to COMPLIANT after M consecutive frames with HELMET detected
            if (frame_history_.get_recent_count(HELMET_DETECTED, config_.compliance_threshold_frames) >= config_.compliance_threshold_frames) {
                new_state = ComplianceState::STATE_COMPLIANT;
                last_state_change_.store(timestamp);
            }
            break;
            
        case ComplianceState::STATE_TRANSITIONING:
            // Handle transitional state (if needed)
            if (frame_history_.get_recent_count(HELMET_DETECTED, 1) > 0) {
                new_state = ComplianceState::STATE_COMPLIANT;
            } else if (frame_history_.get_recent_count(RIDER_DETECTED, 3) >= 3) {
                new_state = ComplianceState::STATE_VIOLATION;
            }
            last_state_change_.store(timestamp);
            break;
    }
    
    // Update stable detections based on new state
    if (new_state != current_state_.load()) {
        current_state_.store(new_state);
        
        // Update stable detections when state changes
        if (new_state == ComplianceState::STATE_COMPLIANT) {
            // Find the most recent helmet detection
            for (size_t i = 0; i < frame_history_.frame_count; ++i) {
                size_t index = (frame_history_.current_index - 1 - i + FrameHistory::HISTORY_SIZE) % FrameHistory::HISTORY_SIZE;
                if (frame_history_.detections[index] == HELMET_DETECTED) {
                    // Create stable detection from history
                    stable_detections_.clear();
                    stable_detections_.emplace_back(frame_history_.confidences[index], 3, current_frame.bbox, "helmet");
                    break;
                }
            }
        } else {
            // Clear stable detections for violation state
            stable_detections_.clear();
            
            // Submit telemetry event for violation (non-blocking)
            if (telemetry_enabled_.load() && telemetry_dispatcher_) {
                TelemetryEvent event("HELMET_VIOLATION", 3, current_vibration_hz, current_frame.confidence);
                telemetry_dispatcher_->submit_event(event);
            }
        }
    }
    
    return new_state;
}

ComplianceState BikeGuardEngine::get_compliance_state() const noexcept {
    return current_state_.load();
}

std::vector<DetectionResult> BikeGuardEngine::get_stable_detections() const noexcept {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return stable_detections_;
}

void BikeGuardEngine::reset_state_machine() noexcept {
    std::lock_guard<std::mutex> lock(state_mutex_);
    frame_history_.reset();
    current_state_.store(ComplianceState::STATE_COMPLIANT);
    stable_detections_.clear();
    state_frozen_.store(false);
    last_state_change_.store(0);
}

// Telemetry dispatcher integration
void BikeGuardEngine::set_telemetry_dispatcher(std::unique_ptr<TelemetryDispatcher> dispatcher) {
    telemetry_dispatcher_ = std::move(dispatcher);
}

void BikeGuardEngine::enable_telemetry(bool enable) noexcept {
    telemetry_enabled_.store(enable);
}

bool BikeGuardEngine::is_telemetry_enabled() const noexcept {
    return telemetry_enabled_.load();
}

// Calibration integration
bool BikeGuardEngine::load_calibration_config() {
    try {
        std::ifstream file("config/calibration_config.json");
        if (!file.is_open()) {
            std::cout << "No calibration config found, using defaults.\n";
            return false;
        }
        
        nlohmann::json json;
        file >> json;
        
        if (json.contains("exclusion_y_coordinate") && json.contains("camera_height")) {
            int y_coord = json["exclusion_y_coordinate"];
            int height = json["camera_height"];
            calibration_loaded_ = true;
            
            // Create exclusion zone rectangle
            exclusion_zone_ = cv::Rect(0, y_coord, config_.input_size.width, height - y_coord);
            
            std::cout << "Calibration loaded: Exclusion zone at Y=" << y_coord << '\n';
            return true;
        }
        
        return false;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to load calibration config: " << e.what() << '\n';
        return false;
    }
}

void BikeGuardEngine::apply_calibration_exclusion(cv::Mat& frame) const {
    if (!calibration_loaded_ || exclusion_zone_.area() == 0) {
        return;
    }
    
    // Apply exclusion zone - set to black to prevent false detections
    cv::Mat roi = frame(exclusion_zone_);
    roi.setTo(cv::Scalar(0, 0, 0));
}

cv::Rect BikeGuardEngine::get_exclusion_zone() const {
    return exclusion_zone_;
}

// GPU management
bool BikeGuardEngine::is_gpu_available() const noexcept {
    return gpu_available_.load();
}

std::string BikeGuardEngine::get_gpu_info() const noexcept {
    if (!gpu_available_.load() || !dxgi_adapter_) {
        return "GPU not available";
    }
    
    try {
        DXGI_ADAPTER_DESC desc;
        HRESULT hr = dxgi_adapter_->GetDesc(&desc);
        if (FAILED(hr)) {
            return "Failed to get GPU info";
        }
        
        // Convert wide string to narrow string
        std::wstring wide_desc(desc.Description);
        std::string narrow_desc(wide_desc.begin(), wide_desc.end());
        
        return std::format("GPU: {} (VRAM: {}MB)", 
                          narrow_desc, 
                          desc.DedicatedVideoMemory / (1024 * 1024));
    } catch (...) {
        return "Error getting GPU info";
    }
}

void BikeGuardEngine::cleanup() {
    session_.reset();
    session_options_.reset();
    memory_info_.reset();
    d3d_context_.Reset();
    d3d_device_.Reset();
    dxgi_adapter_.Reset();
    
    input_names_.clear();
    output_names_.clear();
    input_shapes_.clear();
    output_shapes_.clear();
    input_tensor_data_.clear();
    output_tensor_data_.clear();
    preprocessed_buffer_.release();
    
    camera_.reset();
    vibration_filter_.reset();
    
    initialized_.store(false);
    gpu_available_.store(false);
}

// Exception implementations
BikeGuardException::BikeGuardException(ErrorCode code, std::string message)
    : error_code_(code), message_(std::move(message)) {}

const char* BikeGuardException::what() const noexcept {
    return message_.c_str();
}

BikeGuardException::ErrorCode BikeGuardException::get_error_code() const noexcept {
    return error_code_;
}

} // namespace BikeGuard
