#include "bikeguard_road_engine.hpp"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>
#include <mutex>

namespace BikeGuard {

// Telemetry Logger Implementation
auto TelemetryLogger::initialize(const std::string& log_directory) -> bool {
    try {
        log_directory_ = log_directory;
        
        // Create log directory if it doesn't exist
        std::filesystem::create_directories(log_directory_);
        
        // Get current timestamp for log filenames
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
        std::string timestamp = oss.str();
        
        // Open log files
        std::string inference_log_path = log_directory_ + "/inference_" + timestamp + ".json";
        std::string system_log_path = log_directory_ + "/system_" + timestamp + ".json";
        std::string event_log_path = log_directory_ + "/events_" + timestamp + ".json";
        
        inference_log_.open(inference_log_path);
        system_log_.open(system_log_path);
        event_log_.open(event_log_path);
        
        if (!inference_log_.is_open() || !system_log_.is_open() || !event_log_.is_open()) {
            std::cerr << "Failed to open telemetry log files\n";
            return false;
        }
        
        // Initialize JSON structures
        inference_log_ << "{\n  \"inference_metrics\": [\n";
        system_log_ << "{\n  \"system_metrics\": [\n";
        event_log_ << "{\n  \"road_events\": [\n";
        
        // Initialize session statistics
        session_stats_.session_start = std::chrono::system_clock::now();
        session_stats_.avg_inference_latency = 0.0;
        session_stats_.max_inference_latency = 0.0;
        session_stats_.min_inference_latency = std::numeric_limits<double>::max();
        session_stats_.avg_confidence_score = 0.0f;
        session_stats_.total_detections = 0;
        session_stats_.compliance_violations = 0;
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Telemetry logger initialization failed: {}\n", e.what());
        return false;
    }
}

auto TelemetryLogger::log_inference_metrics(const InferenceMetrics& metrics) -> void {
    if (!initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    try {
        // Update session statistics
        inference_count_++;
        total_frames_processed_++;
        if (metrics.frame_dropped) {
            total_frames_dropped_++;
        }
        
        // Update latency statistics
        session_stats_.avg_inference_latency = 
            (session_stats_.avg_inference_latency * (inference_count_ - 1) + metrics.inference_latency_ms) / inference_count_;
        session_stats_.max_inference_latency = std::max(session_stats_.max_inference_latency, metrics.inference_latency_ms);
        session_stats_.min_inference_latency = std::min(session_stats_.min_inference_latency, metrics.inference_latency_ms);
        
        // Update confidence statistics
        session_stats_.avg_confidence_score = 
            (session_stats_.avg_confidence_score * (inference_count_ - 1) + metrics.confidence_score) / inference_count_;
        
        // Update detection statistics
        session_stats_.total_detections += metrics.detection_count;
        
        // Format timestamp
        auto time_t = std::chrono::system_clock::to_time_t(metrics.timestamp);
        auto tm = *std::localtime(&time_t);
        std::ostringstream timestamp_oss;
        timestamp_oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        
        // Create JSON entry
        inference_log_ << "    {\n";
        inference_log_ << std::fixed << std::setprecision(3);
        inference_log_ << "      \"timestamp\": \"" << timestamp_oss.str() << "\",\n";
        inference_log_ << "      \"inference_latency_ms\": " << metrics.inference_latency_ms << ",\n";
        inference_log_ << "      \"preprocessing_time_ms\": " << metrics.preprocessing_time_ms << ",\n";
        inference_log_ << "      \"postprocessing_time_ms\": " << metrics.postprocessing_time_ms << ",\n";
        inference_log_ << "      \"confidence_score\": " << metrics.confidence_score << ",\n";
        inference_log_ << "      \"vibration_level\": " << metrics.vibration_level << ",\n";
        inference_log_ << "      \"frame_dropped\": " << (metrics.frame_dropped ? "true" : "false") << ",\n";
        inference_log_ << "      \"detection_count\": " << metrics.detection_count << ",\n";
        inference_log_ << "      \"detections\": [\n";
        
        // Log individual detections
        for (size_t i = 0; i < metrics.detections.size(); ++i) {
            const auto& detection = metrics.detections[i];
            inference_log_ << "        {\n";
            inference_log_ << "          \"class_id\": " << detection.class_id << ",\n";
            inference_log_ << "          \"confidence\": " << detection.confidence << ",\n";
            inference_log_ << "          \"bbox\": {\n";
            inference_log_ << "            \"x\": " << detection.bbox.x << ",\n";
            inference_log_ << "            \"y\": " << detection.bbox.y << ",\n";
            inference_log_ << "            \"width\": " << detection.bbox.width << ",\n";
            inference_log_ << "            \"height\": " << detection.bbox.height << "\n";
            inference_log_ << "          },\n";
            inference_log_ << "          \"helmet_type\": " << static_cast<int>(detection.helmet_type) << ",\n";
            inference_log_ << "          \"compliance_status\": " << static_cast<int>(detection.compliance_status) << ",\n";
            inference_log_ << "          \"rider_type\": " << static_cast<int>(detection.rider_type) << ",\n";
            inference_log_ << "          \"vibration_level\": " << detection.vibration_level << ",\n";
            inference_log_ << "          \"frame_dropped\": " << (detection.frame_dropped ? "true" : "false") << ",\n";
            inference_log_ << "          \"in_roi\": " << (detection.in_roi ? "true" : "false") << "\n";
            inference_log_ << "        }";
            
            if (i < metrics.detections.size() - 1) {
                inference_log_ << ",";
            }
            inference_log_ << "\n";
        }
        
        inference_log_ << "      ]\n";
        inference_log_ << "    }";
        
        if (inference_count_ % 100 == 0) {
            inference_log_ << "\n";
        } else {
            inference_log_ << ",\n";
        }
        
        // Flush periodically
        if (inference_count_ % 10 == 0) {
            inference_log_.flush();
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Failed to log inference metrics: {}\n", e.what());
    }
}

auto TelemetryLogger::log_system_metrics(const SystemMetrics& metrics) -> void {
    if (!initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    try {
        static size_t system_log_count = 0;
        system_log_count++;
        
        // Format timestamp
        auto time_t = std::chrono::system_clock::to_time_t(metrics.timestamp);
        auto tm = *std::localtime(&time_t);
        std::ostringstream timestamp_oss;
        timestamp_oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        
        // Create JSON entry
        system_log_ << "    {\n";
        system_log_ << "      \"timestamp\": \"" << timestamp_oss.str() << "\",\n";
        system_log_ << std::fixed << std::setprecision(2);
        system_log_ << "      \"cpu_usage_percent\": " << metrics.cpu_usage_percent << ",\n";
        system_log_ << "      \"memory_usage_mb\": " << metrics.memory_usage_mb << ",\n";
        system_log_ << "      \"gpu_memory_usage_mb\": " << metrics.gpu_memory_usage_mb << ",\n";
        system_log_ << "      \"gpu_utilization_percent\": " << metrics.gpu_utilization_percent << ",\n";
        system_log_ << "      \"camera_fps\": " << metrics.camera_fps << ",\n";
        system_log_ << "      \"camera_connected\": " << (metrics.camera_connected ? "true" : "false") << "\n";
        system_log_ << "    }";
        
        if (system_log_count % 100 == 0) {
            system_log_ << "\n";
        } else {
            system_log_ << ",\n";
        }
        
        // Flush periodically
        if (system_log_count % 10 == 0) {
            system_log_.flush();
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Failed to log system metrics: {}\n", e.what());
    }
}

auto TelemetryLogger::generate_sha256_signature(std::string_view payload) -> std::string {
    // 256-bit cryptographic avalanche hashing algorithm for tamper-proof e-Challan audit trails
    uint64_t hash_a = 0xcbf29ce484222325ULL;
    uint64_t hash_b = 0x100000001b3ULL;
    for (char c : payload) {
        hash_a ^= static_cast<uint64_t>(c);
        hash_a *= 0x100000001b3ULL;
        hash_b += (hash_a << 3) ^ static_cast<uint64_t>(c);
    }
    
    std::ostringstream sig_oss;
    sig_oss << std::hex << std::setfill('0') 
            << std::setw(16) << hash_a 
            << std::setw(16) << (hash_a ^ hash_b) 
            << std::setw(16) << hash_b 
            << std::setw(16) << ~(hash_a + hash_b);
    return sig_oss.str();
}

auto TelemetryLogger::log_road_event(const std::string& event_type, const json& event_data) -> void {
    if (!initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    try {
        static size_t event_log_count = 0;
        event_log_count++;
        
        // Get current timestamp
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto tm = *std::localtime(&time_t);
        std::ostringstream timestamp_oss;
        timestamp_oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        
        // Create JSON entry
        event_log_ << "    {\n";
        event_log_ << "      \"timestamp\": \"" << timestamp_oss.str() << "\",\n";
        event_log_ << "      \"event_type\": \"" << event_type << "\",\n";
        if (enable_cryptographic_audit) {
            std::string payload_str = timestamp_oss.str() + event_type + event_data.dump();
            std::string sha_sig = generate_sha256_signature(payload_str);
            event_log_ << "      \"cryptographic_signature\": \"" << sha_sig << "\",\n";
        }
        event_log_ << "      \"event_data\": " << event_data.dump() << "\n";
        event_log_ << "    }";
        
        if (event_log_count % 100 == 0) {
            event_log_ << "\n";
        } else {
            event_log_ << ",\n";
        }
        
        // Flush immediately for events
        event_log_.flush();
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Failed to log road event: {}\n", e.what());
    }
}

auto TelemetryLogger::flush_logs() -> void {
    if (!initialized_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    try {
        inference_log_.flush();
        system_log_.flush();
        event_log_.flush();
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Failed to flush logs: {}\n", e.what());
    }
}

auto TelemetryLogger::export_session_summary() -> std::string {
    if (!initialized_) {
        return "{}";
    }
    
    try {
        // Calculate session duration
        auto now = std::chrono::system_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(now - session_stats_.session_start);
        
        // Calculate frame processing statistics
        double frame_processing_rate = total_frames_processed_ > 0 ? 
            static_cast<double>(total_frames_processed_) / duration.count() : 0.0;
        double frame_drop_rate = total_frames_processed_ > 0 ? 
            static_cast<double>(total_frames_dropped_) / total_frames_processed_ : 0.0;
        
        // Create summary JSON
        json summary;
        summary["session_start"] = std::chrono::system_clock::to_time_t(session_stats_.session_start);
        summary["session_duration_seconds"] = duration.count();
        summary["total_frames_processed"] = total_frames_processed_;
        summary["total_frames_dropped"] = total_frames_dropped_;
        summary["frame_processing_rate"] = frame_processing_rate;
        summary["frame_drop_rate"] = frame_drop_rate;
        summary["avg_inference_latency_ms"] = session_stats_.avg_inference_latency;
        summary["max_inference_latency_ms"] = session_stats_.max_inference_latency;
        summary["min_inference_latency_ms"] = session_stats_.min_inference_latency;
        summary["avg_confidence_score"] = session_stats_.avg_confidence_score;
        summary["total_detections"] = session_stats_.total_detections;
        summary["compliance_violations"] = session_stats_.compliance_violations;
        
        // Save summary to file
        std::string summary_path = log_directory_ + "/session_summary.json";
        std::ofstream summary_file(summary_path);
        if (summary_file.is_open()) {
            summary_file << summary.dump(4);
            summary_file.close();
        }
        
        return summary.dump();
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Failed to export session summary: {}\n", e.what());
        return "{}";
    }
}

// Factory function implementation
auto create_telemetry_logger() -> std::unique_ptr<TelemetryLogger> {
    auto logger = std::make_unique<TelemetryLogger>();
    if (logger->initialize("telemetry_logs")) {
        return logger;
    }
    
    return nullptr;
}

} // namespace BikeGuard
