#pragma once

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <string>
#include <memory>
#include <vector>

namespace BikeGuard {

// Telemetry event structure for violation reporting
struct TelemetryEvent {
    std::string event_type;
    int severity;
    std::string timestamp;  // ISO-8601 format
    struct Location {
        double lat;
        double lng;
    } location;
    struct Metrics {
        float vibration_hz;
        float confidence;
    } metrics;
    
    TelemetryEvent() = default;
    TelemetryEvent(const std::string& type, int sev, float vib, float conf)
        : event_type(type), severity(sev), timestamp(get_iso_timestamp()) {
        location.lat = 28.6692;  // Ghaziabad coordinates
        location.lng = 77.4538;
        metrics.vibration_hz = vib;
        metrics.confidence = conf;
    }
    
private:
    static std::string get_iso_timestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        
        std::ostringstream oss;
        oss << std::put_time(std::gmtime(&time_t), "%Y-%m-%dT%H:%M:%S");
        oss << '.' << std::setfill('0') << std::setw(3) << ms.count() << 'Z';
        return oss.str();
    }
};

// Asynchronous telemetry dispatcher for non-blocking HTTP requests
class TelemetryDispatcher {
public:
    struct Config {
        std::string shadowmap_url = "http://localhost:5000/api/events";
        std::chrono::seconds retry_interval{5};
        std::chrono::seconds request_timeout{10};
        size_t max_cache_size = 50;
        size_t worker_thread_sleep_ms = 100;
        bool enable_compression = true;
        std::string user_agent = "BikeGuard/1.0";
    };
    
    TelemetryDispatcher();
    explicit TelemetryDispatcher(const Config& config);
    ~TelemetryDispatcher();
    
    // Non-blocking event submission
    auto submit_event(const TelemetryEvent& event) -> void;
    
    // Configuration methods
    auto set_config(const Config& config) -> void;
    auto get_config() const -> Config;
    
    // Control methods
    auto start() -> bool;
    auto stop() -> void;
    auto is_running() const -> bool;
    
    // Statistics
    auto get_events_submitted() const -> size_t;
    auto get_events_sent() const -> size_t;
    auto get_events_failed() const -> size_t;
    auto get_cache_size() const -> size_t;

private:
    // Worker thread function
    auto worker_loop() -> void;
    
    // HTTP request handling
    auto send_event(const TelemetryEvent& event) -> bool;
    auto retry_cached_events() -> void;
    
    // JSON serialization
    auto serialize_event(const TelemetryEvent& event) -> std::string;
    
    // Configuration
    Config config_;
    
    // Thread management
    std::jthread worker_thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> should_stop_{false};
    
    // Event queue and cache
    std::queue<TelemetryEvent> event_queue_;
    std::queue<TelemetryEvent> retry_cache_;
    mutable std::mutex queue_mutex_;
    mutable std::mutex cache_mutex_;
    std::condition_variable queue_cv_;
    
    // Statistics
    std::atomic<size_t> events_submitted_{0};
    std::atomic<size_t> events_sent_{0};
    std::atomic<size_t> events_failed_{0};
    
    // Network state
    std::atomic<bool> network_available_{true};
    std::chrono::steady_clock::time_point last_retry_time_;
    
    // Utility methods
    auto is_cache_full() const -> bool;
    auto cleanup_old_cache() -> void;
    auto test_network_connectivity() -> bool;
};

// Factory function
auto create_telemetry_dispatcher(const TelemetryDispatcher::Config& config = {}) 
    -> std::unique_ptr<TelemetryDispatcher>;

} // namespace BikeGuard
