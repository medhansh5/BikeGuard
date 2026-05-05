#include "telemetry_dispatcher.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>

namespace BikeGuard {

TelemetryDispatcher::TelemetryDispatcher() : config_{} {
    last_retry_time_ = std::chrono::steady_clock::now();
}

TelemetryDispatcher::TelemetryDispatcher(const Config& config) : config_(config) {
    last_retry_time_ = std::chrono::steady_clock::now();
}

TelemetryDispatcher::~TelemetryDispatcher() {
    stop();
}

auto TelemetryDispatcher::submit_event(const TelemetryEvent& event) -> void {
    if (!running_.load()) {
        return; // Silently ignore if dispatcher is not running
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        event_queue_.push(event);
    }
    
    events_submitted_.fetch_add(1);
    queue_cv_.notify_one();
}

auto TelemetryDispatcher::set_config(const Config& config) -> void {
    config_ = config;
}

auto TelemetryDispatcher::get_config() const -> Config {
    return config_;
}

auto TelemetryDispatcher::start() -> bool {
    if (running_.load()) {
        return true; // Already running
    }
    
    try {
        should_stop_.store(false);
        worker_thread_ = std::jthread(&TelemetryDispatcher::worker_loop, this);
        running_.store(true);
        
        std::cout << "TelemetryDispatcher started successfully\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Failed to start TelemetryDispatcher: " << e.what() << '\n';
        return false;
    }
}

auto TelemetryDispatcher::stop() -> void {
    if (!running_.load()) {
        return; // Already stopped
    }
    
    should_stop_.store(true);
    queue_cv_.notify_all();
    
    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    
    running_.store(false);
    std::cout << "TelemetryDispatcher stopped\n";
}

auto TelemetryDispatcher::is_running() const -> bool {
    return running_.load();
}

auto TelemetryDispatcher::get_events_submitted() const -> size_t {
    return events_submitted_.load();
}

auto TelemetryDispatcher::get_events_sent() const -> size_t {
    return events_sent_.load();
}

auto TelemetryDispatcher::get_events_failed() const -> size_t {
    return events_failed_.load();
}

auto TelemetryDispatcher::get_cache_size() const -> size_t {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return retry_cache_.size();
}

auto TelemetryDispatcher::worker_loop() -> void {
    while (!should_stop_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        
        // Wait for events or stop signal
        queue_cv_.wait_for(lock, std::chrono::milliseconds(config_.worker_thread_sleep_ms), 
                          [this] { return !event_queue_.empty() || should_stop_.load(); });
        
        if (should_stop_.load()) {
            break;
        }
        
        // Process all available events
        while (!event_queue_.empty() && !should_stop_.load()) {
            TelemetryEvent event = event_queue_.front();
            event_queue_.pop();
            lock.unlock();
            
            // Send event
            bool success = send_event(event);
            if (success) {
                events_sent_.fetch_add(1);
                
                // Try to send cached events after successful send
                retry_cached_events();
            } else {
                events_failed_.fetch_add(1);
                
                // Cache failed event for retry
                if (!is_cache_full()) {
                    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
                    retry_cache_.push(event);
                }
            }
            
            lock.lock();
        }
        
        // Periodic retry of cached events
        if (!should_stop_.load()) {
            lock.unlock();
            
            auto now = std::chrono::steady_clock::now();
            if (now - last_retry_time_ >= config_.retry_interval) {
                retry_cached_events();
                last_retry_time_ = now;
            }
        }
    }
}

auto TelemetryDispatcher::send_event(const TelemetryEvent& event) -> bool {
    try {
        // Test network connectivity before attempting send
        if (!test_network_connectivity()) {
            network_available_.store(false);
            return false;
        }
        
        network_available_.store(true);
        
        // Serialize event to JSON
        std::string json_payload = serialize_event(event);
        
        // Configure HTTP request
        cpr::Url url(config_.shadowmap_url);
        cpr::Header headers{
            {"Content-Type", "application/json"},
            {"User-Agent", config_.user_agent}
        };
        
        if (config_.enable_compression) {
            headers.emplace("Accept-Encoding", "gzip, deflate");
            headers.emplace("Content-Encoding", "gzip");
        }
        
        cpr::Body body(json_payload);
        cpr::Timeout config_.request_timeout;
        
        // Send POST request (non-blocking)
        auto response = cpr::Post(url, headers, body, config_.request_timeout);
        
        return response.status_code >= 200 && response.status_code < 300;
        
    } catch (const std::exception& e) {
        std::cerr << "HTTP request failed: " << e.what() << '\n';
        return false;
    }
}

auto TelemetryDispatcher::retry_cached_events() -> void {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    if (retry_cache_.empty()) {
        return;
    }
    
    std::queue<TelemetryEvent> temp_cache;
    size_t retry_count = 0;
    
    while (!retry_cache_.empty()) {
        TelemetryEvent event = retry_cache_.front();
        retry_cache_.pop();
        
        if (send_event(event)) {
            events_sent_.fetch_add(1);
            retry_count++;
        } else {
            temp_cache.push(event);
        }
    }
    
    // Put failed events back in cache
    retry_cache_ = std::move(temp_cache);
    
    if (retry_count > 0) {
        std::cout << "Retried " << retry_count << " cached events, " 
                  << retry_cache_.size() << " remaining in cache\n";
    }
}

auto TelemetryDispatcher::serialize_event(const TelemetryEvent& event) -> std::string {
    nlohmann::json json;
    json["event_type"] = event.event_type;
    json["severity"] = event.severity;
    json["timestamp"] = event.timestamp;
    
    json["location"]["lat"] = event.location.lat;
    json["location"]["lng"] = event.location.lng;
    
    json["metrics"]["vibration_hz"] = event.metrics.vibration_hz;
    json["metrics"]["confidence"] = event.metrics.confidence;
    
    return json.dump();
}

auto TelemetryDispatcher::is_cache_full() const -> bool {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    return retry_cache_.size() >= config_.max_cache_size;
}

auto TelemetryDispatcher::cleanup_old_cache() -> void {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    while (retry_cache_.size() > config_.max_cache_size) {
        retry_cache_.pop();
    }
}

auto TelemetryDispatcher::test_network_connectivity() -> bool {
    try {
        // Simple connectivity test - try to connect to the ShadowMap endpoint
        auto response = cpr::Get(cpr::Url{config_.shadowmap_url}, 
                                cpr::Timeout{std::chrono::seconds{2}});
        
        // Any response means the server is reachable
        return response.status_code != 0;
        
    } catch (const std::exception&) {
        return false;
    }
}

// Factory function implementation
auto create_telemetry_dispatcher(const TelemetryDispatcher::Config& config) 
    -> std::unique_ptr<TelemetryDispatcher> {
    
    auto dispatcher = std::make_unique<TelemetryDispatcher>(config);
    
    if (!dispatcher->start()) {
        return nullptr;
    }
    
    return dispatcher;
}

} // namespace BikeGuard
