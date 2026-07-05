#pragma once

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "bikeguard_core.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>

namespace BikeGuard {

// Embedded HTTP MJPEG Video & REST Telemetry Streaming Server
class LiveStreamer {
public:
    struct StreamerConfig {
        int port{8080};
        int max_fps{30};
        int jpeg_quality{80};
        bool enable_rest_api{true};
    };

    LiveStreamer() = default;
    ~LiveStreamer();

    auto initialize(const StreamerConfig& config = StreamerConfig{}) -> bool;
    auto start() -> bool;
    auto stop() -> void;
    auto stream_frame(const cv::Mat& frame) -> void;
    auto update_telemetry(std::string_view status_json) -> void;
    auto is_running() const -> bool;

private:
    bool initialized_{false};
    std::atomic<bool> running_{false};
    StreamerConfig config_;
    
    #ifdef _WIN32
    SOCKET server_socket_{INVALID_SOCKET};
    #endif
    
    std::jthread server_thread_;
    std::mutex frame_mutex_;
    std::mutex telemetry_mutex_;
    
    std::vector<uchar> jpeg_buffer_;
    std::string current_telemetry_json_{"{}"};
    
    auto server_loop() -> void;
    auto handle_client(int client_socket) -> void;
    auto send_http_response(int socket, std::string_view header, std::string_view body) -> void;
    auto get_dashboard_html() -> std::string;
};

auto create_live_streamer() -> std::unique_ptr<LiveStreamer>;

} // namespace BikeGuard
