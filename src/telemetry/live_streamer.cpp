#include "live_streamer.hpp"
#include <iostream>
#include <format>
#include <chrono>

#ifdef _WIN32
#pragma comment(lib, "Ws2_32.lib")
#endif

namespace BikeGuard {

LiveStreamer::~LiveStreamer() {
    stop();
}

auto LiveStreamer::initialize(const StreamerConfig& config) -> bool {
    try {
        config_ = config;
        #ifdef _WIN32
        WSADATA wsaData;
        int res = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (res != 0) {
            std::cerr << std::format("WSAStartup failed: {}\n", res);
            return false;
        }
        #endif
        
        initialized_ = true;
        return true;
    } catch (const std::exception& e) {
        std::cerr << std::format("LiveStreamer init error: {}\n", e.what());
        return false;
    }
}

auto LiveStreamer::start() -> bool {
    if (!initialized_ || running_) {
        return false;
    }

    #ifdef _WIN32
    server_socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (server_socket_ == INVALID_SOCKET) {
        std::cerr << "Error creating server socket\n";
        return false;
    }

    // Allow socket reuse
    char opt = 1;
    setsockopt(server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(static_cast<u_short>(config_.port));

    if (bind(server_socket_, reinterpret_cast<sockaddr*>(&server_addr), sizeof(server_addr)) == SOCKET_ERROR) {
        std::cerr << std::format("Socket bind failed on port {}\n", config_.port);
        closesocket(server_socket_);
        return false;
    }

    if (listen(server_socket_, SOMAXCONN) == SOCKET_ERROR) {
        std::cerr << "Socket listen failed\n";
        closesocket(server_socket_);
        return false;
    }
    #endif

    running_ = true;
    server_thread_ = std::jthread([this]() { server_loop(); });
    std::cout << std::format("Live Streamer HTTP Server listening on http://localhost:{}\n", config_.port);
    return true;
}

auto LiveStreamer::stop() -> void {
    if (!running_) {
        return;
    }
    running_ = false;

    #ifdef _WIN32
    if (server_socket_ != INVALID_SOCKET) {
        closesocket(server_socket_);
        server_socket_ = INVALID_SOCKET;
    }
    WSACleanup();
    #endif
}

auto LiveStreamer::stream_frame(const cv::Mat& frame) -> void {
    if (!running_ || frame.empty()) {
        return;
    }

    try {
        std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, config_.jpeg_quality};
        std::vector<uchar> buffer;
        cv::imencode(".jpg", frame, buffer, params);

        std::lock_guard<std::mutex> lock(frame_mutex_);
        jpeg_buffer_ = std::move(buffer);
    } catch (const std::exception& e) {
        std::cerr << std::format("Error encoding streaming frame: {}\n", e.what());
    }
}

auto LiveStreamer::update_telemetry(std::string_view status_json) -> void {
    std::lock_guard<std::mutex> lock(telemetry_mutex_);
    current_telemetry_json_ = std::string(status_json);
}

auto LiveStreamer::is_running() const -> bool {
    return running_;
}

auto LiveStreamer::server_loop() -> void {
    #ifdef _WIN32
    while (running_) {
        sockaddr_in client_addr{};
        int client_len = sizeof(client_addr);
        SOCKET client_socket = accept(server_socket_, reinterpret_cast<sockaddr*>(&client_addr), &client_len);
        
        if (client_socket == INVALID_SOCKET) {
            if (running_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
            continue;
        }

        // Spawn detached worker thread for client request handling
        std::thread([this, client_socket]() {
            handle_client(static_cast<int>(client_socket));
            closesocket(static_cast<SOCKET>(client_socket));
        }).detach();
    }
    #endif
}

auto LiveStreamer::handle_client(int client_socket) -> void {
    #ifdef _WIN32
    SOCKET sock = static_cast<SOCKET>(client_socket);
    char request_buf[1024] = {0};
    int bytes_read = recv(sock, request_buf, sizeof(request_buf) - 1, 0);
    if (bytes_read <= 0) {
        return;
    }

    std::string_view request(request_buf, bytes_read);

    if (request.find("GET /api/status") != std::string_view::npos) {
        std::string json_data;
        {
            std::lock_guard<std::mutex> lock(telemetry_mutex_);
            json_data = current_telemetry_json_;
        }
        std::string header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
        send_http_response(client_socket, header, json_data);
    } 
    else if (request.find("GET /stream") != std::string_view::npos) {
        std::string header = "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
        send(sock, header.data(), static_cast<int>(header.size()), 0);

        while (running_) {
            std::vector<uchar> buffer_copy;
            {
                std::lock_guard<std::mutex> lock(frame_mutex_);
                if (!jpeg_buffer_.empty()) {
                    buffer_copy = jpeg_buffer_;
                }
            }

            if (!buffer_copy.empty()) {
                std::string frame_header = std::format("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", buffer_copy.size());
                send(sock, frame_header.data(), static_cast<int>(frame_header.size()), 0);
                send(sock, reinterpret_cast<const char*>(buffer_copy.data()), static_cast<int>(buffer_copy.size()), 0);
                send(sock, "\r\n", 2, 0);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1000 / config_.max_fps));
        }
    } 
    else {
        // Serve Web Dashboard HTML
        std::string html = get_dashboard_html();
        std::string header = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n";
        send_http_response(client_socket, header, html);
    }
    #endif
}

auto LiveStreamer::send_http_response(int socket, std::string_view header, std::string_view body) -> void {
    #ifdef _WIN32
    SOCKET sock = static_cast<SOCKET>(socket);
    send(sock, header.data(), static_cast<int>(header.size()), 0);
    send(sock, body.data(), static_cast<int>(body.size()), 0);
    #endif
}

auto LiveStreamer::get_dashboard_html() -> std::string {
    return R"(<!DOCTYPE html>
<html>
<head>
    <title>BikeGuard v1.1.0 Enterprise AI Dashboard</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }
        .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #334155; padding-bottom: 15px; margin-bottom: 20px; }
        .header h1 { margin: 0; color: #38bdf8; font-size: 28px; }
        .badge { background: #10b981; color: #000; padding: 5px 12px; border-radius: 20px; font-weight: bold; font-size: 14px; }
        .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 15px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3); }
        .card h2 { margin-top: 0; color: #cbd5e1; font-size: 18px; border-bottom: 1px solid #334155; padding-bottom: 8px; }
        .stream-container { text-align: center; background: #000; border-radius: 8px; overflow: hidden; }
        img { max-width: 100%; height: auto; display: block; margin: 0 auto; }
        pre { background: #0f172a; padding: 12px; border-radius: 6px; overflow-x: auto; color: #a7f3d0; font-family: monospace; font-size: 13px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ BikeGuard Enterprise AI Ecosystem</h1>
        <span class="badge">v1.1.0 GA LIVE</span>
    </div>
    <div class="grid">
        <div class="card">
            <h2>Live Road Telemetry & ALPR Stream</h2>
            <div class="stream-container">
                <img src="/stream" alt="Live Video Feed Offline" />
            </div>
        </div>
        <div class="card">
            <h2>Real-Time Compliance & e-Challan Audit</h2>
            <pre id="status-box">Loading live telemetry...</pre>
        </div>
    </div>
    <script>
        setInterval(() => {
            fetch('/api/status')
                .then(res => res.json())
                .then(data => {
                    document.getElementById('status-box').textContent = JSON.stringify(data, null, 2);
                })
                .catch(err => {
                    document.getElementById('status-box').textContent = "Stream Disconnected / Offline";
                });
        }, 1000);
    </script>
</body>
</html>)";
}

auto create_live_streamer() -> std::unique_ptr<LiveStreamer> {
    auto streamer = std::make_unique<LiveStreamer>();
    if (streamer->initialize()) {
        return streamer;
    }
    return nullptr;
}

} // namespace BikeGuard
