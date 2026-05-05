#include "bikeguard_engine.hpp"
#include <windows.h>
#include <psapi.h>
#include <dxgi1_2.h>
#include <d3d11.h>
#include <wrl/client.h>
#include <string>
#include <format>
#include <chrono>
#include <thread>

namespace BikeGuard {

using Microsoft::WRL::ComPtr;

// Windows Performance Monitor for system resources
class WindowsPerformanceMonitor {
public:
    struct SystemInfo {
        double cpu_usage_percent;
        size_t memory_usage_mb;
        size_t memory_total_mb;
        size_t gpu_memory_usage_mb;
        size_t gpu_memory_total_mb;
        double gpu_utilization_percent;
    };

    static auto get_system_info() -> SystemInfo {
        SystemInfo info{};
        
        // CPU usage
        info.cpu_usage_percent = get_cpu_usage();
        
        // Memory usage
        info.memory_usage_mb = get_memory_usage();
        info.memory_total_mb = get_total_memory();
        
        // GPU information
        get_gpu_info(info.gpu_memory_usage_mb, info.gpu_memory_total_mb, info.gpu_utilization_percent);
        
        return info;
    }

private:
    static auto get_cpu_usage() -> double {
        static ULARGE_INTEGER last_cpu, last_sys_cpu, last_usr_cpu;
        static int num_processors = 0;
        static HANDLE self = GetCurrentProcess();
        
        if (num_processors == 0) {
            SYSTEM_INFO sysinfo;
            GetSystemInfo(&sysinfo);
            num_processors = sysinfo.dwNumberOfProcessors;
            
            FILETIME ftime, fsys, fusr;
            GetSystemTimeAsFileTime(&ftime);
            memcpy(&last_cpu, &ftime, sizeof(FILETIME));
            
            GetProcessTimes(self, &ftime, &ftime, &fsys, &fusr);
            memcpy(&last_sys_cpu, &fsys, sizeof(FILETIME));
            memcpy(&last_usr_cpu, &fusr, sizeof(FILETIME));
        }
        
        FILETIME ftime, fsys, fusr;
        ULARGE_INTEGER now, sys, usr;
        
        GetSystemTimeAsFileTime(&ftime);
        memcpy(&now, &ftime, sizeof(FILETIME));
        
        GetProcessTimes(self, &ftime, &ftime, &fsys, &fusr);
        memcpy(&sys, &fsys, sizeof(FILETIME));
        memcpy(&usr, &fusr, sizeof(FILETIME));
        
        double percent = (sys.QuadPart - last_sys_cpu.QuadPart) + (usr.QuadPart - last_usr_cpu.QuadPart);
        percent /= (now.QuadPart - last_cpu.QuadPart);
        percent /= num_processors;
        
        last_cpu = now;
        last_usr_cpu = usr;
        last_sys_cpu = sys;
        
        return percent * 100;
    }

    static auto get_memory_usage() -> size_t {
        PROCESS_MEMORY_COUNTERS pmc;
        if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
            return pmc.WorkingSetSize / (1024 * 1024); // Convert to MB
        }
        return 0;
    }

    static auto get_total_memory() -> size_t {
        MEMORYSTATUSEX memInfo;
        memInfo.dwLength = sizeof(MEMORYSTATUSEX);
        if (GlobalMemoryStatusEx(&memInfo)) {
            return memInfo.ullTotalPhys / (1024 * 1024); // Convert to MB
        }
        return 0;
    }

    static auto get_gpu_info(size_t& used_mb, size_t& total_mb, double& utilization) -> void {
        used_mb = 0;
        total_mb = 0;
        utilization = 0.0;
        
        try {
            ComPtr<IDXGIFactory2> factory;
            HRESULT hr = CreateDXGIFactory2(0, __uuidof(IDXGIFactory2), &factory);
            if (FAILED(hr)) return;
            
            ComPtr<IDXGIAdapter1> adapter;
            for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter)); ++i) {
                DXGI_ADAPTER_DESC1 desc;
                hr = adapter->GetDesc1(&desc);
                if (FAILED(hr)) continue;
                
                // Check if it's a display adapter (not software)
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                
                total_mb = desc.DedicatedVideoMemory / (1024 * 1024);
                
                // Try to get GPU memory usage (simplified)
                ComPtr<ID3D11Device> device;
                ComPtr<ID3D11DeviceContext> context;
                D3D_FEATURE_LEVEL featureLevel;
                
                hr = D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0,
                                     nullptr, 0, D3D11_SDK_VERSION, &device, &featureLevel, &context);
                
                if (SUCCEEDED(hr)) {
                    // Get memory usage (approximation)
                    ComPtr<IDXGIAdapter3> adapter3;
                    if (SUCCEEDED(adapter.As(&adapter3))) {
                        DXGI_QUERY_VIDEO_MEMORY_INFO videoMemoryInfo;
                        hr = adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &videoMemoryInfo);
                        if (SUCCEEDED(hr)) {
                            used_mb = videoMemoryInfo.CurrentUsage / (1024 * 1024);
                        }
                    }
                }
                
                break; // Use first available GPU
            }
        } catch (...) {
            // GPU info not available
        }
    }
};

// High-resolution timer for performance measurement
class HighResolutionTimer {
public:
    HighResolutionTimer() {
        reset();
    }

    auto reset() -> void {
        start_time_ = std::chrono::high_resolution_clock::now();
    }

    auto elapsed_seconds() const -> double {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_);
        return duration.count() * 1e-9;
    }

    auto elapsed_milliseconds() const -> double {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
        return duration.count() * 1e-3;
    }

    auto elapsed_microseconds() const -> double {
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
        return duration.count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

// Windows thread affinity manager for CPU optimization
class ThreadAffinityManager {
public:
    static auto set_thread_affinity(std::thread& thread, uint32_t core_id) -> bool {
        if (core_id >= std::thread::hardware_concurrency()) {
            return false;
        }

        HANDLE handle = static_cast<HANDLE>(thread.native_handle());
        DWORD_PTR affinityMask = 1ULL << core_id;
        
        return SetThreadAffinityMask(handle, affinityMask) != 0;
    }

    static auto set_current_thread_affinity(uint32_t core_id) -> bool {
        if (core_id >= std::thread::hardware_concurrency()) {
            return false;
        }

        DWORD_PTR affinityMask = 1ULL << core_id;
        return SetThreadAffinityMask(GetCurrentThread(), affinityMask) != 0;
    }

    static auto set_thread_priority(std::thread& thread, int priority) -> bool {
        HANDLE handle = static_cast<HANDLE>(thread.native_handle());
        return SetThreadPriority(handle, priority) != 0;
    }

    static auto set_current_thread_priority(int priority) -> bool {
        return SetThreadPriority(GetCurrentThread(), priority) != 0;
    }
};

// Windows event logger for debugging and monitoring
class WindowsEventLogger {
public:
    enum class LogLevel {
        INFO,
        WARNING,
        ERROR,
        DEBUG
    };

    static auto log(LogLevel level, const std::string& message) -> void {
        const char* level_str = get_level_string(level);
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        
        std::string timestamp = std::format("[{:%Y-%m-%d %H:%M:%S}]", std::chrono::system_clock::from_time_t(time_t));
        std::string log_message = std::format("{} [{}] {}", timestamp, level_str, message);
        
        // Output to console
        std::cout << log_message << std::endl;
        
        // Output to Windows debug console if available
        OutputDebugStringA((log_message + "\n").c_str());
        
        // Could also write to Windows Event Log here if needed
    }

private:
    static auto get_level_string(LogLevel level) -> const char* {
        switch (level) {
            case LogLevel::INFO: return "INFO";
            case LogLevel::WARNING: return "WARN";
            case LogLevel::ERROR: return "ERROR";
            case LogLevel::DEBUG: return "DEBUG";
            default: return "UNKNOWN";
        }
    }
};

// Windows registry helper for configuration
class WindowsRegistryHelper {
public:
    static auto read_string(const std::string& key, const std::string& value_name, 
                           const std::string& default_value = "") -> std::string {
        HKEY hKey;
        DWORD type, size;
        std::string result = default_value;
        
        if (RegOpenKeyExA(HKEY_CURRENT_USER, key.c_str(), 0, KEY_READ, &hKey) == ERROR_SUCCESS) {
            if (RegQueryValueExA(hKey, value_name.c_str(), nullptr, &type, nullptr, &size) == ERROR_SUCCESS) {
                if (type == REG_SZ) {
                    std::vector<char> buffer(size);
                    if (RegQueryValueExA(hKey, value_name.c_str(), nullptr, &type, 
                                        reinterpret_cast<LPBYTE>(buffer.data()), &size) == ERROR_SUCCESS) {
                        result = std::string(buffer.data());
                    }
                }
            }
            RegCloseKey(hKey);
        }
        
        return result;
    }

    static auto write_string(const std::string& key, const std::string& value_name, 
                            const std::string& value) -> bool {
        HKEY hKey;
        
        if (RegCreateKeyExA(HKEY_CURRENT_USER, key.c_str(), 0, nullptr, 
                           REG_OPTION_NON_VOLATILE, KEY_WRITE, nullptr, &hKey, nullptr) == ERROR_SUCCESS) {
            LONG result = RegSetValueExA(hKey, value_name.c_str(), 0, REG_SZ,
                                        reinterpret_cast<const BYTE*>(value.c_str()),
                                        static_cast<DWORD>(value.size() + 1));
            RegCloseKey(hKey);
            return result == ERROR_SUCCESS;
        }
        
        return false;
    }
};

// Utility function implementations
namespace utils {

template<Numeric T>
auto letterbox_resize(const cv::Mat& image, const cv::Size& target_size, T scale_factor) -> cv::Mat {
    float scale = std::min(static_cast<float>(target_size.width) / image.cols,
                          static_cast<float>(target_size.height) / image.rows);
    scale *= static_cast<float>(scale_factor);
    
    cv::Size new_size(static_cast<int>(image.cols * scale), static_cast<int>(image.rows * scale));
    cv::Mat resized;
    cv::resize(image, resized, new_size, 0, 0, cv::INTER_LINEAR);
    
    cv::Mat letterboxed(target_size, image.type(), cv::Scalar(114, 114, 114));
    
    int top = (target_size.height - new_size.height) / 2;
    int left = (target_size.width - new_size.width) / 2;
    
    resized.copyTo(letterboxed(cv::Rect(left, top, new_size.width, new_size.height)));
    
    return letterboxed;
}

auto draw_detections(cv::Mat& image, std::span<const DetectionResult> detections) -> void {
    for (const auto& detection : detections) {
        // Draw bounding box
        cv::rectangle(image, detection.bbox, cv::Scalar(0, 255, 0), 2);
        
        // Draw label
        std::string label = std::format("{} {:.1f}%", detection.class_name, detection.confidence * 100.0f);
        
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

template<FloatingPoint T>
auto calculate_iou(const cv::Rect& box1, const cv::Rect& box2) noexcept -> T {
    cv::Rect intersection = box1 & box2;
    T intersection_area = static_cast<T>(intersection.area());
    
    T union_area = static_cast<T>(box1.area()) + static_cast<T>(box2.area()) - intersection_area;
    
    return union_area > 0 ? intersection_area / union_area : static_cast<T>(0.0);
}

auto get_gpu_memory_info() -> std::pair<size_t, size_t> {
    size_t used = 0, total = 0;
    
    try {
        ComPtr<IDXGIFactory2> factory;
        HRESULT hr = CreateDXGIFactory2(0, __uuidof(IDXGIFactory2), &factory);
        if (FAILED(hr)) return {0, 0};
        
        ComPtr<IDXGIAdapter1> adapter;
        for (UINT i = 0; SUCCEEDED(factory->EnumAdapters1(i, &adapter)); ++i) {
            DXGI_ADAPTER_DESC1 desc;
            hr = adapter->GetDesc1(&desc);
            if (FAILED(hr)) continue;
            
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
            
            total = desc.DedicatedVideoMemory / (1024 * 1024);
            
            // Try to get current usage
            ComPtr<IDXGIAdapter3> adapter3;
            if (SUCCEEDED(adapter.As(&adapter3))) {
                DXGI_QUERY_VIDEO_MEMORY_INFO videoMemoryInfo;
                hr = adapter3->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &videoMemoryInfo);
                if (SUCCEEDED(hr)) {
                    used = videoMemoryInfo.CurrentUsage / (1024 * 1024);
                }
            }
            
            break;
        }
    } catch (...) {
        // GPU info not available
    }
    
    return {used, total};
}

} // namespace utils

// RAII helper for Windows handles
template<typename T>
class WindowsHandleGuard {
public:
    constexpr WindowsHandleGuard() noexcept = default;
    explicit WindowsHandleGuard(T handle) noexcept : handle_(handle) {}
    
    ~WindowsHandleGuard() {
        if (handle_ != nullptr) {
            CloseHandle(handle_);
        }
    }
    
    auto get() const noexcept -> T { return handle_; }
    auto release() noexcept -> T { 
        T temp = handle_; 
        handle_ = nullptr; 
        return temp; 
    }
    
    // Move-only type
    WindowsHandleGuard(const WindowsHandleGuard&) = delete;
    auto operator=(const WindowsHandleGuard&) -> WindowsHandleGuard& = delete;
    WindowsHandleGuard(WindowsHandleGuard&& other) noexcept : handle_(other.handle_) { other.handle_ = nullptr; }
    auto operator=(WindowsHandleGuard&& other) noexcept -> WindowsHandleGuard& {
        if (this != &other) {
            if (handle_) CloseHandle(handle_);
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

private:
    T handle_ = nullptr;
};

// Type aliases for common Windows handle guards
using FileHandleGuard = WindowsHandleGuard<HANDLE>;
using RegistryHandleGuard = WindowsHandleGuard<HKEY>;

} // namespace BikeGuard
