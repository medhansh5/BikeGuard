#include "bikeguard_core.hpp"
#include <iostream>

#ifdef _WIN32
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

namespace BikeGuard {

// OpenCV Camera Implementation (Cross-platform)
class OpenCVCamera : public CameraCapture {
public:
    OpenCVCamera() = default;
    ~OpenCVCamera() override { release(); }
    
    bool initialize(int camera_id = 0) override {
        cap_.open(camera_id);
        if (!cap_.isOpened()) {
            return false;
        }
        
        // Set optimal properties for performance
        cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
        cap_.set(cv::CAP_PROP_FPS, 30);
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        
        return true;
    }
    
    bool capture_frame(cv::Mat& frame) override {
        if (!cap_.isOpened()) return false;
        return cap_.read(frame);
    }
    
    void release() override {
        if (cap_.isOpened()) {
            cap_.release();
        }
    }
    
    cv::Size get_frame_size() const override {
        if (!cap_.isOpened()) return cv::Size(0, 0);
        return cv::Size(static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                       static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
    }
    
    double get_fps() const override {
        if (!cap_.isOpened()) return 0.0;
        return cap_.get(cv::CAP_PROP_FPS);
    }
    
#ifdef _WIN32
    bool initialize_directx() override {
        // DirectX initialization not applicable for OpenCV camera
        return false;
    }
    
    bool capture_frame_dx(cv::Mat& frame) override {
        // Fallback to regular capture
        return capture_frame(frame);
    }
#endif

private:
    cv::VideoCapture cap_;
};

#ifdef _WIN32
// DirectX Camera Implementation (Windows-specific)
class DirectXCamera : public CameraCapture {
public:
    DirectXCamera() = default;
    ~DirectXCamera() override { release(); }
    
    bool initialize(int camera_id = 0) override {
        // For now, fallback to OpenCV implementation
        // In a full implementation, this would use MediaFoundation or DirectShow
        return opencv_camera_.initialize(camera_id);
    }
    
    bool capture_frame(cv::Mat& frame) override {
        return opencv_camera_.capture_frame(frame);
    }
    
    void release() override {
        opencv_camera_.release();
    }
    
    cv::Size get_frame_size() const override {
        return opencv_camera_.get_frame_size();
    }
    
    double get_fps() const override {
        return opencv_camera_.get_fps();
    }
    
    bool initialize_directx() override {
        // Initialize DirectX capture pipeline
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            nullptr, 0, D3D11_SDK_VERSION,
            &d3d_device_, nullptr, &d3d_context_);
        
        if (FAILED(hr)) {
            std::cerr << "Failed to create D3D11 device" << std::endl;
            return false;
        }
        
        // Initialize DXGI for camera access
        hr = CreateDXGIFactory2(0, __uuidof(IDXGIFactory2), (void**)&dxgi_factory_);
        if (FAILED(hr)) {
            std::cerr << "Failed to create DXGI factory" << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool capture_frame_dx(cv::Mat& frame) override {
        // DirectX-specific capture implementation
        // This would involve texture sharing and direct GPU memory access
        // For now, fallback to OpenCV
        return opencv_camera_.capture_frame(frame);
    }

private:
    OpenCVCamera opencv_camera_;
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
    ComPtr<IDXGIFactory2> dxgi_factory_;
};
#endif

// CPU Hardware Accelerator (Cross-platform)
class CPUAccelerator : public HardwareAccelerator {
public:
    CPUAccelerator() = default;
    ~CPUAccelerator() override = default;
    
    bool initialize(const InferenceConfig& config) override {
        // CPU doesn't need special initialization
        available_ = true;
        return true;
    }
    
    void* get_execution_provider() override {
        // CPU is the default, no special provider needed
        return nullptr;
    }
    
    bool is_available() const override {
        return available_;
    }
    
    std::string get_provider_name() const override {
        return "CPU";
    }
    
#ifdef _WIN32
    bool initialize_d3d(int device_id) override {
        // CPU doesn't use D3D
        return false;
    }
    
    ID3D11Device* get_d3d_device() override {
        return nullptr;
    }
#endif

private:
    bool available_ = false;
};

#ifdef _WIN32
// D3D Hardware Accelerator (Windows-specific)
class D3DAccelerator : public HardwareAccelerator {
public:
    D3DAccelerator() = default;
    ~D3DAccelerator() override { cleanup(); }
    
    bool initialize(const InferenceConfig& config) override {
        return initialize_d3d(config.gpu_device_id);
    }
    
    void* get_execution_provider() override {
        if (!d3d_provider_) {
            // Create DML execution provider
            OrtDmlApi* dml_api = nullptr;
            Ort::GetApi().GetExecutionProviderApi("DML", ORT_API_VERSION, 
                reinterpret_cast<const void**>(&dml_api));
            
            if (dml_api) {
                dml_provider_ = dml_api;
                return dml_api;
            }
        }
        return dml_provider_;
    }
    
    bool is_available() const override {
        return d3d_device_ && available_;
    }
    
    std::string get_provider_name() const override {
        return "D3D_DML";
    }
    
    bool initialize_d3d(int device_id) override {
        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            nullptr, 0, D3D11_SDK_VERSION,
            &d3d_device_, nullptr, &d3d_context_);
        
        if (FAILED(hr)) {
            std::cerr << "Failed to create D3D11 device for inference" << std::endl;
            available_ = false;
            return false;
        }
        
        // Check for DirectML support
        ComPtr<IDXGIAdapter> adapter;
        hr = d3d_device_->GetAdapter(&adapter);
        if (FAILED(hr)) {
            std::cerr << "Failed to get DXGI adapter" << std::endl;
            available_ = false;
            return false;
        }
        
        DXGI_ADAPTER_DESC desc;
        hr = adapter->GetDesc(&desc);
        if (FAILED(hr)) {
            std::cerr << "Failed to get adapter description" << std::endl;
            available_ = false;
            return false;
        }
        
        std::wcout << L"Using GPU: " << desc.Description << std::endl;
        available_ = true;
        return true;
    }
    
    ID3D11Device* get_d3d_device() override {
        return d3d_device_.Get();
    }

private:
    void cleanup() {
        d3d_context_.Reset();
        d3d_device_.Reset();
        dml_provider_ = nullptr;
        available_ = false;
    }
    
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
    void* dml_provider_ = nullptr;
    bool available_ = false;
};
#endif

// Factory function implementations
std::unique_ptr<CameraCapture> create_opencv_camera() {
    return std::make_unique<OpenCVCamera>();
}

std::unique_ptr<HardwareAccelerator> create_cpu_accelerator() {
    return std::make_unique<CPUAccelerator>();
}

#ifdef _WIN32
std::unique_ptr<CameraCapture> create_directx_camera() {
    return std::make_unique<DirectXCamera>();
}

std::unique_ptr<HardwareAccelerator> create_d3d_accelerator() {
    return std::make_unique<D3DAccelerator>();
}
#endif

} // namespace BikeGuard
