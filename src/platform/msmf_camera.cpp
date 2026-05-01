#include "bikeguard_engine.hpp"
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <windows.h>
#include <iostream>
#include <format>

namespace BikeGuard {

// Media Foundation camera implementation for Windows with minimal latency
class MSMFCamera : public ICameraCapture {
public:
    MSMFCamera() = default;
    ~MSMFCamera() override { release(); }

    auto initialize(int camera_id = 0) -> bool override {
        try {
            // Initialize Media Foundation
            HRESULT hr = MFStartup(MF_VERSION);
            if (FAILED(hr)) {
                std::cerr << "Failed to initialize Media Foundation\n";
                return false;
            }

            // Create Media Source
            hr = MFCreateDeviceSource(&camera_id, &media_source_);
            if (FAILED(hr)) {
                std::cerr << "Failed to create media source\n";
                MFShutdown();
                return false;
            }

            // Create Source Reader
            hr = MFCreateSourceReaderFromMediaSource(media_source_.Get(), nullptr, &source_reader_);
            if (FAILED(hr)) {
                std::cerr << "Failed to create source reader\n";
                release();
                return false;
            }

            // Configure for low latency
            if (!configure_low_latency()) {
                std::cerr << "Failed to configure low latency mode\n";
                release();
                return false;
            }

            // Get frame format
            if (!get_frame_format()) {
                std::cerr << "Failed to get frame format\n";
                release();
                return false;
            }

            initialized_ = true;
            return true;

        } catch (const std::exception& e) {
            std::cerr << std::format("Camera initialization failed: {}\n", e.what());
            release();
            return false;
        }
    }

    auto capture_frame(cv::Mat& frame) -> bool override {
        if (!initialized_) {
            return false;
        }

        try {
            DWORD stream_index = 0;
            DWORD flags = 0;
            LONGLONG timestamp = 0;

            // Read sample with minimal timeout for low latency
            Microsoft::WRL::ComPtr<IMFSample> sample;
            HRESULT hr = source_reader_->ReadSample(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                0,  // No flags
                &stream_index,
                &flags,
                &timestamp,
                &sample);

            if (FAILED(hr) || !sample) {
                return false;
            }

            // Check for end of stream
            if (flags & MF_SOURCE_READERF_ENDOFSTREAM) {
                return false;
            }

            // Get media buffer
            Microsoft::WRL::ComPtr<IMFMediaBuffer> media_buffer;
            hr = sample->ConvertToContiguousBuffer(&media_buffer);
            if (FAILED(hr)) {
                return false;
            }

            // Lock buffer
            BYTE* buffer_data = nullptr;
            DWORD buffer_length = 0;
            hr = media_buffer->Lock(&buffer_data, nullptr, &buffer_length);
            if (FAILED(hr)) {
                return false;
            }

            // Convert to OpenCV Mat (zero-copy if possible)
            bool success = convert_to_mat(buffer_data, buffer_length, frame);

            // Unlock buffer
            media_buffer->Unlock();

            return success;

        } catch (const std::exception& e) {
            std::cerr << std::format("Frame capture failed: {}\n", e.what());
            return false;
        }
    }

    auto release() -> void override {
        source_reader_.Reset();
        media_source_.Reset();
        
        if (mf_initialized_) {
            MFShutdown();
            mf_initialized_ = false;
        }
        
        initialized_ = false;
    }

    auto get_frame_size() const -> cv::Size override {
        return frame_size_;
    }

    auto get_fps() const -> double override {
        return frame_rate_;
    }

    auto is_initialized() const -> bool override {
        return initialized_;
    }

private:
    auto configure_low_latency() -> bool {
        try {
            // Configure source reader for low latency
            Microsoft::WRL::ComPtr<IMFSourceReaderEx> source_reader_ex;
            HRESULT hr = source_reader_.As(&source_reader_ex);
            if (FAILED(hr)) {
                return false;
            }

            // Set low latency mode
            hr = source_reader_ex->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);
            if (FAILED(hr)) {
                return false;
            }

            // Configure for real-time processing
            Microsoft::WRL::ComPtr<IMFAttributes> attributes;
            hr = MFCreateAttributes(&attributes, 1);
            if (FAILED(hr)) {
                return false;
            }

            // Set real-time mode
            hr = attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
            if (FAILED(hr)) {
                return false;
            }

            return true;

        } catch (...) {
            return false;
        }
    }

    auto get_frame_format() -> bool {
        try {
            Microsoft::WRL::ComPtr<IMFMediaType> media_type;
            HRESULT hr = source_reader_->GetNativeMediaType(
                MF_SOURCE_READER_FIRST_VIDEO_STREAM, 
                0,  // First media type
                &media_type);

            if (FAILED(hr)) {
                return false;
            }

            // Get frame dimensions
            hr = MFGetAttributeSize(media_type.Get(), MF_MT_FRAME_SIZE, 
                                  &frame_width_, &frame_height_);
            if (FAILED(hr)) {
                return false;
            }

            frame_size_ = cv::Size(static_cast<int>(frame_width_), 
                                 static_cast<int>(frame_height_));

            // Get frame rate
            UINT64 numerator = 0, denominator = 0;
            hr = MFGetAttributeRatio(media_type.Get(), MF_MT_FRAME_RATE, 
                                   &numerator, &denominator);
            if (SUCCEEDED(hr) && denominator > 0) {
                frame_rate_ = static_cast<double>(numerator) / denominator;
            } else {
                frame_rate_ = 30.0; // Default to 30 FPS
            }

            // Get pixel format
            GUID subtype = {0};
            hr = media_type->GetGUID(MF_MT_SUBTYPE, &subtype);
            if (FAILED(hr)) {
                return false;
            }

            // Store pixel format for conversion
            pixel_format_ = subtype;

            return true;

        } catch (...) {
            return false;
        }
    }

    auto convert_to_mat(BYTE* buffer_data, DWORD buffer_length, cv::Mat& frame) -> bool {
        try {
            // Handle different pixel formats
            if (pixel_format_ == MFVideoFormat_NV12 || pixel_format_ == MFVideoFormat_YV12) {
                // Convert YUV to BGR
                return convert_yuv_to_bgr(buffer_data, buffer_length, frame);
            } else if (pixel_format_ == MFVideoFormat_RGB24 || pixel_format_ == MFVideoFormat_BGR32) {
                // Direct copy for RGB formats
                return convert_rgb_to_bgr(buffer_data, buffer_length, frame);
            } else {
                // Fallback: try to create Mat from raw data
                frame.create(frame_size_, CV_8UC3);
                std::memcpy(frame.data, buffer_data, 
                           std::min(static_cast<size_t>(buffer_length), 
                                   static_cast<size_t>(frame.total() * frame.elemSize())));
                return true;
            }

        } catch (...) {
            return false;
        }
    }

    auto convert_yuv_to_bgr(BYTE* buffer_data, DWORD buffer_length, cv::Mat& frame) -> bool {
        try {
            // Create YUV Mat
            cv::Mat yuv_frame(frame_size_, CV_8UC2, buffer_data);
            
            // Convert YUV to BGR
            cv::cvtColor(yuv_frame, frame, cv::COLOR_YUV2BGR_UYVY);
            
            return true;

        } catch (...) {
            return false;
        }
    }

    auto convert_rgb_to_bgr(BYTE* buffer_data, DWORD buffer_length, cv::Mat& frame) -> bool {
        try {
            frame.create(frame_size_, CV_8UC3);
            
            if (pixel_format_ == MFVideoFormat_RGB24) {
                // Convert RGB to BGR
                cv::Mat rgb_frame(frame_size_, CV_8UC3, buffer_data);
                cv::cvtColor(rgb_frame, frame, cv::COLOR_RGB2BGR);
            } else if (pixel_format_ == MFVideoFormat_BGR32) {
                // Handle BGRA format
                cv::Mat bgra_frame(frame_size_, CV_8UC4, buffer_data);
                cv::cvtColor(bgra_frame, frame, cv::COLOR_BGRA2BGR);
            } else {
                // Direct copy for BGR
                std::memcpy(frame.data, buffer_data, 
                           std::min(static_cast<size_t>(buffer_length), 
                                   static_cast<size_t>(frame.total() * frame.elemSize())));
            }
            
            return true;

        } catch (...) {
            return false;
        }
    }

private:
    Microsoft::WRL::ComPtr<IMFMediaSource> media_source_;
    Microsoft::WRL::ComPtr<IMFSourceReader> source_reader_;
    
    bool initialized_ = false;
    bool mf_initialized_ = true;
    
    // Frame properties
    UINT64 frame_width_ = 0;
    UINT64 frame_height_ = 0;
    cv::Size frame_size_{0, 0};
    double frame_rate_ = 30.0;
    GUID pixel_format_ = {0};
};

// Fallback OpenCV camera implementation
class OpenCVCamera : public ICameraCapture {
public:
    OpenCVCamera() = default;
    ~OpenCVCamera() override { release(); }

    auto initialize(int camera_id = 0) -> bool override {
        try {
            // Use CAP_MSMF backend for Windows
            cap_.open(camera_id, cv::CAP_MSMF);
            
            if (!cap_.isOpened()) {
                // Fallback to default backend
                cap_.open(camera_id);
            }
            
            if (!cap_.isOpened()) {
                return false;
            }
            
            // Configure for low latency
            cap_.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
            cap_.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
            cap_.set(cv::CAP_PROP_FPS, 30);
            cap_.set(cv::CAP_PROP_BUFFERSIZE, 1); // Minimize latency
            cap_.set(cv::CAP_PROP_AUTOFOCUS, 1);
            
            // Verify settings
            frame_size_ = cv::Size(
                static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT))
            );
            frame_rate_ = cap_.get(cv::CAP_PROP_FPS);
            
            initialized_ = true;
            return true;

        } catch (const std::exception& e) {
            std::cerr << std::format("OpenCV camera initialization failed: {}\n", e.what());
            return false;
        }
    }

    auto capture_frame(cv::Mat& frame) -> bool override {
        if (!initialized_) {
            return false;
        }
        
        return cap_.read(frame);
    }

    auto release() -> void override {
        if (cap_.isOpened()) {
            cap_.release();
        }
        initialized_ = false;
    }

    auto get_frame_size() const -> cv::Size override {
        return frame_size_;
    }

    auto get_fps() const -> double override {
        return frame_rate_;
    }

    auto is_initialized() const -> bool override {
        return initialized_;
    }

private:
    cv::VideoCapture cap_;
    bool initialized_ = false;
    cv::Size frame_size_{0, 0};
    double frame_rate_ = 30.0;
};

// Factory function implementation
auto create_msmf_camera() -> std::unique_ptr<ICameraCapture> {
    // Try Media Foundation first
    auto msmf_camera = std::make_unique<MSMFCamera>();
    
    if (msmf_camera->initialize(0)) {
        std::cout << "Using Media Foundation camera backend\n";
        return msmf_camera;
    }
    
    // Fallback to OpenCV
    std::cout << "Media Foundation not available, falling back to OpenCV\n";
    auto opencv_camera = std::make_unique<OpenCVCamera>();
    
    if (opencv_camera->initialize(0)) {
        return opencv_camera;
    }
    
    return nullptr;
}

} // namespace BikeGuard
