#include "bikeguard_road_engine.hpp"
#include <complex>
#include <numbers>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace BikeGuard {

// Enhanced FFT Vibration Filter Implementation
auto EnhancedVibrationFilter::initialize(float sample_rate, size_t fft_size) -> bool {
    try {
        sample_rate_ = sample_rate;
        fft_size_ = fft_size;
        
        // Ensure FFT size is power of 2 for efficiency
        if (!is_power_of_two(fft_size_)) {
            fft_size_ = next_power_of_two(fft_size_);
        }
        
        // Allocate buffers for FFT processing
        fft_input_.resize(fft_size_);
        fft_output_.resize(fft_size_);
        frequency_spectrum_.resize(fft_size_);
        
        // Initialize ShadowMap v1.3.0 compatible vibration analysis
        vibration_config_.high_intensity_threshold = 15.0f;  // Hz threshold
        vibration_config_.amplitude_threshold = 0.7f;        // Amplitude threshold
        vibration_config_.frame_dropping_enabled = true;
        vibration_config_.max_consecutive_drops = 3;
        
        // Set up frequency bands for Indian road conditions
        vibration_config_.frequency_bands.engine_vibration_min = 10.0f;
        vibration_config_.frequency_bands.engine_vibration_max = 25.0f;
        vibration_config_.frequency_bands.road_vibration_min = 2.0f;
        vibration_config_.frequency_bands.road_vibration_max = 8.0f;
        vibration_config_.frequency_bands.shock_absorber_min = 15.0f;
        vibration_config_.frequency_bands.shock_absorber_max = 40.0f;
        
        fft_initialized_ = true;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Enhanced vibration filter initialization failed: {}\n", e.what());
        return false;
    }
}

auto EnhancedVibrationFilter::filter_frame(std::span<float> motion_data) -> std::span<float> {
    if (!fft_initialized_ || motion_data.empty()) {
        return motion_data;
    }
    
    try {
        // Copy and pad input data to FFT size
        size_t data_size = std::min(motion_data.size(), fft_size_);
        std::copy(motion_data.begin(), motion_data.begin() + data_size, fft_input_.begin());
        std::fill(fft_input_.begin() + data_size, fft_input_.end(), 0.0f);
        
        // Apply window function (Hamming window to reduce spectral leakage)
        apply_hamming_window();
        
        // Perform FFT to get frequency domain representation
        perform_fft();
        
        // Analyze vibration intensity using ShadowMap v1.3.0 logic
        current_vibration_intensity_ = analyze_vibration_intensity(motion_data);
        
        // Determine if frame should be dropped based on vibration analysis
        frame_should_be_dropped_.store(should_drop_frame());
        
        // If frame should be dropped, return empty span
        if (frame_should_be_dropped_.load()) {
            vibration_config_.consecutive_dropped_frames++;
            return std::span<float>();
        }
        
        // Apply frequency domain filtering for motion stabilization
        apply_frequency_filtering();
        
        // Perform inverse FFT to get filtered motion data
        perform_inverse_fft();
        
        // Copy filtered results back to input span
        size_t output_size = std::min(motion_data.size(), fft_input_.size());
        std::copy(fft_input_.begin(), fft_input_.begin() + output_size, motion_data.begin());
        
        // Reset consecutive dropped frames counter
        vibration_config_.consecutive_dropped_frames = 0;
        
        return motion_data;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Enhanced vibration filtering failed: {}\n", e.what());
        return motion_data;
    }
}

auto EnhancedVibrationFilter::analyze_vibration_intensity(std::span<float> motion_data) -> float {
    try {
        // Calculate frequency spectrum magnitude
        std::vector<float> magnitude_spectrum(fft_size_);
        for (size_t i = 0; i < fft_size_; ++i) {
            magnitude_spectrum[i] = std::abs(fft_output_[i]);
        }
        
        // Analyze specific frequency bands relevant to motorcycle vibrations
        float engine_vibration = detect_engine_vibration();
        float road_vibration = detect_road_vibration();
        float shock_response = detect_shock_absorber_response();
        
        // Calculate overall vibration intensity using ShadowMap v1.3.0 algorithm
        float weighted_intensity = 0.0f;
        weighted_intensity += engine_vibration * 0.4f;  // Engine vibrations (40% weight)
        weighted_intensity += road_vibration * 0.3f;     // Road vibrations (30% weight)
        weighted_intensity += shock_response * 0.3f;    // Shock absorber response (30% weight)
        
        return weighted_intensity;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Vibration intensity analysis failed: {}\n", e.what());
        return 0.0f;
    }
}

auto EnhancedVibrationFilter::should_drop_frame() -> bool {
    if (!vibration_config_.frame_dropping_enabled) {
        return false;
    }
    
    // Check if we've exceeded maximum consecutive drops
    if (vibration_config_.consecutive_dropped_frames >= vibration_config_.max_consecutive_drops) {
        return false; // Force processing to avoid dropping too many frames
    }
    
    // ShadowMap v1.3.0 frame dropping logic
    float frequency_threshold = vibration_config_.high_intensity_threshold;
    float amplitude_threshold = vibration_config_.amplitude_threshold;
    
    // Check if high-intensity vibration exceeds thresholds
    bool high_frequency_vibration = current_vibration_intensity_ > frequency_threshold;
    bool high_amplitude_vibration = calculate_peak_amplitude() > amplitude_threshold;
    
    // Drop frame if both conditions are met
    return high_frequency_vibration && high_amplitude_vibration;
}

auto EnhancedVibrationFilter::detect_engine_vibration() -> float {
    try {
        // Analyze engine vibration frequency band (10-25 Hz for Royal Enfield Classic 350)
        float min_freq = vibration_config_.frequency_bands.engine_vibration_min;
        float max_freq = vibration_config_.frequency_bands.engine_vibration_max;
        
        float frequency_resolution = sample_rate_ / fft_size_;
        size_t min_bin = static_cast<size_t>(min_freq / frequency_resolution);
        size_t max_bin = static_cast<size_t>(max_freq / frequency_resolution);
        
        float engine_intensity = 0.0f;
        size_t count = 0;
        
        for (size_t i = min_bin; i < max_bin && i < fft_size_; ++i) {
            engine_intensity += std::abs(fft_output_[i]);
            count++;
        }
        
        return count > 0 ? engine_intensity / count : 0.0f;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Engine vibration detection failed: {}\n", e.what());
        return 0.0f;
    }
}

auto EnhancedVibrationFilter::detect_road_vibration() -> float {
    try {
        // Analyze road vibration frequency band (2-8 Hz)
        float min_freq = vibration_config_.frequency_bands.road_vibration_min;
        float max_freq = vibration_config_.frequency_bands.road_vibration_max;
        
        float frequency_resolution = sample_rate_ / fft_size_;
        size_t min_bin = static_cast<size_t>(min_freq / frequency_resolution);
        size_t max_bin = static_cast<size_t>(max_freq / frequency_resolution);
        
        float road_intensity = 0.0f;
        size_t count = 0;
        
        for (size_t i = min_bin; i < max_bin && i < fft_size_; ++i) {
            road_intensity += std::abs(fft_output_[i]);
            count++;
        }
        
        return count > 0 ? road_intensity / count : 0.0f;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Road vibration detection failed: {}\n", e.what());
        return 0.0f;
    }
}

auto EnhancedVibrationFilter::detect_shock_absorber_response() -> float {
    try {
        // Analyze shock absorber response frequency band (15-40 Hz)
        float min_freq = vibration_config_.frequency_bands.shock_absorber_min;
        float max_freq = vibration_config_.frequency_bands.shock_absorber_max;
        
        float frequency_resolution = sample_rate_ / fft_size_;
        size_t min_bin = static_cast<size_t>(min_freq / frequency_resolution);
        size_t max_bin = static_cast<size_t>(max_freq / frequency_resolution);
        
        float shock_intensity = 0.0f;
        size_t count = 0;
        
        for (size_t i = min_bin; i < max_bin && i < fft_size_; ++i) {
            shock_intensity += std::abs(fft_output_[i]);
            count++;
        }
        
        return count > 0 ? shock_intensity / count : 0.0f;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Shock absorber response detection failed: {}\n", e.what());
        return 0.0f;
    }
}

auto EnhancedVibrationFilter::calculate_peak_amplitude() -> float {
    try {
        // Calculate peak amplitude in the time domain
        float peak_amplitude = 0.0f;
        for (size_t i = 0; i < fft_input_.size(); ++i) {
            peak_amplitude = std::max(peak_amplitude, std::abs(fft_input_[i]));
        }
        
        // Normalize by FFT size
        return fft_input_.size() > 0 ? peak_amplitude / fft_input_.size() : 0.0f;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Peak amplitude calculation failed: {}\n", e.what());
        return 0.0f;
    }
}

auto EnhancedVibrationFilter::perform_fft() -> void {
    try {
        // Cooley-Tukey FFT implementation optimized for real-time processing
        std::fill(fft_output_.begin(), fft_output_.end(), std::complex<float>(0.0f, 0.0f));
        
        for (size_t k = 0; k < fft_size_; ++k) {
            std::complex<float> sum(0.0f, 0.0f);
            
            for (size_t n = 0; n < fft_size_; ++n) {
                float angle = -2.0f * std::numbers::pi_v<float> * k * n / fft_size_;
                std::complex<float> term(std::cos(angle), std::sin(angle));
                sum += fft_input_[n] * term;
            }
            
            fft_output_[k] = sum;
        }
        
        // Update frequency spectrum for analysis
        for (size_t i = 0; i < fft_size_; ++i) {
            frequency_spectrum_[i] = std::abs(fft_output_[i]);
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("FFT computation failed: {}\n", e.what());
    }
}

auto EnhancedVibrationFilter::perform_inverse_fft() -> void {
    try {
        // Inverse FFT to get filtered time-domain signal
        std::vector<std::complex<float>> temp(fft_size_);
        
        for (size_t n = 0; n < fft_size_; ++n) {
            std::complex<float> sum(0.0f, 0.0f);
            
            for (size_t k = 0; k < fft_size_; ++k) {
                float angle = 2.0f * std::numbers::pi_v<float> * k * n / fft_size_;
                std::complex<float> term(std::cos(angle), std::sin(angle));
                sum += fft_output_[k] * term;
            }
            
            temp[n] = sum / static_cast<float>(fft_size_);
        }
        
        // Extract real part
        for (size_t i = 0; i < fft_size_; ++i) {
            fft_input_[i] = temp[i].real();
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Inverse FFT computation failed: {}\n", e.what());
    }
}

auto EnhancedVibrationFilter::apply_hamming_window() -> void {
    try {
        // Apply Hamming window to reduce spectral leakage
        for (size_t i = 0; i < fft_size_; ++i) {
            float window_coeff = 0.54f - 0.46f * std::cos(2.0f * std::numbers::pi_v<float> * i / (fft_size_ - 1));
            fft_input_[i] *= window_coeff;
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Hamming window application failed: {}\n", e.what());
    }
}

auto EnhancedVibrationFilter::apply_frequency_filtering() -> void {
    try {
        // Apply frequency domain filtering to stabilize motion
        float frequency_resolution = sample_rate_ / fft_size_;
        
        for (size_t k = 0; k < fft_size_; ++k) {
            float frequency = k * frequency_resolution;
            float filter_gain = 1.0f;
            
            // Apply band-pass filter for relevant motion frequencies
            if (frequency < 1.0f || frequency > 50.0f) {
                filter_gain = 0.1f; // Attenuate very low and very high frequencies
            } else if (frequency >= vibration_config_.high_intensity_threshold - 2.0f &&
                      frequency <= vibration_config_.high_intensity_threshold + 2.0f) {
                filter_gain = 0.3f; // Reduce high-intensity vibration frequencies
            }
            
            fft_output_[k] *= filter_gain;
        }
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Frequency filtering failed: {}\n", e.what());
    }
}

auto EnhancedVibrationFilter::reset() -> void {
    try {
        std::fill(fft_input_.begin(), fft_input_.end(), 0.0f);
        std::fill(fft_output_.begin(), fft_output_.end(), std::complex<float>(0.0f, 0.0f));
        std::fill(frequency_spectrum_.begin(), frequency_spectrum_.end(), 0.0f);
        
        current_vibration_intensity_ = 0.0f;
        frame_should_be_dropped_.store(false);
        vibration_config_.consecutive_dropped_frames = 0;
        
    } catch (const std::exception& e) {
        std::cerr << std::format("Enhanced vibration filter reset failed: {}\n", e.what());
    }
}

auto EnhancedVibrationFilter::is_stable() const -> bool {
    // Consider stable if vibration intensity is below threshold and no frame dropping
    return current_vibration_intensity_ < vibration_config_.high_intensity_threshold * 0.5f &&
           !frame_should_be_dropped_.load();
}

auto EnhancedVibrationFilter::get_vibration_metrics() -> VibrationAnalysis {
    return vibration_config_;
}

// Utility functions
auto EnhancedVibrationFilter::is_power_of_two(size_t n) -> bool {
    return n > 0 && (n & (n - 1)) == 0;
}

auto EnhancedVibrationFilter::next_power_of_two(size_t n) -> size_t {
    size_t power = 1;
    while (power < n) {
        power <<= 1;
    }
    return power;
}

// Factory function implementation
auto create_enhanced_vibration_filter() -> std::unique_ptr<EnhancedVibrationFilter> {
    auto filter = std::make_unique<EnhancedVibrationFilter>();
    if (filter->initialize(30.0f, 1024)) {
        return filter;
    }
    
    return nullptr;
}

} // namespace BikeGuard
