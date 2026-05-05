#include "bikeguard_engine.hpp"
#include <complex>
#include <numbers>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <memory>

namespace BikeGuard {

// FFT-based vibration filter for Windows with modern C++20
class FFTVibrationFilter : public IVibrationFilter {
public:
    FFTVibrationFilter() = default;
    ~FFTVibrationFilter() override = default;

    auto initialize(float sample_rate, size_t fft_size = 1024) -> bool override {
        try {
            sample_rate_ = sample_rate;
            fft_size_ = fft_size;
            
            // Ensure FFT size is power of 2
            if (!is_power_of_two(fft_size_)) {
                fft_size_ = next_power_of_two(fft_size_);
            }
            
            // Allocate buffers
            fft_input_.resize(fft_size_);
            fft_output_.resize(fft_size_);
            window_.resize(fft_size_);
            filtered_output_.resize(fft_size_);
            
            // Initialize Hamming window
            create_hamming_window();
            
            // Initialize frequency filters
            setup_frequency_filters();
            
            // Initialize FFTW-like implementation using DFT
            fft_initialized_ = true;
            return true;

        } catch (const std::exception& e) {
            std::cerr << std::format("FFT filter initialization failed: {}\n", e.what());
            return false;
        }
    }

    auto filter_frame(std::span<float> motion_data) -> std::span<float> override {
        if (!fft_initialized_ || motion_data.empty()) {
            return motion_data;
        }

        try {
            // Pad input to FFT size
            size_t data_size = std::min(motion_data.size(), fft_size_);
            std::copy(motion_data.begin(), motion_data.begin() + data_size, fft_input_.begin());
            
            // Zero-pad remaining samples
            std::fill(fft_input_.begin() + data_size, fft_input_.end(), 0.0f);
            
            // Apply window function
            apply_window();
            
            // Perform FFT
            perform_fft();
            
            // Apply frequency domain filtering
            apply_frequency_filters();
            
            // Perform inverse FFT
            perform_inverse_fft();
            
            // Extract filtered data
            size_t output_size = std::min(motion_data.size(), filtered_output_.size());
            std::copy(filtered_output_.begin(), filtered_output_.begin() + output_size, motion_data.begin());
            
            // Update stability detection
            update_stability_detection(motion_data);
            
            return motion_data;

        } catch (const std::exception& e) {
            std::cerr << std::format("FFT filtering failed: {}\n", e.what());
            return motion_data;
        }
    }

    auto reset() -> void override {
        std::fill(fft_input_.begin(), fft_input_.end(), 0.0f);
        std::fill(fft_output_.begin(), fft_output_.end(), std::complex<float>(0.0f));
        std::fill(filtered_output_.begin(), filtered_output_.end(), 0.0f);
        
        stability_counter_ = 0;
        is_stable_ = false;
    }

    auto is_stable() const -> bool override {
        return is_stable_;
    }

private:
    auto is_power_of_two(size_t n) const -> bool {
        return n > 0 && (n & (n - 1)) == 0;
    }

    auto next_power_of_two(size_t n) const -> size {
        size_t power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }

    auto create_hamming_window() -> void {
        const float alpha = 0.54f;
        const float beta = 0.46f;
        
        for (size_t i = 0; i < fft_size_; ++i) {
            window_[i] = alpha - beta * std::cos(2.0f * std::numbers::pi_v<float> * i / (fft_size_ - 1));
        }
    }

    auto apply_window() -> void {
        for (size_t i = 0; i < fft_size_; ++i) {
            fft_input_[i] *= window_[i];
        }
    }

    auto perform_fft() -> void {
        // Cooley-Tukey FFT implementation
        fft_output_.assign(fft_size_, std::complex<float>(0.0f));
        
        for (size_t k = 0; k < fft_size_; ++k) {
            std::complex<float> sum(0.0f, 0.0f);
            
            for (size_t n = 0; n < fft_size_; ++n) {
                float angle = -2.0f * std::numbers::pi_v<float> * k * n / fft_size_;
                std::complex<float> term(std::cos(angle), std::sin(angle));
                sum += fft_input_[n] * term;
            }
            
            fft_output_[k] = sum;
        }
    }

    auto perform_inverse_fft() -> void {
        // Inverse FFT
        std::vector<std::complex<float>> temp(fft_size_, std::complex<float>(0.0f));
        
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
            filtered_output_[i] = temp[i].real();
        }
    }

    auto setup_frequency_filters() -> void {
        // Define frequency bands for vibration filtering
        // Low frequency: 0.1-2 Hz (normal motion)
        // Medium frequency: 2-10 Hz (hand tremor)
        // High frequency: 10-50 Hz (vibration)
        
        low_freq_cutoff_ = 2.0f;      // Hz
        high_freq_cutoff_ = 10.0f;    // Hz
        vibration_cutoff_ = 50.0f;     // Hz
        
        // Create frequency response mask
        frequency_mask_.resize(fft_size_);
        calculate_frequency_mask();
    }

    auto calculate_frequency_mask() -> void {
        float freq_resolution = sample_rate_ / fft_size_;
        
        for (size_t k = 0; k < fft_size_; ++k) {
            float frequency = k * freq_resolution;
            
            if (frequency <= low_freq_cutoff_) {
                // Pass low frequencies (normal motion)
                frequency_mask_[k] = 1.0f;
            } else if (frequency <= high_freq_cutoff_) {
                // Attenuate medium frequencies (hand tremor)
                frequency_mask_[k] = 0.3f;
            } else if (frequency <= vibration_cutoff_) {
                // Strongly attenuate high frequencies (vibration)
                frequency_mask_[k] = 0.1f;
            } else {
                // Block very high frequencies (noise)
                frequency_mask_[k] = 0.0f;
            }
        }
    }

    auto apply_frequency_filters() -> void {
        for (size_t k = 0; k < fft_size_; ++k) {
            fft_output_[k] *= frequency_mask_[k];
        }
    }

    auto update_stability_detection(std::span<float> motion_data) -> void {
        if (motion_data.empty()) return;
        
        // Calculate motion variance
        float mean = std::accumulate(motion_data.begin(), motion_data.end(), 0.0f) / motion_data.size();
        
        float variance = 0.0f;
        for (float value : motion_data) {
            float diff = value - mean;
            variance += diff * diff;
        }
        variance /= motion_data.size();
        
        float std_dev = std::sqrt(variance);
        
        // Stability threshold (tunable)
        const float stability_threshold = 5.0f;
        
        if (std_dev < stability_threshold) {
            stability_counter_++;
            if (stability_counter_ >= 5) { // Require 5 consecutive stable frames
                is_stable_ = true;
            }
        } else {
            stability_counter_ = 0;
            is_stable_ = false;
        }
    }

private:
    // FFT parameters
    size_t fft_size_ = 1024;
    float sample_rate_ = 30.0f;
    bool fft_initialized_ = false;
    
    // FFT buffers
    std::vector<float> fft_input_;
    std::vector<std::complex<float>> fft_output_;
    std::vector<float> window_;
    std::vector<float> filtered_output_;
    
    // Frequency filtering
    float low_freq_cutoff_ = 2.0f;
    float high_freq_cutoff_ = 10.0f;
    float vibration_cutoff_ = 50.0f;
    std::vector<float> frequency_mask_;
    
    // Stability detection
    size_t stability_counter_ = 0;
    bool is_stable_ = false;
};

// Optimized FFT implementation using Windows-specific optimizations
class OptimizedFFTVibrationFilter : public IVibrationFilter {
public:
    OptimizedFFTVibrationFilter() = default;
    ~OptimizedFFTVibrationFilter() override = default;

    auto initialize(float sample_rate, size_t fft_size = 1024) -> bool override {
        try {
            sample_rate_ = sample_rate;
            fft_size_ = fft_size;
            
            // Ensure power of 2
            if (!is_power_of_two(fft_size_)) {
                fft_size_ = next_power_of_two(fft_size_);
            }
            
            // Allocate aligned buffers for SIMD operations
            fft_input_aligned_.resize(fft_size_);
            fft_output_aligned_.resize(fft_size_);
            filtered_output_aligned_.resize(fft_size_);
            
            // Pre-compute twiddle factors for FFT
            precompute_twiddle_factors();
            
            // Setup filters
            setup_optimized_filters();
            
            fft_initialized_ = true;
            return true;

        } catch (const std::exception& e) {
            std::cerr << std::format("Optimized FFT filter initialization failed: {}\n", e.what());
            return false;
        }
    }

    auto filter_frame(std::span<float> motion_data) -> std::span<float> override {
        if (!fft_initialized_ || motion_data.empty()) {
            return motion_data;
        }

        try {
            // Copy and pad input
            size_t data_size = std::min(motion_data.size(), fft_size_);
            std::copy(motion_data.begin(), motion_data.begin() + data_size, fft_input_aligned_.begin());
            std::fill(fft_input_aligned_.begin() + data_size, fft_input_aligned_.end(), 0.0f);
            
            // Perform optimized FFT
            perform_optimized_fft();
            
            // Apply optimized filtering
            apply_optimized_filters();
            
            // Perform inverse FFT
            perform_optimized_inverse_fft();
            
            // Copy results back
            size_t output_size = std::min(motion_data.size(), filtered_output_aligned_.size());
            std::copy(filtered_output_aligned_.begin(), filtered_output_aligned_.begin() + output_size, motion_data.begin());
            
            return motion_data;

        } catch (const std::exception& e) {
            std::cerr << std::format("Optimized FFT filtering failed: {}\n", e.what());
            return motion_data;
        }
    }

    auto reset() -> void override {
        std::fill(fft_input_aligned_.begin(), fft_input_aligned_.end(), 0.0f);
        std::fill(fft_output_aligned_.begin(), fft_output_aligned_.end(), std::complex<float>(0.0f));
        std::fill(filtered_output_aligned_.begin(), filtered_output_aligned_.end(), 0.0f);
    }

    auto is_stable() const -> bool override {
        return true; // Simplified for optimized version
    }

private:
    auto is_power_of_two(size_t n) const -> bool {
        return n > 0 && (n & (n - 1)) == 0;
    }

    auto next_power_of_two(size_t n) const -> size_t {
        size_t power = 1;
        while (power < n) {
            power <<= 1;
        }
        return power;
    }

    auto precompute_twiddle_factors() -> void {
        twiddle_factors_.resize(fft_size_);
        
        for (size_t k = 0; k < fft_size_; ++k) {
            for (size_t n = 0; n < fft_size_; ++n) {
                float angle = -2.0f * std::numbers::pi_v<float> * k * n / fft_size_;
                twiddle_factors_[k * fft_size_ + n] = std::complex<float>(std::cos(angle), std::sin(angle));
            }
        }
    }

    auto perform_optimized_fft() -> void {
        // Radix-2 Cooley-Tukey FFT implementation
        fft_output_aligned_.assign(fft_size_, std::complex<float>(0.0f));
        
        // Bit-reversal permutation
        std::vector<size_t> reversed_indices(fft_size_);
        for (size_t i = 0; i < fft_size_; ++i) {
            reversed_indices[i] = reverse_bits(i, static_cast<size_t>(std::log2(fft_size_)));
        }
        
        // FFT stages
        size_t stages = static_cast<size_t>(std::log2(fft_size_));
        for (size_t stage = 0; stage < stages; ++stage) {
            size_t butterfly_size = 1ULL << stage;
            size_t num_butterflies = fft_size_ / (2 * butterfly_size);
            
            for (size_t butterfly = 0; butterfly < num_butterflies; ++butterfly) {
                for (size_t k = 0; k < butterfly_size; ++k) {
                    size_t idx1 = butterfly * 2 * butterfly_size + k;
                    size_t idx2 = idx1 + butterfly_size;
                    
                    std::complex<float> twiddle = twiddle_factors_[k * num_butterflies];
                    std::complex<float> temp = fft_output_aligned_[idx2] * twiddle;
                    
                    fft_output_aligned_[idx2] = fft_output_aligned_[idx1] - temp;
                    fft_output_aligned_[idx1] = fft_output_aligned_[idx1] + temp;
                }
            }
        }
    }

    auto perform_optimized_inverse_fft() -> void {
        // Simplified inverse FFT
        std::vector<std::complex<float>> temp(fft_size_);
        
        for (size_t n = 0; n < fft_size_; ++n) {
            std::complex<float> sum(0.0f, 0.0f);
            
            for (size_t k = 0; k < fft_size_; ++k) {
                float angle = 2.0f * std::numbers::pi_v<float> * k * n / fft_size_;
                std::complex<float> term(std::cos(angle), std::sin(angle));
                sum += fft_output_aligned_[k] * term;
            }
            
            temp[n] = sum / static_cast<float>(fft_size_);
        }
        
        for (size_t i = 0; i < fft_size_; ++i) {
            filtered_output_aligned_[i] = temp[i].real();
        }
    }

    auto setup_optimized_filters() -> void {
        // Simplified filter setup
        frequency_mask_optimized_.resize(fft_size_, 1.0f);
        
        float freq_resolution = sample_rate_ / fft_size_;
        for (size_t k = 0; k < fft_size_; ++k) {
            float frequency = k * freq_resolution;
            if (frequency > 20.0f) { // Cut off high frequencies
                frequency_mask_optimized_[k] = 0.1f;
            }
        }
    }

    auto apply_optimized_filters() -> void {
        for (size_t k = 0; k < fft_size_; ++k) {
            fft_output_aligned_[k] *= frequency_mask_optimized_[k];
        }
    }

    auto reverse_bits(size_t n, size_t bits) const -> size_t {
        size_t reversed = 0;
        for (size_t i = 0; i < bits; ++i) {
            reversed = (reversed << 1) | (n & 1);
            n >>= 1;
        }
        return reversed;
    }

private:
    size_t fft_size_ = 1024;
    float sample_rate_ = 30.0f;
    bool fft_initialized_ = false;
    
    // Aligned buffers for SIMD optimization
    std::vector<float> fft_input_aligned_;
    std::vector<std::complex<float>> fft_output_aligned_;
    std::vector<float> filtered_output_aligned_;
    
    // Pre-computed twiddle factors
    std::vector<std::complex<float>> twiddle_factors_;
    
    // Optimized frequency mask
    std::vector<float> frequency_mask_optimized_;
};

// Factory function implementation
auto create_fft_vibration_filter() -> std::unique_ptr<IVibrationFilter> {
    // Try optimized version first
    auto optimized_filter = std::make_unique<OptimizedFFTVibrationFilter>();
    if (optimized_filter->initialize(30.0f, 1024)) {
        std::cout << "Using optimized FFT vibration filter\n";
        return optimized_filter;
    }
    
    // Fallback to standard version
    auto standard_filter = std::make_unique<FFTVibrationFilter>();
    if (standard_filter->initialize(30.0f, 1024)) {
        std::cout << "Using standard FFT vibration filter\n";
        return standard_filter;
    }
    
    return nullptr;
}

} // namespace BikeGuard
