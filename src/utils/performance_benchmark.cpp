#include "bikeguard_engine.hpp"
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <format>
#include <fstream>
#include <thread>

namespace BikeGuard {

// Performance benchmark suite for BikeGuard engine
class PerformanceBenchmark {
public:
    struct BenchmarkResult {
        std::string test_name;
        double avg_fps;
        double min_fps;
        double max_fps;
        double avg_inference_time_ms;
        double min_inference_time_ms;
        double max_inference_time_ms;
        double avg_preprocessing_time_ms;
        double avg_postprocessing_time_ms;
        size_t total_frames;
        size_t total_detections;
        double cpu_usage_percent;
        size_t memory_usage_mb;
        size_t gpu_memory_usage_mb;
    };

    struct BenchmarkConfig {
        size_t num_frames = 1000;
        size_t warmup_frames = 50;
        cv::Size test_resolution{1280, 720};
        std::string model_path = "models/helmet_detector.onnx";
        bool use_gpu = true;
        bool enable_vibration_filtering = false;
        bool save_detailed_log = false;
        std::string output_file = "benchmark_results.json";
    };

    static auto run_full_benchmark(BikeGuardEngine& engine, const BenchmarkConfig& config) -> std::vector<BenchmarkResult> {
        std::vector<BenchmarkResult> results;
        
        // Test 1: CPU-only performance
        if (!config.use_gpu) {
            auto cpu_result = run_inference_benchmark(engine, config, false);
            results.push_back(cpu_result);
        }
        
        // Test 2: GPU performance (DirectML)
        if (config.use_gpu && engine.is_gpu_available()) {
            auto gpu_result = run_inference_benchmark(engine, config, true);
            results.push_back(gpu_result);
        }
        
        // Test 3: Vibration filtering impact
        if (config.enable_vibration_filtering) {
            auto filter_result = run_vibration_filter_benchmark(engine, config);
            results.push_back(filter_result);
        }
        
        // Test 4: Memory usage benchmark
        auto memory_result = run_memory_benchmark(engine, config);
        results.push_back(memory_result);
        
        // Save results if requested
        if (config.save_detailed_log) {
            save_results_to_file(results, config.output_file);
        }
        
        return results;
    }

    static auto run_inference_benchmark(BikeGuardEngine& engine, const BenchmarkConfig& config, bool use_gpu) -> BenchmarkResult {
        BenchmarkResult result{};
        result.test_name = use_gpu ? "GPU_DirectML_Inference" : "CPU_Inference";
        
        // Configure engine
        EngineConfig engine_config = engine.get_config();
        engine_config.use_gpu = use_gpu;
        engine_config.model_path = config.model_path;
        
        std::cout << std::format("Running {} benchmark...\n", result.test_name);
        
        // Create test frames
        std::vector<cv::Mat> test_frames = generate_test_frames(config.num_frames, config.test_resolution);
        
        // Warmup
        std::cout << "Warming up...\n";
        for (size_t i = 0; i < config.warmup_frames && i < test_frames.size(); ++i) {
            try {
                engine.run_inference(test_frames[i]);
            } catch (...) {
                // Ignore warmup errors
            }
        }
        
        // Benchmark loop
        std::vector<double> fps_values;
        std::vector<double> inference_times;
        std::vector<double> preprocessing_times;
        std::vector<double> postprocessing_times;
        size_t total_detections = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (size_t i = config.warmup_frames; i < config.num_frames && i < test_frames.size(); ++i) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            try {
                auto detections = engine.run_inference(test_frames[i]);
                total_detections += detections.size();
                
                auto frame_end = std::chrono::high_resolution_clock::now();
                auto frame_time = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
                
                // Get performance metrics
                auto metrics = engine.get_metrics();
                auto [fps, inference_time, preprocess_time, postprocess_time, frame_count, detection_count] = metrics.get_snapshot();
                
                fps_values.push_back(fps);
                inference_times.push_back(inference_time);
                preprocessing_times.push_back(preprocess_time);
                postprocessing_times.push_back(postprocess_time);
                
            } catch (const std::exception& e) {
                std::cerr << std::format("Frame {} failed: {}\n", i, e.what());
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_time = std::chrono::duration<double>(end_time - start_time).count();
        
        // Calculate statistics
        if (!fps_values.empty()) {
            result.avg_fps = std::accumulate(fps_values.begin(), fps_values.end(), 0.0) / fps_values.size();
            result.min_fps = *std::min_element(fps_values.begin(), fps_values.end());
            result.max_fps = *std::max_element(fps_values.begin(), fps_values.end());
        }
        
        if (!inference_times.empty()) {
            result.avg_inference_time_ms = std::accumulate(inference_times.begin(), inference_times.end(), 0.0) / inference_times.size();
            result.min_inference_time_ms = *std::min_element(inference_times.begin(), inference_times.end());
            result.max_inference_time_ms = *std::max_element(inference_times.begin(), inference_times.end());
        }
        
        if (!preprocessing_times.empty()) {
            result.avg_preprocessing_time_ms = std::accumulate(preprocessing_times.begin(), preprocessing_times.end(), 0.0) / preprocessing_times.size();
        }
        
        if (!postprocessing_times.empty()) {
            result.avg_postprocessing_time_ms = std::accumulate(postprocessing_times.begin(), postprocessing_times.end(), 0.0) / postprocessing_times.size();
        }
        
        result.total_frames = config.num_frames - config.warmup_frames;
        result.total_detections = total_detections;
        
        // Get system resource usage
        auto system_info = WindowsPerformanceMonitor::get_system_info();
        result.cpu_usage_percent = system_info.cpu_usage_percent;
        result.memory_usage_mb = system_info.memory_usage_mb;
        result.gpu_memory_usage_mb = system_info.gpu_memory_usage_mb;
        
        return result;
    }

    static auto run_vibration_filter_benchmark(BikeGuardEngine& engine, const BenchmarkConfig& config) -> BenchmarkResult {
        BenchmarkResult result{};
        result.test_name = "Vibration_Filtering_Impact";
        
        std::cout << "Running vibration filtering benchmark...\n";
        
        // Create test motion data
        std::vector<float> test_motion_data = generate_test_motion_data(1000);
        
        // Test with and without filtering
        auto filter = create_fft_vibration_filter();
        if (!filter) {
            result.test_name += "_Failed";
            return result;
        }
        
        filter->initialize(30.0f, 1024);
        engine.set_vibration_filter(std::move(filter));
        engine.enable_vibration_filtering(true);
        
        // Benchmark filtering performance
        std::vector<double> filter_times;
        const size_t iterations = 1000;
        
        for (size_t i = 0; i < iterations; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            
            std::span<float> motion_span(test_motion_data);
            filter = create_fft_vibration_filter();
            if (filter) {
                filter->initialize(30.0f, 1024);
                filter->filter_frame(motion_span);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double, std::micro>(end - start).count();
            filter_times.push_back(duration);
        }
        
        if (!filter_times.empty()) {
            result.avg_inference_time_ms = std::accumulate(filter_times.begin(), filter_times.end(), 0.0) / filter_times.size() / 1000.0;
            result.min_inference_time_ms = *std::min_element(filter_times.begin(), filter_times.end()) / 1000.0;
            result.max_inference_time_ms = *std::max_element(filter_times.begin(), filter_times.end()) / 1000.0;
        }
        
        result.total_frames = iterations;
        result.test_name += "_Completed";
        
        return result;
    }

    static auto run_memory_benchmark(BikeGuardEngine& engine, const BenchmarkConfig& config) -> BenchmarkResult {
        BenchmarkResult result{};
        result.test_name = "Memory_Usage_Analysis";
        
        std::cout << "Running memory usage benchmark...\n";
        
        // Baseline memory usage
        auto baseline_info = WindowsPerformanceMonitor::get_system_info();
        size_t baseline_memory = baseline_info.memory_usage_mb;
        
        // Create test frames and run inference to allocate memory
        std::vector<cv::Mat> test_frames = generate_test_frames(100, config.test_resolution);
        
        for (const auto& frame : test_frames) {
            try {
                engine.run_inference(frame);
            } catch (...) {
                // Ignore errors for memory test
            }
        }
        
        // Peak memory usage
        auto peak_info = WindowsPerformanceMonitor::get_system_info();
        size_t peak_memory = peak_info.memory_usage_mb;
        
        result.memory_usage_mb = peak_memory - baseline_memory;
        result.gpu_memory_usage_mb = peak_info.gpu_memory_usage_mb;
        result.cpu_usage_percent = peak_info.cpu_usage_percent;
        result.total_frames = test_frames.size();
        
        // Test memory deallocation
        engine.cleanup();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto final_info = WindowsPerformanceMonitor::get_system_info();
        size_t final_memory = final_info.memory_usage_mb;
        
        result.test_name += std::format("_Peak_{}MB_Final_{}MB", result.memory_usage_mb, final_memory - baseline_memory);
        
        return result;
    }

    static auto print_benchmark_results(const std::vector<BenchmarkResult>& results) -> void {
        std::cout << "\n" << std::string(80, '=') << "\n";
        std::cout << "           BIKEGUARD PERFORMANCE BENCHMARK RESULTS\n";
        std::cout << std::string(80, '=') << "\n\n";
        
        for (const auto& result : results) {
            std::cout << std::format("Test: {}\n", result.test_name);
            std::cout << std::format("  FPS: Avg={:.1f} Min={:.1f} Max={:.1f}\n", 
                                    result.avg_fps, result.min_fps, result.max_fps);
            std::cout << std::format("  Inference Time: Avg={:.2f}ms Min={:.2f}ms Max={:.2f}ms\n",
                                    result.avg_inference_time_ms, result.min_inference_time_ms, result.max_inference_time_ms);
            std::cout << std::format("  Preprocessing: {:.2f}ms | Postprocessing: {:.2f}ms\n",
                                    result.avg_preprocessing_time_ms, result.avg_postprocessing_time_ms);
            std::cout << std::format("  Total Frames: {} | Total Detections: {}\n",
                                    result.total_frames, result.total_detections);
            std::cout << std::format("  System: CPU={:.1f}% | Memory={}MB | GPU Memory={}MB\n",
                                    result.cpu_usage_percent, result.memory_usage_mb, result.gpu_memory_usage_mb);
            std::cout << std::string(80, '-') << "\n";
        }
        
        // Performance summary
        if (!results.empty()) {
            const auto& best_fps = *std::max_element(results.begin(), results.end(), 
                [](const BenchmarkResult& a, const BenchmarkResult& b) {
                    return a.avg_fps < b.avg_fps;
                });
            
            const auto& best_inference = *std::min_element(results.begin(), results.end(),
                [](const BenchmarkResult& a, const BenchmarkResult& b) {
                    return a.avg_inference_time_ms < b.avg_inference_time_ms;
                });
            
            std::cout << "\nPERFORMANCE SUMMARY:\n";
            std::cout << std::format("Best FPS: {:.1f} ({})\n", best_fps.avg_fps, best_fps.test_name);
            std::cout << std::format("Fastest Inference: {:.2f}ms ({})\n", best_inference.avg_inference_time_ms, best_inference.test_name);
        }
        
        std::cout << std::string(80, '=') << "\n";
    }

private:
    static auto generate_test_frames(size_t count, const cv::Size& size) -> std::vector<cv::Mat> {
        std::vector<cv::Mat> frames;
        frames.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            cv::Mat frame(size, CV_8UC3);
            
            // Generate synthetic test pattern
            cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            
            // Add some structured patterns
            cv::Rect rect(i % 100, i % 100, 50 + (i % 100), 50 + (i % 100));
            cv::rectangle(frame, rect, cv::Scalar(255, 0, 0), 2);
            
            // Add text
            std::string text = std::format("Frame {}", i);
            cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
            
            frames.push_back(frame);
        }
        
        return frames;
    }

    static auto generate_test_motion_data(size_t count) -> std::vector<float> {
        std::vector<float> data;
        data.reserve(count * 4); // x, y, w, h for each detection
        
        for (size_t i = 0; i < count; ++i) {
            // Simulate motion with some noise
            float x = 100.0f + 50.0f * std::sin(i * 0.1f) + (std::rand() % 10 - 5);
            float y = 100.0f + 30.0f * std::cos(i * 0.15f) + (std::rand() % 10 - 5);
            float w = 50.0f + (std::rand() % 20 - 10);
            float h = 80.0f + (std::rand() % 20 - 10);
            
            data.push_back(x);
            data.push_back(y);
            data.push_back(w);
            data.push_back(h);
        }
        
        return data;
    }

    static auto save_results_to_file(const std::vector<BenchmarkResult>& results, const std::string& filename) -> void {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << std::format("Failed to open {} for writing\n", filename);
            return;
        }
        
        file << "{\n  \"benchmark_results\": [\n";
        
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            
            file << "    {\n";
            file << std::format("      \"test_name\": \"{}\",\n", result.test_name);
            file << std::format("      \"avg_fps\": {:.2f},\n", result.avg_fps);
            file << std::format("      \"min_fps\": {:.2f},\n", result.min_fps);
            file << std::format("      \"max_fps\": {:.2f},\n", result.max_fps);
            file << std::format("      \"avg_inference_time_ms\": {:.3f},\n", result.avg_inference_time_ms);
            file << std::format("      \"min_inference_time_ms\": {:.3f},\n", result.min_inference_time_ms);
            file << std::format("      \"max_inference_time_ms\": {:.3f},\n", result.max_inference_time_ms);
            file << std::format("      \"avg_preprocessing_time_ms\": {:.3f},\n", result.avg_preprocessing_time_ms);
            file << std::format("      \"avg_postprocessing_time_ms\": {:.3f},\n", result.avg_postprocessing_time_ms);
            file << std::format("      \"total_frames\": {},\n", result.total_frames);
            file << std::format("      \"total_detections\": {},\n", result.total_detections);
            file << std::format("      \"cpu_usage_percent\": {:.1f},\n", result.cpu_usage_percent);
            file << std::format("      \"memory_usage_mb\": {},\n", result.memory_usage_mb);
            file << std::format("      \"gpu_memory_usage_mb\": {}\n", result.gpu_memory_usage_mb);
            file << "    }";
            
            if (i < results.size() - 1) {
                file << ",";
            }
            file << "\n";
        }
        
        file << "  ]\n}\n";
        
        file.close();
        std::cout << std::format("Benchmark results saved to {}\n", filename);
    }
};

// Simple test utility for basic functionality
class SimpleTestSuite {
public:
    static auto run_basic_tests(BikeGuardEngine& engine) -> bool {
        std::cout << "Running basic functionality tests...\n";
        
        bool all_passed = true;
        
        // Test 1: Engine initialization
        try {
            EngineConfig config;
            config.model_path = "models/helmet_detector.onnx";
            config.use_gpu = false;
            
            if (!engine.initialize(config)) {
                std::cout << "FAIL: Engine initialization\n";
                all_passed = false;
            } else {
                std::cout << "PASS: Engine initialization\n";
            }
        } catch (const std::exception& e) {
            std::cout << std::format("FAIL: Engine initialization - {}\n", e.what());
            all_passed = false;
        }
        
        // Test 2: Basic inference
        try {
            cv::Mat test_frame(640, 640, CV_8UC3, cv::Scalar(128, 128, 128));
            auto detections = engine.run_inference(test_frame);
            std::cout << std::format("PASS: Basic inference - {} detections\n", detections.size());
        } catch (const std::exception& e) {
            std::cout << std::format("FAIL: Basic inference - {}\n", e.what());
            all_passed = false;
        }
        
        // Test 3: Camera creation
        try {
            auto camera = create_msmf_camera();
            if (camera) {
                std::cout << "PASS: Camera creation\n";
            } else {
                std::cout << "FAIL: Camera creation\n";
                all_passed = false;
            }
        } catch (const std::exception& e) {
            std::cout << std::format("FAIL: Camera creation - {}\n", e.what());
            all_passed = false;
        }
        
        // Test 4: Vibration filter
        try {
            auto filter = create_fft_vibration_filter();
            if (filter) {
                std::cout << "PASS: Vibration filter creation\n";
            } else {
                std::cout << "FAIL: Vibration filter creation\n";
                all_passed = false;
            }
        } catch (const std::exception& e) {
            std::cout << std::format("FAIL: Vibration filter creation - {}\n", e.what());
            all_passed = false;
        }
        
        std::cout << std::format("Basic tests completed: {}\n", all_passed ? "PASSED" : "FAILED");
        return all_passed;
    }
};

} // namespace BikeGuard
