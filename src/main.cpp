#include "bikeguard_engine.hpp"
#include "calibration_manager.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>

int main(int argc, char* argv[]) {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "    BikeGuard Native Windows Engine     " << std::endl;
        std::cout << "    C++20 + DirectML + Media Foundation " << std::endl;
        std::cout << "========================================" << std::endl;

        // Parse command line arguments
        bool run_benchmark = false;
        bool run_tests = false;
        bool enable_vibration_filtering = false;
        bool run_calibration = false;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--benchmark") run_benchmark = true;
            else if (arg == "--test") run_tests = true;
            else if (arg == "--vibration") enable_vibration_filtering = true;
            else if (arg == "--calibrate") run_calibration = true;
            else if (arg == "--help") {
                std::cout << "Usage: BikeGuard [options]\n"
                          << "Options:\n"
                          << "  --benchmark     Run performance benchmarks\n"
                          << "  --test          Run basic functionality tests\n"
                          << "  --vibration     Enable vibration filtering\n"
                          << "  --calibrate     Run interactive calibration mode\n"
                          << "  --help          Show this help\n";
                return 0;
            }
        }

        // Handle calibration mode separately
        if (run_calibration) {
            std::cout << "\nStarting BikeGuard Calibration Mode...\n";
            
            BikeGuard::CalibrationConfig calib_config;
            auto calibration_manager = BikeGuard::create_calibration_manager(calib_config);
            
            if (!calibration_manager) {
                std::cerr << "Failed to create calibration manager\n";
                return -1;
            }
            
            // Load existing config if available
            calibration_manager->load_config();
            
            // Initialize camera
            if (!calibration_manager->initialize_camera(0)) {
                std::cerr << "Failed to initialize camera for calibration\n";
                return -1;
            }
            
            // Run interactive calibration
            calibration_manager->run_calibration();
            
            std::cout << "Calibration mode completed\n";
            return 0;
        }

        // 1. Initialize modern BikeGuard engine
        BikeGuard::BikeGuardEngine engine;
        
        // Configure for optimal Windows performance
        BikeGuard::EngineConfig config;
        config.model_path = "models/helmet_detector.onnx";
        config.input_size = cv::Size(640, 640);
        config.confidence_threshold = 0.5f;
        config.nms_threshold = 0.4f;
        config.use_gpu = true; // DirectML GPU acceleration
        config.intra_op_num_threads = 1;
        config.inter_op_num_threads = 1;

        std::cout << "Initializing BikeGuard engine..." << std::endl;
        std::cout << "GPU acceleration: " << (config.use_gpu ? "Enabled (DirectML)" : "Disabled") << std::endl;
        std::cout << "Vibration filtering: " << (enable_vibration_filtering ? "Enabled" : "Disabled") << std::endl;

        if (!engine.initialize(config)) {
            throw BikeGuard::BikeGuardException(
                BikeGuard::ErrorCode::INFERENCE_FAILED, 
                "Failed to initialize BikeGuard engine");
        }

        // Load calibration configuration if available
        if (engine.load_calibration_config()) {
            std::cout << "Calibration loaded: Exclusion zone active" << std::endl;
        } else {
            std::cout << "No calibration found - run with --calibrate to set exclusion zone" << std::endl;
        }

        // Print GPU information
        if (engine.is_gpu_available()) {
            std::cout << "GPU Info: " << engine.get_gpu_info() << std::endl;
        } else {
            std::cout << "Running in CPU mode - GPU not available" << std::endl;
        }

        // 2. Setup Media Foundation camera for minimal latency
        auto camera = BikeGuard::create_msmf_camera();
        if (!camera) {
            std::cout << "Failed to create Media Foundation camera" << std::endl;
            return -1;
        }

        if (!camera->initialize(0)) {
            throw BikeGuard::BikeGuardException(
                BikeGuard::ErrorCode::CAMERA_INIT_FAILED, 
                "Failed to initialize Media Foundation camera");
        }

        auto frame_size = camera->get_frame_size();
        auto fps = camera->get_fps();
        
        std::cout << "Camera initialized successfully" << std::endl;
        std::cout << "Backend: Media Foundation (CAP_MSMF)" << std::endl;
        std::cout << "Resolution: " << frame_size.width << "x" << frame_size.height << std::endl;
        std::cout << "FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;

        // 3. Setup FFT vibration filtering if requested
        if (enable_vibration_filtering) {
            auto vibration_filter = BikeGuard::create_fft_vibration_filter();
            if (vibration_filter && vibration_filter->initialize(30.0f, 1024)) {
                engine.set_vibration_filter(std::move(vibration_filter));
                engine.enable_vibration_filtering(true);
                std::cout << "FFT vibration filtering enabled" << std::endl;
            } else {
                std::cout << "Failed to initialize vibration filter" << std::endl;
            }
        }

        // 4. Run tests if requested
        if (run_tests) {
            std::cout << "\nRunning basic functionality tests..." << std::endl;
            if (BikeGuard::SimpleTestSuite::run_basic_tests(engine)) {
                std::cout << "All tests passed!" << std::endl;
            } else {
                std::cout << "Some tests failed!" << std::endl;
            }
        }

        // 5. Run benchmarks if requested
        if (run_benchmark) {
            std::cout << "\nRunning performance benchmarks..." << std::endl;
            
            BikeGuard::PerformanceBenchmark::BenchmarkConfig benchmark_config;
            benchmark_config.num_frames = 500;
            benchmark_config.use_gpu = engine.is_gpu_available();
            benchmark_config.enable_vibration_filtering = enable_vibration_filtering;
            benchmark_config.save_detailed_log = true;
            benchmark_config.output_file = "bikeguard_benchmark_results.json";
            
            auto results = BikeGuard::PerformanceBenchmark::run_full_benchmark(engine, benchmark_config);
            BikeGuard::PerformanceBenchmark::print_benchmark_results(results);
            
            std::cout << "Benchmark completed. Press Enter to continue to real-time detection..." << std::endl;
            std::cin.get();
        }

        // 6. Real-time detection loop
        std::cout << "\nStarting real-time helmet detection..." << std::endl;
        std::cout << "Controls:" << std::endl;
        std::cout << "  'q' or ESC - Quit" << std::endl;
        std::cout << "  'r' - Reset performance metrics" << std::endl;
        std::cout << "  's' - Save current frame" << std::endl;
        std::cout << "  'g' - Toggle GPU/CPU mode" << std::endl;
        std::cout << "----------------------------------------" << std::endl;

        cv::Mat frame;
        int frame_count = 0;
        const int metrics_interval = 30;
        bool gpu_mode = engine.is_gpu_available();

        // Set high thread priority for real-time performance
        BikeGuard::ThreadAffinityManager::set_current_thread_priority(THREAD_PRIORITY_ABOVE_NORMAL);

        while (camera->capture_frame(frame)) {
            if (frame.empty()) break;

            // Apply calibration exclusion zone if configured
            engine.apply_calibration_exclusion(frame);

            // Run inference with zero-allocation design
            auto detections = engine.run_inference(frame);

            // Draw detection results using modern span-based interface
            std::span<const BikeGuard::DetectionResult> detection_span(detections);
            BikeGuard::utils::draw_detections(frame, detection_span);

            // Create comprehensive status overlay
            auto metrics = engine.get_metrics();
            auto [fps_val, inference_time, preprocess_time, postprocess_time, frame_count_val, detection_count] = metrics.get_snapshot();
            
            // System resource monitoring
            auto system_info = BikeGuard::WindowsPerformanceMonitor::get_system_info();
            
            std::string status_text = std::format(
                "BikeGuard Engine | FPS: {:.1f} | Inference: {:.1f}ms | Detections: {} | CPU: {:.1f}% | Memory: {}MB",
                fps_val, inference_time, detections.size(), system_info.cpu_usage_percent, system_info.memory_usage_mb
            );
            
            cv::putText(frame, status_text, cv::Point(10, 30), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            std::string mode_text = std::format("Mode: {} | Vibration: {}", 
                                              gpu_mode ? "GPU (DirectML)" : "CPU",
                                              enable_vibration_filtering ? "ON" : "OFF");
            cv::putText(frame, mode_text, cv::Point(10, 55), 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

            // Display frame
            cv::imshow("BikeGuard - Native Windows Engine", frame);

            // Print performance metrics periodically
            frame_count++;
            if (frame_count % metrics_interval == 0) {
                std::cout << std::format("Performance: {:.1f} FPS | {:.2f}ms inference | {} detections | {:.1f}% CPU | {}MB memory\n",
                                        fps_val, inference_time, detections.size(), system_info.cpu_usage_percent, system_info.memory_usage_mb);
            }

            // Handle keyboard input
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break; // 'q' or ESC
            else if (key == 'r') {
                engine.reset_metrics();
                frame_count = 0;
                std::cout << "Performance metrics reset" << std::endl;
            }
            else if (key == 's') {
                std::string filename = std::format("bikeguard_capture_{}.jpg", 
                                                 std::chrono::duration_cast<std::chrono::seconds>(
                                                     std::chrono::system_clock::now().time_since_epoch()).count());
                cv::imwrite(filename, frame);
                std::cout << "Frame saved: " << filename << std::endl;
            }
            else if (key == 'g') {
                // Toggle GPU/CPU mode (requires reinitialization)
                gpu_mode = !gpu_mode;
                config.use_gpu = gpu_mode;
                if (engine.initialize(config)) {
                    std::cout << "Switched to " << (gpu_mode ? "GPU (DirectML)" : "CPU") << " mode" << std::endl;
                } else {
                    std::cout << "Failed to switch modes" << std::endl;
                    gpu_mode = !gpu_mode;
                }
            }
        }

        // 7. Cleanup
        std::cout << "\nShutting down BikeGuard engine..." << std::endl;
        
        // Print final statistics
        auto final_metrics = engine.get_metrics();
        auto [final_fps, final_inference_time, final_preprocess_time, final_postprocess_time, final_frame_count, final_detection_count] = final_metrics.get_snapshot();
        
        std::cout << std::format("Final Statistics: {:.1f} FPS | {:.2f}ms avg inference | {} total detections\n",
                                final_fps, final_inference_time, final_detection_count);
        
        camera->release();
        engine.cleanup();
        cv::destroyAllWindows();

        std::cout << "BikeGuard terminated successfully" << std::endl;
        return 0;

    } catch (const BikeGuard::BikeGuardException& e) {
        std::cerr << std::format("BikeGuard Error [{}]: {}\n", static_cast<int>(e.get_error_code()), e.what());
        return -1;
    } catch (const std::exception& e) {
        std::cerr << std::format("System Error: {}\n", e.what());
        return -1;
    }
}