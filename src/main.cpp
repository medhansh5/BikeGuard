#include "bikeguard_road_engine.hpp"
#include "calibration_manager.hpp"
#include "alpr_engine.hpp"
#include "trajectory_analyzer.hpp"
#include "live_streamer.hpp"
#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <format>

int main(int argc, char* argv[]) {
    try {
        std::cout << "==========================================================" << std::endl;
        std::cout << "    BikeGuard v1.2.0 GA Enterprise Physical AI Engine     " << std::endl;
        std::cout << "    C++20 + DirectML + Media Foundation + Live Streaming  " << std::endl;
        std::cout << "==========================================================" << std::endl;

        // Parse command line arguments
        bool run_benchmark = false;
        bool run_tests = false;
        bool enable_vibration_filtering = false;
        bool run_calibration = false;
        bool run_road_mode = true; // Default to Enterprise Road Mode
        int stream_port = 8080;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--benchmark") run_benchmark = true;
            else if (arg == "--test") run_tests = true;
            else if (arg == "--vibration") enable_vibration_filtering = true;
            else if (arg == "--calibrate") run_calibration = true;
            else if (arg == "--basic") run_road_mode = false;
            else if (arg == "--port" && i + 1 < argc) {
                stream_port = std::stoi(argv[++i]);
            }
            else if (arg == "--help") {
                std::cout << "Usage: BikeGuard [options]\n"
                          << "Options:\n"
                          << "  --benchmark     Run performance benchmarks\n"
                          << "  --test          Run basic functionality tests\n"
                          << "  --vibration     Enable vibration filtering\n"
                          << "  --calibrate     Run interactive calibration mode\n"
                          << "  --basic         Run basic engine without road extensions\n"
                          << "  --port <N>      Set Live Streamer port (default: 8080)\n"
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
            
            calibration_manager->load_config();
            if (!calibration_manager->initialize_camera(0)) {
                std::cerr << "Failed to initialize camera for calibration\n";
                return -1;
            }
            
            calibration_manager->run_calibration();
            std::cout << "Calibration mode completed\n";
            return 0;
        }

        // 1. Initialize BikeGuard Road Engine
        BikeGuard::BikeGuardRoadEngine engine;
        
        BikeGuard::EngineConfig config;
        config.model_path = "models/helmet_detector.onnx";
        config.input_size = cv::Size(640, 640);
        config.confidence_threshold = 0.5f;
        config.nms_threshold = 0.4f;
        config.use_gpu = true; // DirectML GPU acceleration
        config.intra_op_num_threads = 1;
        config.inter_op_num_threads = 1;

        std::cout << "Initializing BikeGuard Enterprise Road Engine..." << std::endl;
        std::cout << "GPU acceleration: " << (config.use_gpu ? "Enabled (DirectML)" : "Disabled") << std::endl;
        std::cout << "Vibration filtering: " << (enable_vibration_filtering ? "Enabled" : "Disabled") << std::endl;

        BikeGuard::BaronProfile baron_profile;
        baron_profile.enabled = true;

        if (!engine.initialize_road_mode(config, baron_profile)) {
            std::cerr << "Warning: Road mode initialization returned false, falling back to base init..." << std::endl;
            if (!engine.initialize(config)) {
                throw BikeGuard::BikeGuardException(
                    BikeGuard::ErrorCode::INFERENCE_FAILED, 
                    "Failed to initialize BikeGuard engine");
            }
        }

        // Enable live streaming
        if (engine.enable_live_streaming(stream_port)) {
            std::cout << "\n🌐 Embedded Live MJPEG & REST Telemetry Server Active at http://localhost:" << stream_port << "\n";
        } else {
            std::cout << "\n⚠️ Could not start Live Streamer on port " << stream_port << "\n";
        }

        if (engine.load_calibration_config()) {
            std::cout << "Calibration loaded: Exclusion zone active" << std::endl;
        } else {
            std::cout << "No calibration found - run with --calibrate to set exclusion zone" << std::endl;
        }

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
        
        std::cout << "Camera initialized successfully | Resolution: " << frame_size.width << "x" << frame_size.height << " | FPS: " << std::fixed << std::setprecision(1) << fps << std::endl;

        // 3. Setup FFT vibration filtering if requested
        if (enable_vibration_filtering) {
            auto vibration_filter = BikeGuard::create_fft_vibration_filter();
            if (vibration_filter && vibration_filter->initialize(30.0f, 1024)) {
                engine.set_vibration_filter(std::move(vibration_filter));
                engine.enable_vibration_filtering(true);
                std::cout << "FFT vibration filtering enabled" << std::endl;
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
        std::cout << "\nStarting real-time Physical AI road compliance detection..." << std::endl;
        std::cout << "Controls: 'q'/ESC - Quit | 'r' - Reset metrics | 's' - Save frame | 'g' - Toggle GPU" << std::endl;
        std::cout << "----------------------------------------------------------------------------------" << std::endl;

        cv::Mat frame;
        int frame_count = 0;
        const int metrics_interval = 30;
        bool gpu_mode = engine.is_gpu_available();

        BikeGuard::ThreadAffinityManager::set_current_thread_priority(THREAD_PRIORITY_ABOVE_NORMAL);

        while (camera->capture_frame(frame)) {
            if (frame.empty()) break;

            engine.apply_calibration_exclusion(frame);

            // Run enterprise road processing pipeline
            auto detections = engine.process_road_frame(frame);

            // Draw detection bounding boxes & custom compliance annotations
            for (const auto& det : detections) {
                cv::Scalar box_color = cv::Scalar(0, 255, 0); // Green for compliant
                std::string status_tag = "COMPLIANT";
                
                if (det.compliance_status == BikeGuard::RoadDetectionResult::ComplianceStatus::NON_COMPLIANT) {
                    box_color = cv::Scalar(0, 0, 255); // Red for violation
                    status_tag = "VIOLATION";
                } else if (det.compliance_status == BikeGuard::RoadDetectionResult::ComplianceStatus::EXEMPT) {
                    box_color = cv::Scalar(255, 200, 0); // Cyan/Gold for exempt (Sikh Turban)
                    status_tag = "EXEMPT (TURBAN)";
                }

                std::string rider_tag = "DRIVER";
                if (det.rider_type == BikeGuard::RoadDetectionResult::RiderType::PILLION) rider_tag = "PILLION";
                else if (det.rider_type == BikeGuard::RoadDetectionResult::RiderType::PEDIATRIC_PILLION) rider_tag = "CHILD (<4Y)";

                cv::rectangle(frame, det.bbox, box_color, 2);
                
                std::string label = std::format("{} [{}] ({:.1f}%)", rider_tag, status_tag, det.confidence * 100.0f);
                int baseLine = 0;
                cv::Size label_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::rectangle(frame, cv::Rect(cv::Point(det.bbox.x, det.bbox.y - label_size.height - 5),
                              cv::Size(label_size.width, label_size.height + 5)), box_color, cv::FILLED);
                cv::putText(frame, label, cv::Point(det.bbox.x, det.bbox.y - 5),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
            }

            // Status overlay
            auto metrics = engine.get_metrics();
            auto [fps_val, inference_time, preprocess_time, postprocess_time, frame_count_val, detection_count] = metrics.get_snapshot();
            auto system_info = BikeGuard::WindowsPerformanceMonitor::get_system_info();
            
            std::string status_text = std::format(
                "BikeGuard v1.2.0 GA | FPS: {:.1f} | Inf: {:.1f}ms | Dets: {} | CPU: {:.1f}% | Mem: {}MB",
                fps_val, inference_time, detections.size(), system_info.cpu_usage_percent, system_info.memory_usage_mb
            );
            cv::putText(frame, status_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

            std::string mode_text = std::format("Mode: Enterprise Road Engine | Stream: Port {} | SHA-256 Audit: ACTIVE", stream_port);
            cv::putText(frame, mode_text, cv::Point(10, 55), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

            cv::imshow("BikeGuard - Enterprise Physical AI Road Engine", frame);

            frame_count++;
            if (frame_count % metrics_interval == 0) {
                std::cout << std::format("Performance: {:.1f} FPS | {:.2f}ms inf | {} dets | {:.1f}% CPU | {}MB mem\n",
                                        fps_val, inference_time, detections.size(), system_info.cpu_usage_percent, system_info.memory_usage_mb);
            }

            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q' || key == 27) break;
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
                gpu_mode = !gpu_mode;
                config.use_gpu = gpu_mode;
                if (engine.initialize_road_mode(config, baron_profile)) {
                    std::cout << "Switched to " << (gpu_mode ? "GPU (DirectML)" : "CPU") << " mode" << std::endl;
                } else {
                    gpu_mode = !gpu_mode;
                }
            }
        }

        // 7. Cleanup
        std::cout << "\nShutting down BikeGuard Enterprise Road Engine..." << std::endl;
        auto final_metrics = engine.get_metrics();
        auto [final_fps, final_inference_time, final_preprocess_time, final_postprocess_time, final_frame_count, final_detection_count] = final_metrics.get_snapshot();
        std::cout << std::format("Final Statistics: {:.1f} FPS | {:.2f}ms avg inf | {} total detections\n",
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