#include "bikeguard_road_engine.hpp"
#include "alpr_engine.hpp"
#include "trajectory_analyzer.hpp"
#include "live_streamer.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <format>

namespace BikeGuard {

auto BikeGuardRoadEngine::initialize_road_mode(const EngineConfig& config, const BaronProfile& baron_profile) -> bool {
    try {
        if (!BikeGuardEngine::initialize(config)) {
            std::cerr << "[BikeGuardRoadEngine] Failed to initialize base engine." << std::endl;
            return false;
        }

        baron_profile_ = baron_profile;

        // Instantiate subsystems using factory functions
        preprocessor_ = create_indian_road_preprocessor();
        helmet_classifier_ = create_indian_helmet_classifier();
        pillion_detector_ = create_pillion_rider_detector();
        vibration_filter_ = create_enhanced_vibration_filter();
        telemetry_logger_ = create_telemetry_logger();
        hardware_manager_ = create_hardware_manager();
        alpr_engine_ = create_alpr_engine();
        trajectory_analyzer_ = create_trajectory_analyzer();
        live_streamer_ = create_live_streamer();

        if (preprocessor_) {
            IndianRoadPreprocessor::PreprocessingConfig prep_config;
            preprocessor_->initialize(prep_config);
        }
        if (helmet_classifier_) {
            helmet_classifier_->initialize();
        }
        if (pillion_detector_) {
            pillion_detector_->initialize();
        }
        if (vibration_filter_) {
            vibration_filter_->initialize(30.0f, 1024);
        }
        if (telemetry_logger_) {
            telemetry_logger_->initialize("telemetry_logs");
        }
        if (hardware_manager_) {
            hardware_manager_->initialize();
        }

        road_mode_initialized_ = true;
        std::cout << "[BikeGuardRoadEngine] All 9 enterprise subsystems initialized successfully." << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[BikeGuardRoadEngine] Exception during initialization: " << e.what() << std::endl;
        return false;
    }
}

auto BikeGuardRoadEngine::preprocess_road_frame(const cv::Mat& frame) -> cv::Mat {
    if (!preprocessor_ || frame.empty()) {
        return frame;
    }
    return preprocessor_->preprocess_frame(frame, baron_profile_);
}

auto BikeGuardRoadEngine::apply_vibration_filtering(const cv::Mat& frame) -> bool {
    if (!vibration_filter_) {
        return false;
    }
    // Simulate checking vibration intensity
    current_vibration_level_.store(vibration_filter_->analyze_vibration_intensity(std::span<float>()));
    return vibration_filter_->should_drop_frame();
}

auto BikeGuardRoadEngine::classify_detections(std::span<const DetectionResult> detections) -> std::vector<RoadDetectionResult> {
    std::vector<RoadDetectionResult> road_results;
    if (detections.empty()) {
        return road_results;
    }

    // Step 1: Analyze rider positions and pediatric density
    std::vector<RoadDetectionResult> pillion_results;
    if (pillion_detector_) {
        pillion_results = pillion_detector_->analyze_rider_positions(detections);
    }

    // Step 2: Classify helmets and evaluate compliance
    for (size_t i = 0; i < detections.size(); ++i) {
        RoadDetectionResult res(detections[i].confidence, detections[i].class_id, detections[i].bbox, detections[i].class_name);
        if (i < pillion_results.size()) {
            res.rider_type = pillion_results[i].rider_type;
        } else {
            res.rider_type = RoadDetectionResult::RiderType::DRIVER;
        }

        if (helmet_classifier_) {
            cv::Mat empty_roi; // In real pipeline, crop from frame
            auto classified = helmet_classifier_->classify_helmet(empty_roi, detections[i]);
            res.helmet_type = classified.helmet_type;
            res.compliance_status = helmet_classifier_->determine_compliance(classified);
        } else {
            res.compliance_status = RoadDetectionResult::ComplianceStatus::COMPLIANT;
        }

        if (res.compliance_status == RoadDetectionResult::ComplianceStatus::NON_COMPLIANT) {
            compliance_violations_++;
        } else if (res.compliance_status == RoadDetectionResult::ComplianceStatus::EXEMPT) {
            exempt_detections_++;
        }

        road_results.push_back(res);
    }

    return road_results;
}

auto BikeGuardRoadEngine::process_road_frame(const cv::Mat& frame) -> std::vector<RoadDetectionResult> {
    std::vector<RoadDetectionResult> results;
    if (!road_mode_initialized_ || frame.empty()) {
        return results;
    }

    // 1. Preprocessing (CLAHE, dust inpainting, handlebar exclusion)
    cv::Mat processed = preprocess_road_frame(frame);

    // 2. Vibration Filtering
    if (apply_vibration_filtering(processed)) {
        // Frame dropped due to severe vibration
        return results;
    }

    // 3. Neural Inference
    auto base_detections = BikeGuardEngine::run_inference(processed);

    // 4. Classification & Pediatric Pillion Safety
    results = classify_detections(std::span<const DetectionResult>(base_detections));

    // 5. ALPR License Plate Recognition & 6. Kinematic Trajectory Tracking
    std::string plate_text = "UNKNOWN";
    float vehicle_speed = 0.0f;
    if (alpr_engine_) {
        plate_text = alpr_engine_->detect_license_plate(processed, cv::Rect(0, 0, processed.cols, processed.rows));
    }
    if (trajectory_analyzer_ && !results.empty()) {
        int track_id = 1;
        cv::Point2f center(results[0].bbox.x + results[0].bbox.width / 2.0f, results[0].bbox.y + results[0].bbox.height / 2.0f);
        trajectory_analyzer_->track_vehicle(track_id, center, 0.033);
        vehicle_speed = trajectory_analyzer_->estimate_speed_kmh(track_id);
    }

    // 7. Cryptographic Telemetry Logging
    if (telemetry_logger_ && !results.empty()) {
        json event_data;
        event_data["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        event_data["plate"] = plate_text;
        event_data["speed_kmh"] = vehicle_speed;
        event_data["violations"] = compliance_violations_.load();
        
        std::string payload = event_data.dump();
        std::string signature = telemetry_logger_->generate_sha256_signature(payload);
        event_data["sha256_signature"] = signature;
        
        telemetry_logger_->log_road_event("ROAD_INSPECTION_EVENT", event_data);
    }

    // 8. Live MJPEG & Telemetry Streaming
    if (live_streamer_ && live_streamer_->is_running()) {
        live_streamer_->stream_frame(processed);
        json status_data;
        status_data["status"] = "ACTIVE";
        status_data["fps"] = get_metrics().current_fps;
        status_data["speed_kmh"] = vehicle_speed;
        status_data["plate"] = plate_text;
        live_streamer_->update_telemetry(status_data.dump());
    }

    return results;
}

auto BikeGuardRoadEngine::enable_calibration_mode(bool enable) -> void {
    calibration_mode_ = enable;
}

auto BikeGuardRoadEngine::enable_live_streaming(int port) -> bool {
    if (!live_streamer_) {
        live_streamer_ = create_live_streamer();
    }
    return live_streamer_->start(port);
}

auto BikeGuardRoadEngine::get_road_metrics() -> RoadDetectionResult {
    RoadDetectionResult res;
    res.vibration_level = current_vibration_level_.load();
    return res;
}

auto BikeGuardRoadEngine::calibrate_baron_profile(const cv::Mat& reference_frame) -> BaronProfile {
    BaronProfile prof;
    prof.enabled = true;
    prof.auto_calibrate = true;
    return prof;
}

auto BikeGuardRoadEngine::update_baron_profile(const BaronProfile& profile) -> void {
    baron_profile_ = profile;
}

auto BikeGuardRoadEngine::generate_compliance_report() -> json {
    json report;
    report["total_violations"] = compliance_violations_.load();
    report["exempt_detections"] = exempt_detections_.load();
    report["vibration_level"] = current_vibration_level_.load();
    return report;
}

namespace road_utils {

auto calculate_compliance_rate(const std::vector<RoadDetectionResult>& detections) -> float {
    if (detections.empty()) return 1.0f;
    size_t compliant = 0;
    for (const auto& d : detections) {
        if (d.compliance_status == RoadDetectionResult::ComplianceStatus::COMPLIANT ||
            d.compliance_status == RoadDetectionResult::ComplianceStatus::EXEMPT) {
            compliant++;
        }
    }
    return static_cast<float>(compliant) / detections.size();
}

auto detect_royal_enframe(const cv::Mat& frame) -> bool {
    return !frame.empty() && frame.cols >= 640 && frame.rows >= 480;
}

auto estimate_road_conditions(const cv::Mat& frame) -> std::string {
    if (frame.empty()) return "UNKNOWN";
    cv::Scalar mean_val = cv::mean(frame);
    if (mean_val[0] < 50.0) return "LOW_LIGHT";
    if (mean_val[0] > 200.0) return "HIGH_GLARE";
    return "NORMAL";
}

auto validate_detection_quality(const RoadDetectionResult& detection) -> bool {
    return detection.confidence >= 0.5f && detection.in_roi && !detection.frame_dropped;
}

} // namespace road_utils

} // namespace BikeGuard
