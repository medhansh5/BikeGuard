#pragma once

#include "bikeguard_core.hpp"
#include <opencv2/opencv.hpp>
#include <string>
#include <string_view>
#include <vector>
#include <regex>
#include <memory>
#include <chrono>
#include <span>

namespace BikeGuard {

// Structure representing a detected Indian High Security Registration Plate (HSRP)
struct LicensePlateResult {
    std::string registration_number;
    float confidence{0.0f};
    cv::Rect bbox;
    std::string state_code;
    bool is_hsrp_compliant{false};
    bool detected{false};
    
    LicensePlateResult() = default;
    LicensePlateResult(std::string_view reg_num, float conf, const cv::Rect& box, std::string_view state)
        : registration_number(reg_num), confidence(conf), bbox(box), state_code(state), detected(true) {}
};

// Automatic License Plate Recognition (ALPR) Engine for Indian Road Conditions
class ALPREngine {
public:
    struct ALPRConfig {
        float min_plate_confidence{0.65f};
        bool enable_hsrp_validation{true};
        bool apply_perspective_correction{true};
        bool apply_mud_cleaning{true};
        float max_plate_tilt_degrees{25.0f};
    };

    ALPREngine() = default;
    ~ALPREngine() = default;

    auto initialize(const ALPRConfig& config = ALPRConfig{}) -> bool;
    auto detect_license_plate(const cv::Mat& frame, const cv::Rect& motorcycle_bbox) -> LicensePlateResult;
    auto validate_hsrp_format(std::string_view plate_text) -> bool;
    auto extract_state_code(std::string_view plate_text) -> std::string;
    auto get_state_name(std::string_view state_code) -> std::string;

private:
    bool initialized_{false};
    ALPRConfig config_;
    
    // Indian RTO State codes map
    std::vector<std::pair<std::string, std::string>> state_rto_codes_;
    std::regex hsrp_regex_;
    
    // Internal processing pipelines (Zero-allocation reusable buffers)
    cv::Mat gray_buffer_;
    cv::Mat thresh_buffer_;
    cv::Mat morph_kernel_;
    
    // Helper processing methods
    auto localize_plate_region(const cv::Mat& roi) -> cv::Rect;
    auto clean_dirty_plate(cv::Mat& plate_roi) -> void;
    auto correct_perspective(const cv::Mat& input, const cv::Rect& plate_box) -> cv::Mat;
    auto segment_characters(const cv::Mat& plate_img) -> std::vector<cv::Rect>;
    auto recognize_characters(const cv::Mat& plate_img, const std::vector<cv::Rect>& char_boxes) -> std::string;
    auto setup_rto_codes() -> void;
};

auto create_alpr_engine() -> std::unique_ptr<ALPREngine>;

} // namespace BikeGuard
