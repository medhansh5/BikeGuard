#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // 1. Initialize Camera
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cerr << "Error: Camera not found!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame; // Non-blocking capture
        if (frame.empty()) break;

        // 2. Pre-processing for AI (YOLO usually needs 640x640)
        cv::Mat blob;
        cv::Size modelInputSize(640, 640);
        cv::resize(frame, blob, modelInputSize);

        // 3. Visual Feedback
        cv::rectangle(frame, cv::Point(100, 100), cv::Point(400, 400), cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "BIKEGUARD ACTIVE", cv::Point(20, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        // Display results
        cv::imshow("BikeGuard Engine (C++)", frame);

        if (cv::waitKey(1) == 'q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}