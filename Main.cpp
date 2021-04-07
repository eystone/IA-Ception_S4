#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

cv::Vec3f getEyeball(cv::Mat& eye, std::vector<cv::Vec3f>& circles) {
    std::vector<int> sums(circles.size(), 0);
    for (int y = 0; y < eye.rows; y++) {
        uchar* ptr = eye.ptr<uchar>(y);
        for (int x = 0; x < eye.cols; x++) {
            int value = static_cast<int>(*ptr);
            for (int i = 0; i < circles.size(); i++) {
                cv::Point center(int(std::round(circles[i][0])), int(std::round(circles[i][1])));
                int radius = int(std::round(circles[i][2]));
                if (std::pow(x - center.x, 2) + std::pow(y - center.y, 2) < std::pow(radius, 2)) {
                    sums[i] += value;
                }
            }
            ++ptr;
        }
    }
    int smallestSum = 9999999;
    int smallestSumIndex = -1;
    for (int i = 0; i < circles.size(); i++) {
        if (sums[i] < smallestSum) {
            smallestSum = sums[i];
            smallestSumIndex = i;
        }
    }
    return circles[smallestSumIndex];
}

std::vector<cv::Point> centers;

cv::Point stabilize(std::vector<cv::Point>& points, int windowSize) {
    float sumX = 0;
    float sumY = 0;
    int count = 0;
    for (int i = std::max(0, (int)(points.size() - windowSize)); i < points.size(); i++) {
        sumX += points[i].x;
        sumY += points[i].y;
        ++count;
    }
    if (count > 0) {
        sumX /= count;
        sumY /= count;
    }
    return cv::Point(sumX, sumY);
}

unsigned detectEyes(cv::Mat& resultFrame, cv::CascadeClassifier& faceCascade, cv::CascadeClassifier& eyeCascade) {

    //////////////////////////////////frame//////////////////////////////////////////////////////

    cv::Mat grayscale,
        frame(resultFrame); // copy and del init frame
//resultFrame.release();
//resultFrame.copySize(frame);
    cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY); // convert image to grayscale
    cv::equalizeHist(grayscale, grayscale); // enhance image contrast

    //////////////////////////////////Faces//////////////////////////////////////////////////////

    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayscale, faces, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(120, 80));
    if (faces.size() == 0) return 0; // none face was detected
    for (int j(0); j < faces.size(); ++j) {
        cv::Mat face = grayscale(faces[j]); // crop the faces
        cv::equalizeHist(face, face);

        //////////////////////////////////eyes//////////////////////////////////////////////////////

        std::vector<cv::Rect> eyes;
        eyeCascade.detectMultiScale(face, eyes, 1.05, 8, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(20, 20)); // same thing as above
        rectangle(resultFrame, faces[j].tl(), faces[j].br(), cv::Scalar(255, 0, 0), 2); //Draw Faces Border
        if (eyes.size() > (faces.size() * 2)) return faces.size(); // a least both eyes were not detected
        for (cv::Rect& eye : eyes) {
            rectangle(resultFrame, faces[j].tl() + eye.tl(), faces[j].tl() + eye.br(), cv::Scalar(0, 255, 0), 2); //Draw eyes border
        }

        for (int i(0); i < eyes.size(); ++i) {
            cv::Mat eye = face(eyes[i]);
            cv::equalizeHist(eye, eye); //engance eye contrast

            //////////////////////////////////Iris//////////////////////////////////////////////////////

            std::vector<cv::Vec3f> circles;
            cv::HoughCircles(eye, circles, cv::HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 2); //search circle(eyes)
            if (circles.size() > 0) {
                cv::Vec3f eyeball = getEyeball(eye, circles);
                cv::Point center(eyeball[0], eyeball[1]);
                centers.push_back(center);
                center = stabilize(centers, 5);
                int radius = int(eyeball[2]);
                cv::circle(resultFrame, faces[j].tl() + eyes[i].tl() + center, radius, cv::Scalar(0, 0, 255), 2);
                cv::circle(eye, center, radius, cv::Scalar(255, 255, 255), 2); //draw iris border
            }
        }
        return faces.size();
    }
}
int main() {
    cv::CascadeClassifier faceCascade, eyeCascade;
    if (!faceCascade.load("haarcascade_frontalface_alt.xml")) {
        std::cerr << "Could not load face detector." << std::endl;
        return -1;
    }
    if (!eyeCascade.load("haarcascade_eye_tree_eyeglasses.xml")) {
        std::cerr << "Could not load eye detector." << std::endl;
        return -1;
    }
    cv::VideoCapture cap(0);
    if (!cap.open(0)) {
        std::cerr << "Webcam not detected." << std::endl;
        return -1;
    }

    cv::Mat initFrame, processFrame;
    while (1) {
        cap >> initFrame
            >> processFrame; // outputs the webcam image to a Mat
        if (!initFrame.data) break;

        if (!initFrame.empty())cv::imshow("init Webcam", initFrame);

        unsigned frontRes(detectEyes(processFrame, faceCascade, eyeCascade));//show get the best result

        if (frontRes > 0) cv::imshow("process Webcam", processFrame); // displays the best result

        if (cv::waitKey(120) >= 0) break; //wait 120 before next initFrame or user input
    }
    return 0;
}
