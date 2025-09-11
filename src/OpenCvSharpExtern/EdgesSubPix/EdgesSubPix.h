#ifndef __EDGES_SUBPIX_H__
#define __EDGES_SUBPIX_H__
#include <opencv2/opencv.hpp>
#include <vector>

struct Contour
{
    std::vector<cv::Point2f> points;
    std::vector<float> normal_angles;
    std::vector<float> response;
    std::vector<cv::Point> intPoints;
};
// only 8-bit
void EdgesSubPix(const cv::Mat& gray, double alpha, int low, int high,
    std::vector<Contour>& contours, cv::OutputArray hierarchy,
    int mode);

void EdgesSubPix(const cv::Mat& gray, double alpha, int low, int high,
    std::vector<Contour>& contours);

void PrecomputeEdgesSubPix(const cv::Mat& gray, double alpha, cv::Mat& dx, cv::Mat& dy);

void PrecomputeEdgesSubPixBilateral(const cv::Mat& gray, int d, double sigmaColor, double sigmaSpace,
    double gradientAlpha,
    cv::Mat& dx, cv::Mat& dy);


void RefineContourSubPix(const cv::Mat& dx, const cv::Mat& dy,
    const std::vector<cv::Point>& initialContour,
    int searchRadius,
    Contour& refinedContour,
    bool fixCorners = false);

void RefineContoursSubPix(const cv::Mat& dx, const cv::Mat& dy,
    const std::vector<std::vector<cv::Point>>& initialContours,
    int searchRadius,
    std::vector<Contour>& refinedContours,
    bool fixCorners = false);

void RefineContourCentroid(const cv::Mat& gradX, const cv::Mat& gradY,
    const std::vector<cv::Point>& initialContour,
    int searchRadius,
    int windowSize,
    Contour& refinedContour);

void RefineContoursCentroid(const cv::Mat& gradX, const cv::Mat& gradY,
    const std::vector<std::vector<cv::Point>>& initialContours,
    int searchRadius,
    int windowSize,
    std::vector<Contour>& refinedContours);

void PrecomputeGradientsSobel(const cv::Mat& gray, cv::Mat& gradX, cv::Mat& gradY, int ksize = 3);


#endif // __EDGES_SUBPIX_H__
