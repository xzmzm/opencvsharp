#ifndef __EDGES_SUBPIX_H__
#define __EDGES_SUBPIX_H__
#include <opencv2/opencv.hpp>
#include <vector>

struct Contour
{
    std::vector<cv::Point2f> points;
    std::vector<float> direction;
    std::vector<float> response;
};
// only 8-bit
void EdgesSubPix(const cv::Mat& gray, double alpha, int low, int high,
    std::vector<Contour>& contours, cv::OutputArray hierarchy,
    int mode);

void EdgesSubPix(const cv::Mat& gray, double alpha, int low, int high,
    std::vector<Contour>& contours);

void PrecomputeEdgesSubPix(const cv::Mat& gray, double alpha, cv::Mat& dx, cv::Mat& dy);

void RefineContourSubPix(const cv::Mat& dx, const cv::Mat& dy,
    const std::vector<cv::Point>& initialContour,
    int searchRadius,
    Contour& refinedContour);

void RefineContoursSubPix(const cv::Mat& dx, const cv::Mat& dy,
    const std::vector<std::vector<cv::Point>>& initialContours,
    int searchRadius,
    std::vector<Contour>& refinedContours);

#endif // __EDGES_SUBPIX_H__
