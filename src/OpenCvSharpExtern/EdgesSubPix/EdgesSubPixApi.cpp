#include "../ShapeMatcher/line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include "../ShapeMatcher/cuda_icp/icp.h"
#include "../shapematcher.h"
#include "EdgesSubPix.h"
#include <combaseapi.h> // For CoTaskMemAlloc and CoTaskMemFree

// C-style struct for marshaling
struct Contour_C
{
    cv::Point2f* points;
    int num_points;
    float* normal_angles;
    float* response;
};

CVAPI(ExceptionStatus) cv2ex_EdgesSubPix(
    cv::Mat* gray, double alpha, int low, int high,
    Contour_C** out_contours, int* out_num_contours,
    cv::Mat* hierarchy, int mode)
{
    BEGIN_WRAP
    
    std::vector<Contour> cpp_contours;
    
    if (hierarchy != nullptr)
    {
        EdgesSubPix(*gray, alpha, low, high, cpp_contours, *hierarchy, mode);
    }
    else
    {
        // Passing cv::noArray() if hierarchy is null
        EdgesSubPix(*gray, alpha, low, high, cpp_contours, cv::noArray(), mode);
    }
    
    // Convert result to C-style array of structs
    *out_num_contours = static_cast<int>(cpp_contours.size());
    if (*out_num_contours > 0)
    {
        *out_contours = (Contour_C*)CoTaskMemAlloc(sizeof(Contour_C) * (*out_num_contours));
        for (int i = 0; i < *out_num_contours; ++i)
        {
            Contour_C& c_contour = (*out_contours)[i];
            const auto& cpp_contour = cpp_contours[i];
            
            c_contour.num_points = static_cast<int>(cpp_contour.points.size());
            if (c_contour.num_points > 0)
            {
                size_t points_size = sizeof(cv::Point2f) * c_contour.num_points;
                c_contour.points = (cv::Point2f*)CoTaskMemAlloc(points_size);
                memcpy(c_contour.points, cpp_contour.points.data(), points_size);
                
                size_t data_size = sizeof(float) * c_contour.num_points;
                c_contour.normal_angles = (float*)CoTaskMemAlloc(data_size);
                memcpy(c_contour.normal_angles, cpp_contour.normal_angles.data(), data_size);
                
                c_contour.response = (float*)CoTaskMemAlloc(data_size);
                memcpy(c_contour.response, cpp_contour.response.data(), data_size);
            }
            else
            {
                c_contour.points = nullptr;
                c_contour.normal_angles = nullptr;
                c_contour.response = nullptr;
            }
        }
    }
    else
    {
        *out_contours = nullptr;
    }

    END_WRAP
}

CVAPI(ExceptionStatus) cv2ex_FreeContours(Contour_C* contours, int num_contours)
{
    BEGIN_WRAP
    if (contours != nullptr)
    {
        for (int i = 0; i < num_contours; ++i)
        {
            if(contours[i].points) CoTaskMemFree(contours[i].points);
            if(contours[i].normal_angles) CoTaskMemFree(contours[i].normal_angles);
            if(contours[i].response) CoTaskMemFree(contours[i].response);
        }
        CoTaskMemFree(contours);
    }
    END_WRAP
}

CVAPI(ExceptionStatus) cv2ex_PrecomputeEdgesSubPix(cv::Mat* gray, double alpha, cv::Mat* dx, cv::Mat* dy)
{
    BEGIN_WRAP
    PrecomputeEdgesSubPix(*gray, alpha, *dx, *dy);
    END_WRAP
}

CVAPI(ExceptionStatus) cv2ex_PrecomputeEdgesSubPixBilateral(cv::Mat* gray, int d, double sigmaColor, double sigmaSpace,
    double gradientAlpha, cv::Mat* dx, cv::Mat* dy)
{
    BEGIN_WRAP
    PrecomputeEdgesSubPixBilateral(*gray, d, sigmaColor, sigmaSpace, gradientAlpha, *dx, *dy);
    END_WRAP
}


CVAPI(ExceptionStatus) cv2ex_RefineContourSubPix(
    cv::Mat* dx, cv::Mat* dy, cv::Point* initialContour, int contourLength, int searchRadius,
    bool fixCorners, Contour_C* out_refinedContour)
{
    BEGIN_WRAP

    if (initialContour == nullptr || contourLength <= 0 || out_refinedContour == nullptr)
    {
        if (out_refinedContour)
        {
            out_refinedContour->num_points = 0;
            out_refinedContour->points = nullptr;
            out_refinedContour->normal_angles = nullptr;
            out_refinedContour->response = nullptr;
        }
        return ExceptionStatus::NotOccurred;
    }

    std::vector<cv::Point> cpp_initialContour(initialContour, initialContour + contourLength);
    Contour cpp_refinedContour;

    RefineContourSubPix(*dx, *dy, cpp_initialContour, searchRadius, cpp_refinedContour, fixCorners);

    // Convert result to C-style struct
    out_refinedContour->num_points = static_cast<int>(cpp_refinedContour.points.size());
    if (out_refinedContour->num_points > 0)
    {
        size_t points_size = sizeof(cv::Point2f) * out_refinedContour->num_points;
        out_refinedContour->points = (cv::Point2f*)CoTaskMemAlloc(points_size);
        memcpy(out_refinedContour->points, cpp_refinedContour.points.data(), points_size);

        size_t data_size = sizeof(float) * out_refinedContour->num_points;
        out_refinedContour->normal_angles = (float*)CoTaskMemAlloc(data_size);
        memcpy(out_refinedContour->normal_angles, cpp_refinedContour.normal_angles.data(), data_size);

        out_refinedContour->response = (float*)CoTaskMemAlloc(data_size);
        memcpy(out_refinedContour->response, cpp_refinedContour.response.data(), data_size);
    }
    else
    {
        out_refinedContour->points = nullptr;
        out_refinedContour->normal_angles = nullptr;
        out_refinedContour->response = nullptr;
    }

    END_WRAP
}

CVAPI(ExceptionStatus) cv2ex_RefineContoursSubPix(
    cv::Mat* dx, cv::Mat* dy,
    cv::Point* initialContoursData, int* contourLengths, int numContours,
    int searchRadius, bool fixCorners,
    Contour_C** out_refinedContours, int* out_num_contours)
{
    BEGIN_WRAP

    if (initialContoursData == nullptr || contourLengths == nullptr || numContours <= 0) {
        *out_num_contours = 0;
        *out_refinedContours = nullptr;
        return ExceptionStatus::NotOccurred;
    }

    // Reconstruct contours
    std::vector<std::vector<cv::Point>> cpp_initialContours(numContours);
    cv::Point* current_point_ptr = initialContoursData;
    for (int i = 0; i < numContours; ++i)
    {
        int len = contourLengths[i];
        if (len > 0)
        {
            cpp_initialContours[i].assign(current_point_ptr, current_point_ptr + len);
            current_point_ptr += len;
        }
    }

    std::vector<Contour> cpp_refinedContours;
    RefineContoursSubPix(*dx, *dy, cpp_initialContours, searchRadius, cpp_refinedContours, fixCorners);

    // Convert result to C-style array of structs
    *out_num_contours = static_cast<int>(cpp_refinedContours.size());
    if (*out_num_contours > 0)
    {
        *out_refinedContours = (Contour_C*)CoTaskMemAlloc(sizeof(Contour_C) * (*out_num_contours));
        for (int i = 0; i < *out_num_contours; ++i)
        {
            Contour_C& c_contour = (*out_refinedContours)[i];
            const auto& cpp_contour = cpp_refinedContours[i];

            c_contour.num_points = static_cast<int>(cpp_contour.points.size());
            if (c_contour.num_points > 0)
            {
                size_t points_size = sizeof(cv::Point2f) * c_contour.num_points;
                c_contour.points = (cv::Point2f*)CoTaskMemAlloc(points_size);
                memcpy(c_contour.points, cpp_contour.points.data(), points_size);

                size_t data_size = sizeof(float) * c_contour.num_points;
                c_contour.normal_angles = (float*)CoTaskMemAlloc(data_size);
                memcpy(c_contour.normal_angles, cpp_contour.normal_angles.data(), data_size);

                c_contour.response = (float*)CoTaskMemAlloc(data_size);
                memcpy(c_contour.response, cpp_contour.response.data(), data_size);
            }
            else
            {
                c_contour.points = nullptr;
                c_contour.normal_angles = nullptr;
                c_contour.response = nullptr;
            }
        }
    }
    else
    {
        *out_refinedContours = nullptr;
    }

    END_WRAP
}

CVAPI(ExceptionStatus) cv2ex_FreeContourData(Contour_C* contour)
{
    BEGIN_WRAP
    if (contour != nullptr)
    {
        if (contour->points) CoTaskMemFree(contour->points);
        if (contour->normal_angles) CoTaskMemFree(contour->normal_angles);
        if (contour->response) CoTaskMemFree(contour->response);
        // Zero out to prevent double-free
        contour->points = nullptr;
        contour->normal_angles = nullptr;
        contour->response = nullptr;
    }
    END_WRAP
}
