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
    float* direction;
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
                c_contour.direction = (float*)CoTaskMemAlloc(data_size);
                memcpy(c_contour.direction, cpp_contour.direction.data(), data_size);
                
                c_contour.response = (float*)CoTaskMemAlloc(data_size);
                memcpy(c_contour.response, cpp_contour.response.data(), data_size);
            }
            else
            {
                c_contour.points = nullptr;
                c_contour.direction = nullptr;
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
            if(contours[i].direction) CoTaskMemFree(contours[i].direction);
            if(contours[i].response) CoTaskMemFree(contours[i].response);
        }
        CoTaskMemFree(contours);
    }
    END_WRAP
}
