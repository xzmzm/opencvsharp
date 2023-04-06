#pragma once

// ReSharper disable IdentifierTypo
// ReSharper disable CppInconsistentNaming
// ReSharper disable CppNonInlineFunctionDefinitionInHeaderFile

#include "include_opencv.h"

#ifndef _WINRT_DLL

#pragma region ShapeMatcher

class ShapeMatcher
{
public:
    ShapeMatcher();
    void teach(cv::Mat* pattern);
    void search(cv::Mat* image, cv::Point2d* retPoint, double* angle);
    void preprocess();
    void setAngleRange(double minAngle, double maxAngle, double angleStep);
};

CVAPI(ExceptionStatus) shapematcher_ShapeMatcher_new(ShapeMatcher **returnValue)
{
    BEGIN_WRAP
    *returnValue = new ShapeMatcher;
    END_WRAP
}
CVAPI(ExceptionStatus)shapematcher_ShapeMatcher_delete(ShapeMatcher *obj)
{
    BEGIN_WRAP
    delete obj;
    END_WRAP
}

#endif
