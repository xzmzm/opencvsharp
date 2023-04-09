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
    ~ShapeMatcher();
    void teach(cv::Mat* pattern);
    void search(cv::Mat* image, cv::Point2d* retPoint, double* angle);
    void preprocess();
    void setAngleRange(double minAngle, double maxAngle, double angleStep);

    std::vector<std::vector<line2Dup::Feature>> getFeatures();

    double minAngle;
    double maxAngle;
    double angleStep;
    cv::Mat pattern;
    cv::Ptr<line2Dup::Detector> detector;
    cv::Ptr<shape_based_matching::shapeInfo_producer> shapes;
    std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;

};

CVAPI(ExceptionStatus) shapematcher_ShapeMatcher_new(cv::Mat* pattern, double minAngle, double maxAngle, double angleStep, double acceptancePercentage, ShapeMatcher **returnValue)
{
    BEGIN_WRAP
    auto shapeMatcher = new ShapeMatcher;
    shapeMatcher->setAngleRange(minAngle, maxAngle, angleStep);
    shapeMatcher->teach(pattern);
    shapeMatcher->preprocess();
    *returnValue = shapeMatcher;
    END_WRAP
}
CVAPI(ExceptionStatus) shapematcher_ShapeMatcher_delete(ShapeMatcher* obj)
{
    BEGIN_WRAP
    delete obj;
    END_WRAP
}
CVAPI(ExceptionStatus) shapematcher_ShapeMatcher_teach(ShapeMatcher* obj, cv::Mat* pattern)
{
    BEGIN_WRAP
    obj->teach(pattern);
    END_WRAP
}
CVAPI(ExceptionStatus) shapematcher_ShapeMatcher_search(ShapeMatcher* obj, cv::Mat* image, cv::Point2d* retPoint, double* angle)
{
    BEGIN_WRAP
    obj->search(image, retPoint, angle);
    END_WRAP
}
CVAPI(ExceptionStatus) shapematcher_ShapeMatcher_getFeaturesCount(ShapeMatcher* obj, int templateIndex, int* featuresCount)
{
    BEGIN_WRAP
    auto features = obj->getFeatures();
    if (templateIndex < features.size())
        *featuresCount = (int)features[templateIndex].size();
    else
        *featuresCount = -1;
    END_WRAP
}
CVAPI(ExceptionStatus) shapematcher_ShapeMatcher_getFeatures(ShapeMatcher* obj, int templateIndex, line2Dup::Feature* features)
{
    BEGIN_WRAP
    auto f1 = obj->getFeatures();
    if (templateIndex < f1.size())
    {
        auto f = f1[templateIndex];
        std::copy(f.begin(), f.end(), features);
    }
    END_WRAP
}

#endif
