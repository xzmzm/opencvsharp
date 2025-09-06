#pragma once

// ReSharper disable IdentifierTypo
// ReSharper disable CppInconsistentNaming
// ReSharper disable CppNonInlineFunctionDefinitionInHeaderFile

#include "include_opencv.h"

#ifndef _WINRT_DLL

inline void printMIPPInfo()
{
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl
              << std::endl;

    std::cout << "Instr. type:       " << mipp::InstructionType << std::endl;
    std::cout << "Instr. full type:  " << mipp::InstructionFullType << std::endl;
    std::cout << "Instr. version:    " << mipp::InstructionVersion << std::endl;
    std::cout << "Instr. size:       " << mipp::RegisterSizeBit << " bits" << std::endl;
    std::cout << "Instr. lanes:      " << mipp::Lanes << std::endl;
    std::cout << "64-bit support:    " << (mipp::Support64Bit ? "yes" : "no") << std::endl;
    std::cout << "Byte/word support: " << (mipp::SupportByteWord ? "yes" : "no") << std::endl;

#ifndef has_max_int8_t
    std::cout << "in this SIMD, int8 max is not inplemented by MIPP" << std::endl;
#endif

#ifndef has_shuff_int8_t
    std::cout << "in this SIMD, int8 shuff is not inplemented by MIPP" << std::endl;
#endif

    std::cout << "----------" << std::endl
              << std::endl;
}

#pragma region ShapeMatcher

class ShapeMatcher
{
public:
    ShapeMatcher();
    ~ShapeMatcher();
    void teach(cv::Mat *pattern, int nFeatures, int pyramidLevels);
    void search(cv::Mat *image, int refinementLevel, bool useFusion, cv::Point2d *retPoint, double *angle, double *score, int *templateID, cv::RotatedRect *rotatedBounds);
    void preprocess();
    void setAngleRange(double minAngle, double maxAngle, double angleStep);
    void getPaddedPattern(double angle, cv::Mat *outPaddedPattern);

    double minAngle;
    double maxAngle;
    double angleStep;
    cv::Mat pattern;
    cv::Ptr<line2Dup::Detector> detector;
    cv::Ptr<shape_based_matching::shapeInfo_producer> shapes;
    std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
    int pad_x;
    int pad_y;
};

CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_new(cv::Mat *pattern, double minAngle, double maxAngle, double angleStep, int nFeatures, int pyramidLevels, ShapeMatcher **returnValue);
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_delete(ShapeMatcher *obj);
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_teach(ShapeMatcher *obj, cv::Mat *pattern, int nFeatures, int pyramidLevels);
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_search(ShapeMatcher *obj, cv::Mat *image, int refinementLevel, cv::Point2d *retPoint, double *angle, double *score, int *templateID, cv::RotatedRect *rotatedBounds);
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_searchFusion(ShapeMatcher *obj, cv::Mat *image, int refinementLevel, cv::Point2d *retPoint, double *angle, double *score, int *templateID, cv::RotatedRect *rotatedBounds);
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_getPaddedPattern(ShapeMatcher *obj, double angle, cv::Mat *outPaddedPattern);
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_getTemplate(ShapeMatcher *obj, int templateIndex, float *angle, float *scale, line2Dup::Feature *features, int *count);
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_getPatternOffset(ShapeMatcher *obj, cv::Point *offset);
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_getFeatures(ShapeMatcher *obj, int templateIndex, line2Dup::Feature *features, int *count);
#pragma endregion
#endif
