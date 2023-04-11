#include "ShapeMatcher/line2Dup.h"
#include <memory>
#include <iostream>
#include <assert.h>
#include <chrono>
#include <vector>
#include <algorithm>
#include "ShapeMatcher/cuda_icp/icp.h"
#include "shapematcher.h"

class Timer
{
public:
    Timer() : beg_(clock_::now()) {}
    void reset() { beg_ = clock_::now(); }
    double elapsed() const {
        return std::chrono::duration_cast<second_>
            (clock_::now() - beg_).count();
    }
    void out(std::string message = "") {
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }
private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1> > second_;
    std::chrono::time_point<clock_> beg_;
};
ShapeMatcher::ShapeMatcher()
{
    std::cout << "MIPP tests" << std::endl;
    std::cout << "----------" << std::endl << std::endl;

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

    std::cout << "----------" << std::endl << std::endl;
}
ShapeMatcher::~ShapeMatcher()
{
}

void ShapeMatcher::teach(cv::Mat* pattern, int nFeatures = 63, int pyramidLevels = 2)
{
    if (pattern->channels() == 1)
        this->pattern = *pattern;
    //pattern->copyTo();
    else
        cv::cvtColor(*pattern, this->pattern, pattern->channels() == 3 ? cv::COLOR_RGB2GRAY : cv::COLOR_RGBA2GRAY);
    cv::Mat mask(this->pattern.size(), CV_8UC1, { 255 });
    pyramidLevels = MAX(pyramidLevels, 1);
    std::vector<int> t(pyramidLevels);
    t[0] = 4;
    for (int i = 1; i < t.size(); ++i)
        t[i] = 8;
    this->detector = cv::makePtr<line2Dup::Detector>(nFeatures, t);

    // padding to avoid rotating out
    int half = (int)ceil(0.5 * sqrt(mask.cols * mask.cols + mask.rows * mask.rows));
    int padc = half - (mask.cols + 1) / 2;
    int padr = half - (mask.rows + 1) / 2;
    cv::Mat padded_img = cv::Mat(mask.rows + 2 * padr, mask.cols + 2 * padc, mask.type(), cv::Scalar::all(0));
    this->pattern.copyTo(padded_img(cv::Rect(padc, padr, mask.cols, mask.rows)));

    cv::Mat padded_mask = cv::Mat(mask.rows + 2 * padr, mask.cols + 2 * padc, mask.type(), cv::Scalar::all(0));
    mask.copyTo(padded_mask(cv::Rect(padc, padr, mask.cols, mask.rows)));

    this->shapes = cv::makePtr<shape_based_matching::shapeInfo_producer>(padded_img, padded_mask);
    this->shapes->angle_range = { (float)this->minAngle, (float)this->maxAngle };
    this->shapes->angle_step = 1;
    this->shapes->scale_range = { 1.0f };// { 0.97f, 1.1f };
    this->shapes->scale_step = 100.0f;// one scale, 0.97

    this->shapes->produce_infos();
    std::string class_id = "test";
    this->infos_have_templ.clear();
    for (auto& info : this->shapes->infos) {
        int templ_id = this->detector->addTemplate(this->shapes->src_of(info), class_id, this->shapes->mask_of(info));
        if (templ_id != -1) {
            this->infos_have_templ.push_back(info);
        }
    }
}
cv::Mat* ShapeMatcher::getPaddedPattern(double angle)
{ 
    shape_based_matching::shapeInfo_producer::Info info((float)angle, 1.0f);
    auto paddedPattern = new cv::Mat;
    *paddedPattern = this->shapes->src_of(info);
    return paddedPattern;
}

void ShapeMatcher::search(cv::Mat* image, bool refineResults, bool useFusion, cv::Point2d* retPoint, double* angle, double* score, int* templateID)
{
    std::vector<std::string> ids;
    ids.push_back("test");
    int padding = 100;
    cv::Mat img1;
    if (image->channels() == 1)
        img1 = *image;
    else
        cv::cvtColor(*image, img1, image->channels() == 3 ? cv::COLOR_RGB2GRAY : cv::COLOR_RGBA2GRAY);
    //std::cout << "img1.type(): " << img1.type() << " " << img1.channels() << std::endl;
    std::vector<line2Dup::Match> matches;
    Timer timer;
    cv::Mat padded_img;
    if (useFusion)
    {
        padded_img = cv::Mat(image->rows + 2 * padding,
            image->cols + 2 * padding, img1.type(), cv::Scalar::all(0));
        img1.copyTo(padded_img(cv::Rect(padding, padding, image->cols, image->rows)));
        //std::cout << "padded_img.type(): " << padded_img.type() << " " << padded_img.channels() << std::endl;
        matches = this->detector->match_fusion(padded_img, *score, ids);
    }
    else
    {
        int stride = 16;
        int n = (image->rows + 2 * padding) / stride;
        int m = (image->cols + 2 * padding) / stride;
        padded_img = cv::Mat(stride*n, stride*m, img1.type(), cv::Scalar::all(0));
        img1.copyTo(padded_img(cv::Rect(padding, padding, img1.cols, img1.rows)));
        assert(padded_img.isContinuous());
        //std::cout << "padded_img.type(): " << padded_img.type() << " " << padded_img.channels() << std::endl;
        matches =  this->detector->match(padded_img, *score, ids);
    }
    //timer.out();
    std::cout << "matches.size(): " << matches.size() << std::endl;
    size_t top5 = 5;
    if (top5 > matches.size()) top5 = matches.size();

    for (size_t i = 0; i < matches.size(); ++i)
    {
        auto match = matches[i];
        std::cout << "match.similarity: " << match.template_id << " " << match.similarity << std::endl;
    }
    for (size_t i = top5 - 1; i >= 0; i--)
    {
        auto match = matches[i];
        auto templ = this->detector->getTemplates("test", match.template_id);

        // 270 is width of template image
        // 100 is padding when training
        // tl_x/y: template croping topleft corner when training

        float r_scaled = this->pattern.cols / 2.0f*this->infos_have_templ[match.template_id].scale;

        // scaling won't affect this, because it has been determined by warpAffine
        // cv::warpAffine(src, dst, rot_mat, src.size()); last param
        int half = (int)ceil(0.5 * sqrt(this->pattern.cols * this->pattern.cols + this->pattern.rows * this->pattern.rows));
        float train_img_half_width = half; // this->pattern.cols / 2.0f + padding;

        // center x,y of train_img in test img
        float x = match.x - templ[0].tl_x + train_img_half_width;
        float y = match.y - templ[0].tl_y + train_img_half_width;
        double init_angle = this->infos_have_templ[match.template_id].angle;
        init_angle = init_angle >= 180 ? (init_angle - 360) : init_angle;
        if (refineResults)
        {
            // construct scene
            Scene_edge scene;
            // buffer
            std::vector<::Vec2f> pcd_buffer, normal_buffer;
            scene.init_Scene_edge_cpu(padded_img, pcd_buffer, normal_buffer);

            if (padded_img.channels() == 1) cvtColor(padded_img, padded_img, cv::COLOR_GRAY2BGR);

            cv::Mat edge_global;  // get edge
            {
                cv::Mat gray;
                if (padded_img.channels() > 1) {
                    cv::cvtColor(padded_img, gray, cv::COLOR_BGR2GRAY);
                }
                else {
                    gray = padded_img;
                }

                cv::Mat smoothed = gray;
                cv::Canny(smoothed, edge_global, 30, 60);

                if (edge_global.channels() == 1) cvtColor(edge_global, edge_global, cv::COLOR_GRAY2BGR);
            }
            cv::Mat edge = edge_global.clone();
            std::vector<::Vec2f> model_pcd(templ[0].features.size());
            for (int i = 0; i < templ[0].features.size(); i++) {
                auto& feat = templ[0].features[i];
                model_pcd[i] = {
                    float(feat.x + match.x),
                    float(feat.y + match.y)
                };
            }
            cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);

            cv::Vec3b randColor;
            randColor[0] = 0;
            randColor[1] = 0;
            randColor[2] = 255;
            for (int i = 0; i < templ[0].features.size(); i++) {
                auto feat = templ[0].features[i];
                //cv::circle(edge, { feat.x + match.x, feat.y + match.y }, 2, randColor, -1);
            }

            randColor[0] = 0;
            randColor[1] = 255;
            randColor[2] = 0;
            for (int i = 0; i < templ[0].features.size(); i++) {
                auto feat = templ[0].features[i];
                double x = feat.x + match.x;
                double y = feat.y + match.y;
                double new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
                double new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];

                //cv::circle(edge, { int(new_x + 0.5f), int(new_y + 0.5f) }, 2, randColor, -1);
            }

            double ori_diff_angle = std::abs(init_angle);
            double icp_diff_angle = std::abs(-std::atan(result.transformation_[1][0] / result.transformation_[0][0]) / CV_PI * 180 +
                init_angle);
            double improved_angle = ori_diff_angle - icp_diff_angle;
            {
                double x = match.x - templ[0].tl_x + train_img_half_width;
                double y = match.y - templ[0].tl_y + train_img_half_width;
                double new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
                double new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
                //std::cout << "x: " << x << std::endl;
                //std::cout << "y: " << y << std::endl;
                //std::cout << "new_x: " << new_x << std::endl;
                //std::cout << "new_y: " << new_y << std::endl;
                *retPoint = cv::Point2d(new_x, new_y);
            }
            *angle = icp_diff_angle;
            *score = match.similarity;
            *templateID = match.template_id;
            //std::cout << "\n---------------" << std::endl;
            //std::cout << "scale: " << std::sqrt(result.transformation_[0][0] * result.transformation_[0][0] +
            //    result.transformation_[1][0] * result.transformation_[1][0]) << std::endl;
            //std::cout << "init diff angle: " << ori_diff_angle << std::endl;
            //std::cout << "improved angle: " << improved_angle << std::endl;
            //std::cout << "match.template_id: " << match.template_id << std::endl;
            //std::cout << "match.similarity: " << match.similarity << std::endl;
        }
        else
        {
            *retPoint = cv::Point2d(x, y);
            *angle = init_angle;
            *score = match.similarity;
            *templateID = match.template_id;
        }

        break;
    }

    //std::cout << "test end" << std::endl << std::endl;
}
void ShapeMatcher::preprocess()
{
}
void ShapeMatcher::setAngleRange(double minAngle, double maxAngle, double angleStep)
{
    this->minAngle = minAngle;
    this->maxAngle = maxAngle;
    this->angleStep = angleStep;
}

std::vector<std::vector<line2Dup::Feature>> ShapeMatcher::getFeatures()
{
    int n = this->detector->numTemplates("test");
    std::cout << "n:" << n << std::endl;
    std::vector<std::vector<line2Dup::Feature>> features(n);
    for (int i = 0; i < n; ++i)
    {
        auto templates = this->detector->getTemplates("test", i);
        auto t = templates[0];
        features[i] = t.features;
        for (auto& ff : features[i])
        {
            ff.x += t.tl_x;
            ff.y += t.tl_y;
        }
    }
    return features;
}

