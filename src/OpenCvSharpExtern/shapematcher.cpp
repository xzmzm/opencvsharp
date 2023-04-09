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
}
ShapeMatcher::~ShapeMatcher()
{
}

void ShapeMatcher::teach(cv::Mat* pattern)
{
    pattern->copyTo(this->pattern);
    cv::Mat mask(this->pattern.size(), CV_8UC1, { 255 });
    this->detector = cv::makePtr<line2Dup::Detector>(128, std::vector<int>{ 4, 8 });

    // padding to avoid rotating out
    int padding = (int)ceil(0.5 * (sqrt(mask.cols * mask.cols + mask.rows * mask.rows) - MAX(mask.cols, mask.rows)));
    cv::Mat padded_img = cv::Mat(mask.rows + 2 * padding, mask.cols + 2 * padding, mask.type(), cv::Scalar::all(0));
    this->pattern.copyTo(padded_img(cv::Rect(padding, padding, mask.cols, mask.rows)));

    cv::Mat padded_mask = cv::Mat(mask.rows + 2 * padding, mask.cols + 2 * padding, mask.type(), cv::Scalar::all(0));
    mask.copyTo(padded_mask(cv::Rect(padding, padding, mask.cols, mask.rows)));

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
void ShapeMatcher::search(cv::Mat* image, cv::Point2d* retPoint, double* angle)
{
    std::vector<std::string> ids;
    ids.push_back("test");
    int padding = (int)ceil(0.5 * (sqrt(this->pattern.cols * this->pattern.cols + this->pattern.rows * this->pattern.rows) - MAX(this->pattern.cols, this->pattern.rows)));
    cv::Mat padded_img = cv::Mat(image->rows + 2 * padding,
        image->cols + 2 * padding, image->type(), cv::Scalar::all(0));
    image->copyTo(padded_img(cv::Rect(padding, padding, image->cols, image->rows)));

    int stride = 16;
    int n = padded_img.rows / stride;
    int m = padded_img.cols / stride;
    cv::Rect roi(0, 0, stride*m, stride*n);
    cv::Mat img = padded_img(roi).clone();
    assert(img.isContinuous());
    Timer timer;
    auto matches = this->detector->match(img, 90, ids);
    //timer.out();
    //std::cout << "matches.size(): " << matches.size() << std::endl;
    size_t top5 = 5;
    if (top5 > matches.size()) top5 = matches.size();

    // construct scene
    Scene_edge scene;
    // buffer
    std::vector<::Vec2f> pcd_buffer, normal_buffer;
    scene.init_Scene_edge_cpu(img, pcd_buffer, normal_buffer);

    if (img.channels() == 1) cvtColor(img, img, cv::COLOR_GRAY2BGR);

    cv::Mat edge_global;  // get edge
    {
        cv::Mat gray;
        if (img.channels() > 1) {
            cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        }
        else {
            gray = img;
        }

        cv::Mat smoothed = gray;
        cv::Canny(smoothed, edge_global, 30, 60);

        if (edge_global.channels() == 1) cvtColor(edge_global, edge_global, cv::COLOR_GRAY2BGR);
    }

    for (size_t i = top5 - 1; i >= 0; i--)
    {
        cv::Mat edge = edge_global.clone();

        auto match = matches[i];
        auto templ = this->detector->getTemplates("test",
            match.template_id);

        // 270 is width of template image
        // 100 is padding when training
        // tl_x/y: template croping topleft corner when training

        float r_scaled = this->pattern.cols / 2.0f*this->infos_have_templ[match.template_id].scale;

        // scaling won't affect this, because it has been determined by warpAffine
        // cv::warpAffine(src, dst, rot_mat, src.size()); last param
        float train_img_half_width = this->pattern.cols / 2.0f + padding;

        // center x,y of train_img in test img
        float x = match.x - templ[0].tl_x + train_img_half_width;
        float y = match.y - templ[0].tl_y + train_img_half_width;

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
        double init_angle = this->infos_have_templ[match.template_id].angle;
        init_angle = init_angle >= 180 ? (init_angle - 360) : init_angle;

        double ori_diff_angle = std::abs(init_angle);
        double icp_diff_angle = std::abs(-std::atan(result.transformation_[1][0] / result.transformation_[0][0]) / CV_PI * 180 +
            init_angle);
        double improved_angle = ori_diff_angle - icp_diff_angle;
        {
            double x = match.x - templ[0].tl_x + train_img_half_width;
            double y = match.y - templ[0].tl_y + train_img_half_width;
            double new_x = result.transformation_[0][0] * x + result.transformation_[0][1] * y + result.transformation_[0][2];
            double new_y = result.transformation_[1][0] * x + result.transformation_[1][1] * y + result.transformation_[1][2];
            std::cout << "x: " << x << std::endl;
            std::cout << "y: " << y << std::endl;
            std::cout << "new_x: " << new_x << std::endl;
            std::cout << "new_y: " << new_y << std::endl;
            *retPoint = cv::Point2d(new_x, new_y);
        }
        *angle = icp_diff_angle;

        break;

        std::cout << "\n---------------" << std::endl;
        std::cout << "scale: " << std::sqrt(result.transformation_[0][0] * result.transformation_[0][0] +
            result.transformation_[1][0] * result.transformation_[1][0]) << std::endl;
        std::cout << "init diff angle: " << ori_diff_angle << std::endl;
        std::cout << "improved angle: " << improved_angle << std::endl;
        std::cout << "match.template_id: " << match.template_id << std::endl;
        std::cout << "match.similarity: " << match.similarity << std::endl;
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
    auto templates = this->detector->getTemplates("test", 0);
    std::vector<std::vector<line2Dup::Feature>> features(templates.size());
    std::transform(templates.begin(), templates.end(), features.begin(),
        [](const auto& t) { return t.features; });
    return features;
}

