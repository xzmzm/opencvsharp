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
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
    }
    void out(std::string message = "")
    {
        double t = elapsed();
        std::cout << message << "\nelasped time:" << t << "s" << std::endl;
        reset();
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};
ShapeMatcher::ShapeMatcher()
{
    printMIPPInfo();
}
ShapeMatcher::~ShapeMatcher()
{
}

void ShapeMatcher::teach(cv::Mat *pattern, int nFeatures = 63, int pyramidLevels = 2)
{
    if (pattern->channels() == 1)
        this->pattern = *pattern;
    // pattern->copyTo();
    else
        cv::cvtColor(*pattern, this->pattern, pattern->channels() == 3 ? cv::COLOR_RGB2GRAY : cv::COLOR_RGBA2GRAY);
    cv::Mat mask(this->pattern.size(), CV_8UC1, {255});
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
    this->shapes->angle_range = {(float)this->minAngle, (float)this->maxAngle};
    this->shapes->angle_step = 1;
    this->shapes->scale_range = {1.0f}; // { 0.97f, 1.1f };
    this->shapes->scale_step = 100.0f;  // one scale, 0.97

    this->shapes->produce_infos();
    std::string class_id = "test";
    this->infos_have_templ.clear();
    for (auto &info : this->shapes->infos)
    {
        int templ_id = this->detector->addTemplate(this->shapes->src_of(info), class_id, this->shapes->mask_of(info));
        if (templ_id != -1)
        {
            this->infos_have_templ.push_back(info);
        }
    }
}
void ShapeMatcher::getPaddedPattern(double angle, cv::Mat *outPaddedPattern)
{
    shape_based_matching::shapeInfo_producer::Info info((float)angle, 1.0f);
    *outPaddedPattern = this->shapes->src_of(info);
}

void ShapeMatcher::search(cv::Mat *image, bool refineResults, bool useFusion, cv::Point2d *retPoint, double *angle, double *score, int *templateID)
{
    const int ImagePadding = 100; // A fixed padding value used in search.
    std::vector<std::string> ids;
    ids.push_back("test");
    cv::Mat img1;
    if (image->channels() == 1)
        img1 = *image;
    else
        cv::cvtColor(*image, img1, image->channels() == 3 ? cv::COLOR_RGB2GRAY : cv::COLOR_RGBA2GRAY);

    std::vector<line2Dup::Match> matches;
    Timer timer;
    cv::Mat padded_img;

    // NOTE: The padding logic is different for fusion vs. non-fusion paths.
    // Fusion uses a fixed padding of 100px.
    // Non-fusion pads the image to be a multiple of stride (16).
    // This could lead to slightly different results between the two modes.
    if (useFusion)
    {
        padded_img = cv::Mat(image->rows + 2 * ImagePadding,
                             image->cols + 2 * ImagePadding, img1.type(), cv::Scalar::all(0));
        img1.copyTo(padded_img(cv::Rect(ImagePadding, ImagePadding, image->cols, image->rows)));
        matches = this->detector->match_fusion(padded_img, *score, ids);
    }
    else
    {
        int stride = 16;
        int n = (image->rows + 2 * ImagePadding) / stride;
        int m = (image->cols + 2 * ImagePadding) / stride;
        padded_img = cv::Mat(stride * n, stride * m, img1.type(), cv::Scalar::all(0));
        img1.copyTo(padded_img(cv::Rect(ImagePadding, ImagePadding, img1.cols, img1.rows)));
        assert(padded_img.isContinuous());
        matches = this->detector->match(padded_img, *score, ids);
    }
    // timer.out();
    std::cout << "matches.size(): " << matches.size() << std::endl;

    for (size_t i = 0; i < matches.size() && i < 5; ++i)
    {
        auto match = matches[i];
        std::cout << "match.similarity: " << match.template_id << " " << match.similarity << std::endl;
    }

    // Process only the top match, which is at index 0 because matches are sorted descending by score.
    if (!matches.empty())
    {
        auto match = matches[0];
        auto templ = this->detector->getTemplates("test", match.template_id);

        // NOTE: The coordinate system calculations below are sensitive to the padding and cropping
        // logic used during the 'teach' phase in line2Dup::Detector::addTemplate.
        // 'train_img_half_width' is calculated to find the center of the original, padded template image.
        // The final coordinates (x, y) represent the center of the template match in the original, unpadded 'image' coordinate system.

        // The padded image during teaching has a size determined by the diagonal of the original pattern
        // to avoid cropping during rotation. 'half' is half of that diagonal.
        int half = (int)ceil(0.5 * sqrt(this->pattern.cols * this->pattern.cols + this->pattern.rows * this->pattern.rows));
        float train_img_half_width = (float)half;

        // Calculate the center of the match in the padded search image's coordinate system.
        // match.x, match.y: Top-left of the matched template region in the padded search image.
        // templ[0].tl_x, tl_y: Top-left corner of the template feature bounding box relative to the padded template image.
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

            if (padded_img.channels() == 1)
                cvtColor(padded_img, padded_img, cv::COLOR_GRAY2BGR);

            std::vector<::Vec2f> model_pcd(templ[0].features.size());
            for (int i = 0; i < templ[0].features.size(); i++)
            {
                auto &feat = templ[0].features[i];
                model_pcd[i] = {
                    float(feat.x + match.x),
                    float(feat.y + match.y)};
            }
            cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);

            // Refine position
            {
                double center_x = match.x - templ[0].tl_x + train_img_half_width;
                double center_y = match.y - templ[0].tl_y + train_img_half_width;
                double new_x = result.transformation_[0][0] * center_x + result.transformation_[0][1] * center_y + result.transformation_[0][2];
                double new_y = result.transformation_[1][0] * center_x + result.transformation_[1][1] * center_y + result.transformation_[1][2];
                *retPoint = cv::Point2d(new_x, new_y);
            }

            // Refine angle
            {
                double initial_angle_rad = init_angle * CV_PI / 180.0;
                // The rotation part of the 2D affine transformation matrix is:
                // [ cos(t), -sin(t) ]
                // [ sin(t),  cos(t) ]
                // We extract the refinement angle 't' from this. Note the negative sign for atan2.
                double refinement_angle_rad = -std::atan2(result.transformation_[1][0], result.transformation_[0][0]);

                double final_angle_rad = initial_angle_rad + refinement_angle_rad;

                // Convert back to degrees and normalize to [-180, 180]
                double final_angle_deg = final_angle_rad * 180.0 / CV_PI;
                while (final_angle_deg > 180.0)
                    final_angle_deg -= 360.0;
                while (final_angle_deg <= -180.0)
                    final_angle_deg += 360.0;

                *angle = final_angle_deg; // Return the final computed angle
            }

            *score = match.similarity;
            *templateID = match.template_id;
        }
        else
        {
            *retPoint = cv::Point2d(x, y);
            *angle = init_angle;
            *score = match.similarity;
            *templateID = match.template_id;
        }
    }
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
        for (auto &ff : features[i])
        {
            ff.x += t.tl_x;
            ff.y += t.tl_y;
        }
    }
    return features;
}