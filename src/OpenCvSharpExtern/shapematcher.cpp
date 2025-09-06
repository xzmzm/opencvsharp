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
    pad_x = 0;
    pad_y = 0;
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

    // The line2Dup::Detector is the core engine for feature extraction and matching.
    // It's initialized with the number of features to extract per template and
    // the T values for spreading at each pyramid level. T controls how gradients
    // are grouped spatially, affecting the robustness of the matching.
    // A smaller T (like 4) is used for higher-resolution pyramid levels,
    // while a larger T (like 8) is used for lower-resolution levels.
    this->detector = cv::makePtr<line2Dup::Detector>(nFeatures, t);

    // padding to avoid rotating out
    int half = (int)ceil(0.5 * sqrt(mask.cols * mask.cols + mask.rows * mask.rows));
    int padc = half - (mask.cols + 1) / 2;
    int padr = half - (mask.rows + 1) / 2;
    cv::Mat padded_img = cv::Mat(mask.rows + 2 * padr, mask.cols + 2 * padc, mask.type(), cv::Scalar::all(0));
    this->pad_x = padc;
    this->pad_y = padr;
    // 1. Why padding is needed:
    // To create rotated versions of the template, we need a canvas large enough
    // to hold the template at any angle without clipping it. The smallest square
    // that can contain the template rotated by any angle has a side length equal to the
    // diagonal of the original template's bounding box. We copy the original pattern into the center of this padded image.
    this->pattern.copyTo(padded_img(cv::Rect(padc, padr, mask.cols, mask.rows)));

    cv::Mat padded_mask = cv::Mat(mask.rows + 2 * padr, mask.cols + 2 * padc, mask.type(), cv::Scalar::all(0));
    mask.copyTo(padded_mask(cv::Rect(padc, padr, mask.cols, mask.rows)));

    this->shapes = cv::makePtr<shape_based_matching::shapeInfo_producer>(padded_img, padded_mask);
    this->shapes->angle_range = {(float)this->minAngle, (float)this->maxAngle};
    this->shapes->angle_step = (float)this->angleStep;
    this->shapes->scale_range = {1.0f}; // { 0.97f, 1.1f };
    this->shapes->scale_step = 100.0f;  // one scale, 0.97

    // 2. How rotated templates are generated:
    // The shapeInfo_producer generates a list of angle/scale combinations based on the specified ranges and steps.
    this->shapes->produce_infos();
    std::string class_id = "test";
    this->infos_have_templ.clear();

    // 3. Pyramid construction and feature extraction:
    // This loop iterates through each generated angle/scale combination.
    for (auto &info : this->shapes->infos)
    {
        // For each angle, `src_of(info)` creates a new image by rotating the padded template.
        int templ_id = this->detector->addTemplate(this->shapes->src_of(info), class_id, this->shapes->mask_of(info));
        // The detector's addTemplate method then builds an image pyramid for this rotated template and extracts features at each level.
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

void ShapeMatcher::search(cv::Mat *image, int refinementLevel, bool useFusion, cv::Point2d *retPoint, double *angle, double *score, int *templateID, cv::RotatedRect *rotatedBounds)
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
    // The two paths use different optimization strategies, which dictate their padding requirements.
    // 1. Fusion Path (useFusion = true): This pipeline processes the image in parallel tiles. It uses a simple,
    //    fixed-size padding (ImagePadding) which is sufficient to ensure kernels (e.g., blur, Sobel)
    //    have enough context when operating on tiles near the image border.
    // 2. Non-Fusion Path (useFusion = false): This pipeline is optimized for global SIMD processing on a single,
    //    continuous memory block. It requires the image's stride (width in memory) to be a multiple of 16
    //    to allow for efficient, branch-free vector processing in its inner loops.
    // Forcing one padding strategy on the other would break its specific performance optimizations.
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
    if (matches.empty())
    {
        *retPoint = cv::Point2d(-1, -1);
        *angle = 0;
        *score = 0;
        *templateID = -1;
        if (rotatedBounds != nullptr)
            *rotatedBounds = cv::RotatedRect();
        return;
    }
    std::cout << "matches.size(): " << matches.size() << std::endl;

    for (size_t i = 0; i < matches.size() && i < 5; ++i)
    {
        auto match = matches[i];
        std::cout << "match.similarity: " << match.template_id << " " << match.similarity << std::endl;
    }

    // Process only the top match, which is at index 0 because matches are sorted descending by score.
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

    switch (refinementLevel)
    {
    case 0: // None
        *retPoint = cv::Point2d(x, y);
        *angle = init_angle;
        *score = match.similarity;
        *templateID = match.template_id;
        break;
    case 1: // Quadratic
    {
        cv::Mat rotated_template;
        cv::Point2f rot_center(this->pattern.cols / 2.0f, this->pattern.rows / 2.0f);
        cv::Mat rot_mat = cv::getRotationMatrix2D(rot_center, -init_angle, 1.0);
        cv::warpAffine(this->pattern, rotated_template, rot_mat, this->pattern.size());

        int search_radius = 4; // search in a (2*radius+1) x (2*radius+1) neighborhood
        cv::Point coarse_match_center_int(cvRound(x), cvRound(y));

        cv::Rect search_roi(
            coarse_match_center_int.x - search_radius - rotated_template.cols / 2,
            coarse_match_center_int.y - search_radius - rotated_template.rows / 2,
            rotated_template.cols + 2 * search_radius,
            rotated_template.rows + 2 * search_radius);

        // Boundary check
        if (search_roi.x < 0 || search_roi.y < 0 ||
            search_roi.x + search_roi.width >= padded_img.cols ||
            search_roi.y + search_roi.height >= padded_img.rows)
        {
            // Fallback to coarse result if ROI is out of bounds
            *retPoint = cv::Point2d(x, y);
            *angle = init_angle;
            *score = match.similarity;
            *templateID = match.template_id;
            break;
        }

        cv::Mat search_window = padded_img(search_roi);
        cv::Mat ncc_result;
        cv::matchTemplate(search_window, rotated_template, ncc_result, cv::TM_CCOEFF_NORMED);

        cv::Point peak_loc;
        double peak_val;
        cv::minMaxLoc(ncc_result, nullptr, &peak_val, nullptr, &peak_loc);

        float refined_peak_x = (float)peak_loc.x;
        float refined_peak_y = (float)peak_loc.y;

        if (peak_loc.x > 0 && peak_loc.x < ncc_result.cols - 1 &&
            peak_loc.y > 0 && peak_loc.y < ncc_result.rows - 1)
        {
            float s_c = ncc_result.at<float>(peak_loc);
            float s_l = ncc_result.at<float>(peak_loc.y, peak_loc.x - 1);
            float s_r = ncc_result.at<float>(peak_loc.y, peak_loc.x + 1);
            float s_t = ncc_result.at<float>(peak_loc.y - 1, peak_loc.x);
            float s_b = ncc_result.at<float>(peak_loc.y + 1, peak_loc.x);

            float den_x = 2 * (s_l + s_r - 2 * s_c);
            if (std::abs(den_x) > 1e-5)
            {
                float dx = (s_l - s_r) / den_x;
                if (std::abs(dx) < 1.0f)
                    refined_peak_x += dx;
            }

            float den_y = 2 * (s_t + s_b - 2 * s_c);
            if (std::abs(den_y) > 1e-5)
            {
                float dy = (s_t - s_b) / den_y;
                if (std::abs(dy) < 1.0f)
                    refined_peak_y += dy;
            }
        }

        double final_x = search_roi.x + refined_peak_x + rotated_template.cols / 2.0;
        double final_y = search_roi.y + refined_peak_y + rotated_template.rows / 2.0;

        *retPoint = cv::Point2d(final_x, final_y);
        *angle = init_angle;
        *score = match.similarity; // Using original score for consistency
        *templateID = match.template_id;
        break;
    }
    case 2: // ICP
    {
        Scene_kdtree scene;
        KDTree_cpu kdtree;
        scene.init_Scene_kdtree_cpu(this->detector->dx_, this->detector->dy_, kdtree);

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

        // subpixel, also refine scale
        cuda_icp::RegistrationResult result = cuda_icp::sim3::ICP2D_Point2Plane_cpu(model_pcd, scene);

        // cuda_icp::RegistrationResult result = cuda_icp::ICP2D_Point2Plane_cpu(model_pcd, scene);

        {
            double center_x = match.x - templ[0].tl_x + train_img_half_width;
            double center_y = match.y - templ[0].tl_y + train_img_half_width;
            double new_x = result.transformation_[0][0] * center_x + result.transformation_[0][1] * center_y + result.transformation_[0][2];
            double new_y = result.transformation_[1][0] * center_x + result.transformation_[1][1] * center_y + result.transformation_[1][2];
            *retPoint = cv::Point2d(new_x, new_y);
        }
        {
            double initial_angle_rad = init_angle * CV_PI / 180.0;
            double refinement_angle_rad = std::atan2(result.transformation_[1][0], result.transformation_[0][0]);
            double final_angle_rad = initial_angle_rad + refinement_angle_rad;
            double final_angle_deg = final_angle_rad * 180.0 / CV_PI;
            while (final_angle_deg > 180.0)
                final_angle_deg -= 360.0;
            while (final_angle_deg <= -180.0)
                final_angle_deg += 360.0;
            *angle = final_angle_deg;
        }

        *score = match.similarity;
        *templateID = match.template_id;
        break;
    }
    } // end switch

    retPoint->x -= ImagePadding;
    retPoint->y -= ImagePadding;

    if (rotatedBounds != nullptr)
    {
        *rotatedBounds = cv::RotatedRect(cv::Point2f((float)retPoint->x, (float)retPoint->y), this->pattern.size(), (float)*angle);
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

CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_new(cv::Mat *pattern, double minAngle, double maxAngle, double angleStep, int nFeatures, int pyramidLevels, ShapeMatcher **returnValue)
{
    BEGIN_WRAP
    auto shapeMatcher = new ShapeMatcher;
    shapeMatcher->setAngleRange(minAngle, maxAngle, angleStep);
    shapeMatcher->teach(pattern, nFeatures, pyramidLevels);
    shapeMatcher->preprocess();
    *returnValue = shapeMatcher;
    END_WRAP
}
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_delete(ShapeMatcher *obj)
{
    BEGIN_WRAP
    delete obj;
    END_WRAP
}
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_teach(ShapeMatcher *obj, cv::Mat *pattern, int nFeatures, int pyramidLevels)
{
    BEGIN_WRAP
    obj->teach(pattern, nFeatures, pyramidLevels);
    END_WRAP
}
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_search(ShapeMatcher *obj, cv::Mat *image, int refinementLevel, cv::Point2d *retPoint, double *angle, double *score, int *templateID, cv::RotatedRect *rotatedBounds)
{
    BEGIN_WRAP
    obj->search(image, refinementLevel, false, retPoint, angle, score, templateID, rotatedBounds);
    END_WRAP
}
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_searchFusion(ShapeMatcher *obj, cv::Mat *image, int refinementLevel, cv::Point2d *retPoint, double *angle, double *score, int *templateID, cv::RotatedRect *rotatedBounds)
{
    BEGIN_WRAP
    obj->search(image, refinementLevel, true, retPoint, angle, score, templateID, rotatedBounds);
    END_WRAP
}
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_getPaddedPattern(ShapeMatcher *obj, double angle, cv::Mat *outPaddedPattern)
{
    BEGIN_WRAP
    obj->getPaddedPattern(angle, outPaddedPattern);
    END_WRAP
}
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_getTemplate(ShapeMatcher *obj, int templateIndex, float *angle, float *scale, line2Dup::Feature *features, int *count)
{
    BEGIN_WRAP
    int n = obj->detector->numTemplates("test");
    if (templateIndex >= 0 && templateIndex < n)
    {
        auto info = obj->infos_have_templ[templateIndex];
        if (angle != nullptr)
            *angle = info.angle;
        if (scale != nullptr)
            *scale = info.scale;

        auto templates = obj->detector->getTemplates("test", templateIndex);
        auto t = templates[0];
        *count = (int)t.features.size();

        if (features != nullptr)
        {
            auto f = t.features;
            for (auto &ff : f)
            {
                ff.x += t.tl_x;
                ff.y += t.tl_y;
            }
            std::copy(f.begin(), f.end(), features);
        }
    }
    else
    {
        *count = 0;
        if (angle != nullptr)
            *angle = 0;
        if (scale != nullptr)
            *scale = 0;
    }
    END_WRAP
}
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_getPatternOffset(ShapeMatcher *obj, cv::Point *offset)
{
    BEGIN_WRAP
    offset->x = obj->pad_x;
    offset->y = obj->pad_y;
    END_WRAP
}
CVAPI(ExceptionStatus)
shapematcher_ShapeMatcher_getFeatures(ShapeMatcher *obj, int templateIndex, line2Dup::Feature *features, int *count)
{
    BEGIN_WRAP
    int n = obj->detector->numTemplates("test");
    if (templateIndex < n)
    {
        auto templates = obj->detector->getTemplates("test", templateIndex);
        auto t = templates[0];
        *count = (int)t.features.size();

        if (features != nullptr)
        {
            auto f = t.features;
            for (auto &ff : f)
            {
                ff.x += t.tl_x;
                ff.y += t.tl_y;
            }
            std::copy(f.begin(), f.end(), features);
        }
    }
    else
    {
        *count = 0;
    }
    END_WRAP
}
