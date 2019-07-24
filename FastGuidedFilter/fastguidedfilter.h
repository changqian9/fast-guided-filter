#ifndef GUIDED_FILTER_H
#define GUIDED_FILTER_H

#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#ifdef __cplusplus
#include <opencv2/opencv.hpp>
class FastGuidedFilterImpl;

class FastGuidedFilter
{
public:
    FastGuidedFilter(const cv::Mat &I, int r, double eps,int s);
    ~FastGuidedFilter();

    int filter(cv::Mat &dst, const cv::Mat &p, int depth = -1) const;

private:
    FastGuidedFilterImpl *impl_;
};

cv::Mat fastGuidedFilter(const cv::Mat &I, int r, double eps, int s = 1);
#endif

#ifdef __cplusplus
extern "C" {
#endif
int fastguidedfilter_c(IplImage *dst, IplImage *src, int r, int s, double eps);
#ifdef __cplusplus
}
#endif

#endif
