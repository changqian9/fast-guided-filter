#include "fastguidedfilter.h"
//#include <thread>
#include <iostream>
static cv::Mat boxfilter(const cv::Mat &I, int r)
{
    cv::Mat result;
    cv::blur(I, result, cv::Size(r, r));
    return result;
}

static cv::Mat convertTo(const cv::Mat &mat, int depth)
{
    if (mat.depth() == depth)
        return mat;

    cv::Mat result;
    mat.convertTo(result, depth);
    return result;
}

class Parallel_process : public cv::ParallelLoopBody
{

    public:
        std::vector<cv::Mat> &pc;
        int r;
        std::vector<cv::Mat> Ichannels;
        std::vector<cv::Mat> origIchannels;

        cv::Mat mean_I_r;
        cv::Mat mean_I_g;
        cv::Mat mean_I_b;

        cv::Mat invrr;
        cv::Mat invrg;
        cv::Mat invrb;
        cv::Mat invgg;
        cv::Mat invgb;
        cv::Mat invbb;

        Parallel_process(std::vector<cv::Mat> &channels):pc(channels) { }

        virtual void operator()(const cv::Range& range) const
        {
            for(int i = range.start; i < range.end; i++)
            {
                cv::Mat p = pc[i];
                /* divide image in 'diff' number
                   of parts and process simultaneously */
                cv::Mat mean_p = boxfilter(p, r);

                cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p), r);
                cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p), r);
                cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p), r);

                // covariance of (I, p) in each local patch.
                cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
                cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
                cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

                cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
                cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
                cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

                cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b);

                cv::Mat mean_a_r = boxfilter(a_r, r);
                cv::Mat mean_a_g = boxfilter(a_g, r);
                cv::Mat mean_a_b = boxfilter(a_b, r);
                cv::Mat mean_b = boxfilter(b, r);
                cv::resize(mean_a_r ,mean_a_r,cv::Size(origIchannels[0].cols,origIchannels[0].rows),0,0,cv::INTER_LINEAR);
                cv::resize(mean_a_g ,mean_a_g,cv::Size(origIchannels[1].cols,origIchannels[1].rows),0,0,cv::INTER_LINEAR);
                cv::resize(mean_a_b ,mean_a_b,cv::Size(origIchannels[2].cols,origIchannels[2].rows),0,0,cv::INTER_LINEAR);
                cv::resize(mean_b,mean_b,cv::Size(origIchannels[2].cols,origIchannels[2].rows),0,0,cv::INTER_LINEAR);
                pc[i] = mean_a_r.mul(origIchannels[0]) +mean_a_g.mul(origIchannels[1]) +mean_a_b.mul(origIchannels[2]) + mean_b;
            }
        }
};

class FastGuidedFilterImpl
{
public:
    FastGuidedFilterImpl(int r, double eps,int s):r(r),eps(eps),s(s){}
    virtual ~FastGuidedFilterImpl() {}

    int filter(cv::Mat &dst, const cv::Mat &p, int depth);

protected:
    int Idepth,r,s;
    double eps;

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const = 0;
};

class FastGuidedFilterMono : public FastGuidedFilterImpl
{
public:
    FastGuidedFilterMono(const cv::Mat &I, int r, double eps,int s);

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;

private:

    cv::Mat I,origI, mean_I, var_I;
};

class FastGuidedFilterColor : public FastGuidedFilterImpl
{
public:
    FastGuidedFilterColor(const cv::Mat &I, int r, double eps,int s);
    std::vector<cv::Mat> origIchannels,Ichannels;
    cv::Mat mean_I_r, mean_I_g, mean_I_b;
    cv::Mat invrr, invrg, invrb, invgg, invgb, invbb;

private:
    virtual cv::Mat filterSingleChannel(const cv::Mat &p) const;
};


int FastGuidedFilterImpl::filter(cv::Mat &dst, const cv::Mat &p, int depth)
{
    cv::Mat p2;
    cv::resize(p, p2, cv::Size(p.cols/s,p.rows/s),0,0,cv::INTER_NEAREST);
    p2 = convertTo(p2, Idepth);
    cv::Mat result;
    if (p.channels() == 1)
    {
        result = filterSingleChannel(p2);
    }
    else
    {
        std::vector<cv::Mat> pc;
        cv::split(p2, pc);

        FastGuidedFilterColor *filter_instance = (FastGuidedFilterColor*) this;
        Parallel_process process(pc);
        process.r = filter_instance->r;
        process.Ichannels = filter_instance->Ichannels;
        process.origIchannels = filter_instance->origIchannels;

        process.mean_I_r = filter_instance->mean_I_r;
        process.mean_I_g = filter_instance->mean_I_g;
        process.mean_I_b = filter_instance->mean_I_b;

        process.invrr = filter_instance->invrr;
        process.invrg = filter_instance->invrg;
        process.invrb = filter_instance->invrb;
        process.invgg = filter_instance->invgg;
        process.invgb = filter_instance->invgb;
        process.invbb = filter_instance->invbb;

        cv::parallel_for_(cv::Range(0, 3), process);
        //for (std::size_t i = 0; i < pc.size(); ++i)
        //    pc[i] = filterSingleChannel(pc[i]);
        /*
        std::vector<std::thread> workers;
        for (std::size_t i = 0; i < pc.size(); ++i)
            workers.push(std::thread(&filterSingleColorChannel,std::ref(pc[i]), g_r, g_Ichannels, g_origIchannels, g_mean_I_r, g_mean_I_g, g_mean_I_b, g_invrr, g_invrg, g_invrb, g_invgg, g_invgb, g_invbb));
        std::for_each(workers.begin(), workers.end(), [](std::thread &t) { t.join(); });
        */

        cv::merge(pc, result);
    }

    depth = depth == -1 ? p.depth() : depth;

    if (result.depth() == depth) {
        dst = result;
    } else {
        result.convertTo(dst, depth);
    }
    return 0;
}

FastGuidedFilterMono::FastGuidedFilterMono(const cv::Mat &origI, int r, double eps,int s):FastGuidedFilterImpl(r,eps,s)
{

    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        this->origI = origI.clone();
    else
        origI.convertTo(this->origI, CV_32F);
    cv::resize(this->origI ,I,cv::Size(this->origI.cols/s,this->origI.rows/s),0,0,cv::INTER_NEAREST);
    Idepth = I.depth();

    mean_I = boxfilter(I, r);
    cv::Mat mean_II = boxfilter(I.mul(I), r);
    var_I = mean_II - mean_I.mul(mean_I);
}

cv::Mat FastGuidedFilterMono::filterSingleChannel(const cv::Mat &p) const
{

    cv::Mat mean_p = boxfilter(p, r);
    cv::Mat mean_Ip = boxfilter(I.mul(p), r);
    cv::Mat cov_Ip = mean_Ip - mean_I.mul(mean_p); // this is the covariance of (I, p) in each local patch.

    cv::Mat a = cov_Ip / (var_I + eps);
    cv::Mat b = mean_p - a.mul(mean_I);

    cv::Mat mean_a = boxfilter(a, r);
    cv::Mat mean_b = boxfilter(b, r);
    cv::resize(mean_a ,mean_a,cv::Size(origI.cols,origI.rows),0,0,cv::INTER_LINEAR);
    cv::resize(mean_b ,mean_b,cv::Size(origI.cols,origI.rows),0,0,cv::INTER_LINEAR);
    return mean_a.mul(origI) + mean_b;
}

FastGuidedFilterColor::FastGuidedFilterColor(const cv::Mat &origI, int r, double eps, int s):FastGuidedFilterImpl(r,eps,s)// : r(r), eps(eps)
{

    cv::Mat I;
    if (origI.depth() == CV_32F || origI.depth() == CV_64F)
        I = origI.clone();
    else
        I = convertTo(origI, CV_32F);
    Idepth = I.depth();

    cv::split(I, origIchannels);
    cv::resize(I,I,cv::Size(I.cols/s,I.rows/s),0,0,cv::INTER_NEAREST);
    cv::split(I, Ichannels);

    mean_I_r = boxfilter(Ichannels[0], r);
    mean_I_g = boxfilter(Ichannels[1], r);
    mean_I_b = boxfilter(Ichannels[2], r);

    // variance of I in each local patch: the matrix Sigma.
    // Note the variance in each local patch is a 3x3 symmetric matrix:
    //           rr, rg, rb
    //   Sigma = rg, gg, gb
    //           rb, gb, bb
    cv::Mat var_I_rr = boxfilter(Ichannels[0].mul(Ichannels[0]), r) - mean_I_r.mul(mean_I_r) + eps;
    cv::Mat var_I_rg = boxfilter(Ichannels[0].mul(Ichannels[1]), r) - mean_I_r.mul(mean_I_g);
    cv::Mat var_I_rb = boxfilter(Ichannels[0].mul(Ichannels[2]), r) - mean_I_r.mul(mean_I_b);
    cv::Mat var_I_gg = boxfilter(Ichannels[1].mul(Ichannels[1]), r) - mean_I_g.mul(mean_I_g) + eps;
    cv::Mat var_I_gb = boxfilter(Ichannels[1].mul(Ichannels[2]), r) - mean_I_g.mul(mean_I_b);
    cv::Mat var_I_bb = boxfilter(Ichannels[2].mul(Ichannels[2]), r) - mean_I_b.mul(mean_I_b) + eps;
    // Inverse of Sigma + eps * I
    invrr = var_I_gg.mul(var_I_bb) - var_I_gb.mul(var_I_gb);
    invrg = var_I_gb.mul(var_I_rb) - var_I_rg.mul(var_I_bb);
    invrb = var_I_rg.mul(var_I_gb) - var_I_gg.mul(var_I_rb);
    invgg = var_I_rr.mul(var_I_bb) - var_I_rb.mul(var_I_rb);
    invgb = var_I_rb.mul(var_I_rg) - var_I_rr.mul(var_I_gb);
    invbb = var_I_rr.mul(var_I_gg) - var_I_rg.mul(var_I_rg);

    cv::Mat covDet = invrr.mul(var_I_rr) + invrg.mul(var_I_rg) + invrb.mul(var_I_rb);

    invrr /= covDet;
    invrg /= covDet;
    invrb /= covDet;
    invgg /= covDet;
    invgb /= covDet;
    invbb /= covDet;
}

cv::Mat FastGuidedFilterColor::filterSingleChannel(const cv::Mat &p) const
{
    cv::Mat mean_p = boxfilter(p, r);

    cv::Mat mean_Ip_r = boxfilter(Ichannels[0].mul(p), r);
    cv::Mat mean_Ip_g = boxfilter(Ichannels[1].mul(p), r);
    cv::Mat mean_Ip_b = boxfilter(Ichannels[2].mul(p), r);

    // covariance of (I, p) in each local patch.
    cv::Mat cov_Ip_r = mean_Ip_r - mean_I_r.mul(mean_p);
    cv::Mat cov_Ip_g = mean_Ip_g - mean_I_g.mul(mean_p);
    cv::Mat cov_Ip_b = mean_Ip_b - mean_I_b.mul(mean_p);

    cv::Mat a_r = invrr.mul(cov_Ip_r) + invrg.mul(cov_Ip_g) + invrb.mul(cov_Ip_b);
    cv::Mat a_g = invrg.mul(cov_Ip_r) + invgg.mul(cov_Ip_g) + invgb.mul(cov_Ip_b);
    cv::Mat a_b = invrb.mul(cov_Ip_r) + invgb.mul(cov_Ip_g) + invbb.mul(cov_Ip_b);

    cv::Mat b = mean_p - a_r.mul(mean_I_r) - a_g.mul(mean_I_g) - a_b.mul(mean_I_b);

    cv::Mat mean_a_r = boxfilter(a_r, r);
    cv::Mat mean_a_g = boxfilter(a_g, r);
    cv::Mat mean_a_b = boxfilter(a_b, r);
    cv::Mat mean_b = boxfilter(b, r);
    cv::resize(mean_a_r ,mean_a_r,cv::Size(origIchannels[0].cols,origIchannels[0].rows),0,0,cv::INTER_LINEAR);
    cv::resize(mean_a_g ,mean_a_g,cv::Size(origIchannels[1].cols,origIchannels[1].rows),0,0,cv::INTER_LINEAR);
    cv::resize(mean_a_b ,mean_a_b,cv::Size(origIchannels[2].cols,origIchannels[2].rows),0,0,cv::INTER_LINEAR);
    cv::resize(mean_b,mean_b,cv::Size(origIchannels[2].cols,origIchannels[2].rows),0,0,cv::INTER_LINEAR);
    return (mean_a_r.mul(origIchannels[0]) +mean_a_g.mul(origIchannels[1]) +mean_a_b.mul(origIchannels[2]) + mean_b);
}

FastGuidedFilter::FastGuidedFilter(const cv::Mat &I, int r, double eps,int s)
{
    CV_Assert(I.channels() == 1 || I.channels() == 3);

    if (I.channels() == 1)
        impl_ = new FastGuidedFilterMono(I, 2 * (r/s) + 1, eps,s);
    else
        impl_ = new FastGuidedFilterColor(I, 2 * (r/s) + 1, eps,s);
}

FastGuidedFilter::~FastGuidedFilter()
{
    delete impl_;
}

int FastGuidedFilter::filter(cv::Mat &dst, const cv::Mat &p, int depth) const
{
    return impl_->filter(dst, p, depth);
}

cv::Mat fastGuidedFilter(const cv::Mat &I, int r, double eps, int s)
{
    cv::Mat result;
    const cv::Mat &p = I;
    FastGuidedFilter(I, r, eps,s).filter(result, p, -1);
    return result;
}

int fastguidedfilter_c(IplImage *inimg, IplImage *outimg, int r, int s, double eps) {
    cv::Mat result = cv::cvarrToMat(outimg);
    cv::Mat I = cv::cvarrToMat(inimg);
    FastGuidedFilter(I, r, eps,s).filter(result, I, -1);
    std::cout << "fast guided filter: " << r << " " << eps << " " << s << std::endl;
    return 0;
}
