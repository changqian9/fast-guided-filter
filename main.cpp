#include "FastGuidedFilter/fastguidedfilter.h"
using namespace std;
using namespace cv;
int main() {

    int R[]={2,4,8};
    double EPS[]={0.1,0.2,0.4};
    Mat result;
    for(int s=1;s<=2;++s) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                //Mat P = imread("../imgs/cat.jpg", CV_LOAD_IMAGE_GRAYSCALE);
                //Mat P = imread("../imgs/people.png", CV_LOAD_IMAGE_ANYCOLOR);
                Mat P = imread("../imgs/people.png", CV_LOAD_IMAGE_ANYCOLOR);
                Mat I;
                //cvtColor(P,I,CV_BGR2GRAY);
                I = P;
                int r = R[i];
                float eps = EPS[j]*EPS[j];
                eps *= 255 * 255;
                clock_t start_time1 = clock();
                result = fastGuidedFilter(I, P, r, eps, s);
                clock_t end_time1 = clock();
                cout << "fastguidedfilter Running time is: "
                     << static_cast<double>(end_time1 - start_time1) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
                string name = "I:color_result_s:" + to_string(s) + "_r:" + to_string(r) + "_eps:" + to_string(EPS[j]) + "^2.png";
                imwrite(name, result);
            }
        }
    }

}
