#include "FastGuidedFilter/fastguidedfilter.h"
using namespace std;
using namespace cv;
int main() {

    //Mat img = imread("../imgs/people.jpg", CV_LOAD_IMAGE_ANYCOLOR);
    int R[]={2,4,8};
    double EPS[]={0.1,0.2,0.4};
    for(int s=1;s<=2;++s) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                Mat img = imread("../imgs/cat.png", CV_LOAD_IMAGE_GRAYSCALE);
                int r = R[i];
                float eps = EPS[j]*EPS[j];
                eps *= 255 * 255;
                clock_t start_time1 = clock();
                img = fastGuidedFilter(img, img, r, eps, s);
                clock_t end_time1 = clock();
                cout << "fastguidedfilter Running time is: "
                     << static_cast<double>(end_time1 - start_time1) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
                string name = "result_s:" + to_string(s) + "_r:" + to_string(r) + "_eps:" + to_string(EPS[j]) + "^2.png";
                imwrite(name, img);
            }
        }
    }
/*    int r = 2;// 2 4 8
    double eps = 0.2*0.2; //0.01 0.04 0.16  0.02*0.02=0.0004  eps less smooth weaker
    eps *= 255 * 255;   // Because the intensity range of our images is [0, 255]*/

}
