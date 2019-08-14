#include "FastGuidedFilter/fastguidedfilter.h"
using namespace std;
#define DEBUG
const int s = 2, r = 8;
const float eps = 0.04 * 0.04 * 255 * 255;
const int output_width = 854, output_height = 480;

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << argv[0] << " source_video output_video" << endl;
        return -1;
    }
    cv::Mat output, I, input, input_rz;
    cv::VideoCapture cap(argv[1]);
    cv::VideoWriter out;
    if (cap.isOpened()) {
        double fps = cap.get( cv::CAP_PROP_FPS );
        int width = int( cap.get( cv::CAP_PROP_FRAME_WIDTH ) + 0.5 );
        int height = int( cap.get( cv::CAP_PROP_FRAME_HEIGHT ) + 0.5 );
        out.open(argv[2], cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(output_width, output_height));
    } else {
        cerr << "Open file " << argv[1] << "failed. " << std::endl;
    }
    if (out.isOpened()) {
        cap >> input;
        while (!input.empty()) {
            cv::resize(input, input_rz, cv::Size(output_width, output_height));
            I = input_rz;
#ifdef DEBUG
            clock_t start_time1 = clock();
#endif
            FastGuidedFilter fgfilter(I, r, eps,s);
            fgfilter.filter(output, I, -1);
#ifdef DEBUG
            clock_t end_time1 = clock();
            cout << "Running time is: "
                << static_cast<double>(end_time1 - start_time1) / CLOCKS_PER_SEC * 1000 << "ms" << endl;
            string name = "color_output.jpg";
#endif
            out << output;
            cap >> input;
        }
    }
    return 0;
}
