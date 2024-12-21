#include "imageUtil.hpp"

using namespace std;

cv::Mat readImage(string path) {
	cv::Mat rgbImage = cv::imread(path, cv::IMREAD_COLOR);

	return rgbImage;
}