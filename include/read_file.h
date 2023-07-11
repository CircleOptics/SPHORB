#ifndef READ_FILE_H
#define READ_FILE_H

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace cv;

typedef std::vector<DMatch> Matches;

void readKeypointsMatchesFromFile(const std::string& filename, std::vector<KeyPoint>& keypoints1, std::vector<KeyPoint>& keypoints2, Matches& matches);

#endif // READ_FROM_FILE_H