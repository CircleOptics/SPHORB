#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "SPHORB.h"
#include "utility.h"
#include "read_file.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: ./program keypoints_matches_file" << endl;
        return 1;
    }

    float ratio = 0.75f;
    Matches matches;

    // Read keypoints and matches from file
    vector<KeyPoint> kPoint1, kPoint2;
    readKeypointsMatchesFromFile(argv[1], kPoint1, kPoint2, matches);

    // Load images
    Mat img1 = imread(argv[2]);
    Mat img2 = imread(argv[3]);

    cout << "Keypoint1: " << kPoint1.size() << ", Keypoint2: " << kPoint2.size() << endl;

    cout << "Matches: " << matches.size() << endl;

    Mat imgMatches;
    ::drawMatches(img1, kPoint1, img2, kPoint2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS, true);

    imwrite("1_matches.jpg", imgMatches);

    return 0;
}
