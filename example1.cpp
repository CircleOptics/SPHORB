/*
	AUTHOR:
	Qiang Zhao, email: qiangzhao@tju.edu.cn
	Copyright (C) 2015 Tianjin University
	School of Computer Software
	School of Computer Science and Technology

	LICENSE:
	SPHORB is distributed under the GNU General Public License.  For information on 
	commercial licensing, please contact the authors at the contact address below.

	REFERENCE:
	@article{zhao-SPHORB,
	author   = {Qiang Zhao and Wei Feng and Liang Wan and Jiawan Zhang},
	title    = {SPHORB: A Fast and Robust Binary Feature on the Sphere},
	journal  = {International Journal of Computer Vision},
	year     = {2015},
	volume   = {113},
	number   = {2},
	pages    = {143-159},
	}
*/

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "SPHORB.h"
#include "utility.h"
using namespace std;
using namespace cv;
#include <fstream>
#include <iomanip>  // Include the <iomanip> header for std::setprecision
#include "read_file.h"

int main(int argc, char * argv[])
{
	float ratio = 0.75f;
	SPHORB sorb;

	Mat img1 = imread(argv[1]);
	Mat img2 = imread(argv[2]);

	Mat descriptors1;
	Mat descriptors2;

	vector<KeyPoint> kPoint1;
	vector<KeyPoint> kPoint2;

	sorb(img1, Mat(), kPoint1, descriptors1);
	sorb(img2, Mat(), kPoint2, descriptors2);

	cout<<"Keypoint1: "<<kPoint1.size()<<", Keypoint2: "<<kPoint2.size()<<endl;

	BFMatcher matcher(NORM_HAMMING, false);
	Matches matches;

	vector<Matches> dupMatches;
	matcher.knnMatch(descriptors1, descriptors2, dupMatches, 2);
	ratioTest(dupMatches, ratio, matches);
	cout<<"Matches: "<<matches.size()<<endl;

	Mat imgMatches;
	::drawMatches(img1, kPoint1, img2, kPoint2, matches, imgMatches, Scalar::all(-1), Scalar::all(-1),  
	vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS,true);

	imwrite("1_matches.jpg", imgMatches);

	// Write keypoints and matches to a file
    std::ofstream outputFile("keypoints_matches.txt");
    if (outputFile.is_open()) {
        // Write kPoint1
        outputFile << "Keypoints1: " << kPoint1.size() << std::endl;
        for (const cv::KeyPoint& kp : kPoint1) {
            outputFile << std::fixed << std::setprecision(15)
               << kp.pt.x << " " << kp.pt.y << " "
               << std::setprecision(2)
               << kp.size << " "
               << std::fixed << std::setprecision(15)
               << kp.angle << " "
               << std::setprecision(2)
               << kp.response << " " << kp.octave << " " << kp.class_id << endl;
        }
        outputFile << std::endl;

        // Write kPoint2
        outputFile << "Keypoints2: " << kPoint2.size() << std::endl;
        for (const cv::KeyPoint& kp : kPoint2) {
            outputFile << std::fixed << std::setprecision(15)
               << kp.pt.x << " " << kp.pt.y << " "
               << std::setprecision(2)
               << kp.size << " "
               << std::fixed << std::setprecision(15)
               << kp.angle << " "
               << std::setprecision(2)
               << kp.response << " " << kp.octave << " " << kp.class_id << endl;
        }
        outputFile << std::endl;

        // Write matches
        outputFile << "Matches: " << matches.size() << std::endl;
        for (const cv::DMatch& match : matches) {
			outputFile << match.queryIdx << " " << match.trainIdx << " " << std::setprecision(2) << match.distance << std::endl;
		}
        outputFile.close();

        // Read keypoints and matches from file
        vector<KeyPoint> readKPoint1, readKPoint2;
        vector<DMatch> readMatches;
        readKeypointsMatchesFromFile("keypoints_matches.txt", readKPoint1, readKPoint2, readMatches);

        // Compare the number of keypoints and matches
        cout << "Number of keypoints1 (original vs read): " << kPoint1.size() << " vs " << readKPoint1.size() << endl;
        cout << "Number of keypoints2 (original vs read): " << kPoint2.size() << " vs " << readKPoint2.size() << endl;
        cout << "Number of matches (original vs read): " << matches.size() << " vs " << readMatches.size() << endl;

        // Validate individual keypoints and matches
        for (size_t i = 0; i < kPoint1.size(); i++) {
            const KeyPoint& origKP = kPoint1[i];
            const KeyPoint& readKP = readKPoint1[i];
            cout << "Keypoint1[" << i << "] (original vs read):" << endl;
            cout << "  x: " << origKP.pt.x << " vs " << readKP.pt.x << endl;
            cout << "  y: " << origKP.pt.y << " vs " << readKP.pt.y << endl;
            // Compare other attributes as needed
        }

        for (size_t i = 0; i < kPoint2.size(); i++) {
            const KeyPoint& origKP = kPoint2[i];
            const KeyPoint& readKP = readKPoint2[i];
            cout << "Keypoint2[" << i << "] (original vs read):" << endl;
            cout << "  x: " << origKP.pt.x << " vs " << readKP.pt.x << endl;
            cout << "  y: " << origKP.pt.y << " vs " << readKP.pt.y << endl;
            // Compare other attributes as needed
        }

        for (size_t i = 0; i < matches.size(); i++) {
            const DMatch& origMatch = matches[i];
            const DMatch& readMatch = readMatches[i];
            cout << "Match[" << i << "] (original vs read):" << endl;
            cout << "  queryIdx: " << origMatch.queryIdx << " vs " << readMatch.queryIdx << endl;
            cout << "  trainIdx: " << origMatch.trainIdx << " vs " << readMatch.trainIdx << endl;
        }
    } else {
        std::cerr << "Failed to open the output file." << std::endl;
    }
}

// Take images and resize to 1280 x 640
// For each image pair, write the imgMatches and the actual data to a file
// Remove outliers from the data - in particular the edge pixels (could use the mask to do this)