#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "SPHORB.h"
#include "utility.h"
#include <iomanip>  // Include the <iomanip> header for std::setprecision

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

int main(int argc, char* argv[])
{
    float ratio = 0.75f;
    SPHORB sorb;
    BFMatcher matcher(NORM_HAMMING, false);
    
    // Check if a directory path is provided
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <directory_path>" << endl;
        return 1;
    }

    string directoryPath = argv[1];

    // Check if the provided path is a directory
    if (!fs::is_directory(directoryPath)) {
        cerr << "Error: Invalid directory path." << endl;
        return 1;
    }

    vector<string> filePaths;

    // Collect the paths of all image files in the directory
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (fs::is_regular_file(entry) && entry.path().extension() != ".txt") {
            string filePath = entry.path().string();

            // Add the file path to the vector
            filePaths.push_back(filePath);
        }
    }

    // Iterate over all image files
    for (size_t i = 0; i < filePaths.size(); i++) {
        string filePath1 = filePaths[i];
        Mat image1 = imread(filePath1);
        if (image1.empty()) {
            cerr << "Error: Could not read the image " << filePath1 << endl;
            continue;
        }

        Mat descriptors1;
        vector<KeyPoint> kPoint1;

        // Calculate SORB keypoints and descriptors for the first image
        sorb(image1, Mat(), kPoint1, descriptors1);

        cout << "Processing file: " << filePath1 << endl;
        cout << "Keypoints: " << kPoint1.size() << endl;

        string keypointsFilePath1 = filePath1 + "_keypoints.txt";

        ifstream fileExists(keypointsFilePath1);
        if (!fileExists) {
            ofstream keypointsFile1(keypointsFilePath1);

            if (!keypointsFile1.is_open()) {
                cerr << "Error: Could not create keypoints file for " << filePath1 << endl;
                continue;
            }

            keypointsFile1 << "Keypoints1: " << kPoint1.size() << std::endl;
            for (const cv::KeyPoint& kp : kPoint1) {
                keypointsFile1 << std::fixed << std::setprecision(15)
                   << kp.pt.x << " " << kp.pt.y << " "
                   << std::setprecision(2)
                   << kp.size << " "
                   << std::fixed << std::setprecision(15)
                   << kp.angle << " "
                   << std::setprecision(2)
                   << kp.response << " " << kp.octave << " " << kp.class_id << endl;
            }

            keypointsFile1.close();
        }

        // Iterate over the remaining image files
        for (size_t j = i + 1; j < filePaths.size(); j++) {
            string filePath2 = filePaths[j];
            Mat image2 = imread(filePath2);
            if (image2.empty()) {
                cerr << "Error: Could not read the image " << filePath2 << endl;
                continue;
            }

            Mat descriptors2;
            vector<KeyPoint> kPoint2;

            // Calculate SORB keypoints and descriptors for the second image
            sorb(image2, Mat(), kPoint2, descriptors2);

            cout << "Processing file: " << filePath2 << endl;
            cout << "Keypoints: " << kPoint2.size() << endl;

            // Find matches between the descriptors of the two images
            Matches matches;
            vector<Matches> dupMatches;
            matcher.knnMatch(descriptors1, descriptors2, dupMatches, 2);
            ratioTest(dupMatches, ratio, matches);

            cout << "Matches: " << matches.size() << endl;

            // Save the matches to a file with the suffix "im1_number" and "im2_number"
            string matchesFilePath1 = filePath1 + "_" + to_string(j) + ".txt";

            ofstream matchesFile(matchesFilePath1);

            if (!matchesFile.is_open()) {
                cerr << "Error: Could not create matches files for " << filePath1 << endl;
                continue;
            }

            matchesFile << "Matches: " << matches.size() << std::endl;
            for (const cv::DMatch& match : matches) {
                matchesFile << match.queryIdx << " " << match.trainIdx << " " << std::fixed << std::setprecision(2) << match.distance << std::endl;
            }
            matchesFile.close();


            matchesFile.close();
        }
    }

    return 0;
}
