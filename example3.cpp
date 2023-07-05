#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include "SPHORB.h"
#include "utility.h"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

int main(int argc, char* argv[])
{
    float ratio = 0.75f;
    SPHORB sorb;

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

    // Iterate over all files in the directory
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (fs::is_regular_file(entry)) {
            string filePath = entry.path().string();

            if (filePath.substr(filePath.size() - 4) == ".txt") {
                continue; // Skip processing this file
            }
            cout << "Processing file: " << filePath << endl;

            // Load the image
            Mat image = imread(filePath);
            if (image.empty()) {
                cerr << "Error: Could not read the image " << filePath << endl;
                continue;
            }

            // Calculate SORB keypoints
            Mat descriptors;
            vector<KeyPoint> keypoints;
            sorb(image, Mat(), keypoints, descriptors);

            // Save keypoints to a file
            string keypointsFilePath = filePath + "_keypoints.txt";
            ofstream keypointsFile(keypointsFilePath);
            if (!keypointsFile.is_open()) {
                cerr << "Error: Could not create keypoints file " << keypointsFilePath << endl;
                continue;
            }

            for (const auto& keypoint : keypoints) {
                keypointsFile << keypoint.pt.x << " " << keypoint.pt.y << endl;
            }
            keypointsFile.close();

            // Save descriptors to a file
            string descriptorsFilePath = filePath + "_descriptors.txt";
            ofstream descriptorsFile(descriptorsFilePath);
            if (!descriptorsFile.is_open()) {
                cerr << "Error: Could not create descriptors file " << descriptorsFilePath << endl;
                continue;
            }

            for (int i = 0; i < descriptors.rows; i++) {
                for (int j = 0; j < descriptors.cols; j++) {
                    descriptorsFile << (int)descriptors.at<uchar>(i, j) << " ";
                }
                descriptorsFile << endl;
            }
            descriptorsFile.close();
        }
    }

    return 0;
}
