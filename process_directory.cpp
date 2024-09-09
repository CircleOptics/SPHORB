#include <iostream>
#include <vector>
#include <sstream>
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
    // change to make this follow our dataset file structure (see lexicon onenote)
    string equirectangular_dir_path = directoryPath + "/equirectangulars";
    string keypoints_dir_path = directoryPath + "/keypoints/SPHORB";
    string matches_dir_path = directoryPath + "/matches/SPHORB";
    if(!fs::is_directory(equirectangular_dir_path))
    {
        cerr << "Error: no Equirectangulars folder within this dir." << endl;
        return 1;
    }
    // create intermediate keypoints dir
    fs::create_directory(directoryPath + "/keypoints");
    // create actual keypoints dir
    fs::create_directory(keypoints_dir_path);
    // do the same for matches
    fs::create_directory(directoryPath + "/matches");
    fs::create_directory(matches_dir_path);

    vector<string> filePaths;

    // Collect the paths of all image files in the directory
    for (const auto& entry : fs::directory_iterator(equirectangular_dir_path)) {
        if (fs::is_regular_file(entry) && entry.path().extension() != ".txt") {
            string filePath = entry.path().string();

            // Add the file path to the vector
            filePaths.push_back(filePath);
        }
    }

    vector<vector<KeyPoint>> keypointsList;
    vector<Mat> descriptorsList;

    // Iterate over all image files to calculate keypoints and descriptors
    for (const string& filePath : filePaths) {
        Mat image = imread(filePath);
        //vector<string> tokens = split(matches_dir_path, '/');
        if (image.empty()) {
            cerr << "Error: Could not read the image " << filePath << endl;
            continue;
        }

        vector<KeyPoint> keypoints;
        Mat descriptors;

        // Calculate SORB keypoints and descriptors for the image
        sorb(image, Mat(), keypoints, descriptors);
        string name = fs::path(filePath).stem().string();
        cout << "Processing file: " << name << " ... ";
        cout << "Keypoints: " << keypoints.size() << endl;

        // Store the keypoints and descriptors in lists
        keypointsList.push_back(keypoints);
        descriptorsList.push_back(descriptors);

        // Get the file name without the extension
        string baseFileName = fs::path(filePath).stem().string();

        // Get the original directory path
        //string directoryPath = fs::path(filePath).parent_path().string();

        // Create a path object for the "keypoints" subdirectory
        //fs::path keypointsSubdirectory = fs::path(directoryPath) / "keypoints";
        fs::path keypointsSubdirectory = fs::path(keypoints_dir_path);
        // Create the "keypoints" subdirectory if it doesn't exist
        fs::create_directory(keypointsSubdirectory);

        // Create the keypoints file path with the original directory
        string keypointsFilePath = (keypointsSubdirectory / baseFileName).string() + "_keypoints.txt";

        // cout << keypointsFilePath << endl;

        ifstream fileExists(keypointsFilePath);
        if (!fileExists) {
            ofstream keypointsFile(keypointsFilePath);

            if (!keypointsFile.is_open()) {
                cerr << "Error: Could not create keypoints file for " << filePath << endl;
                continue;
            }

            keypointsFile << "Keypoints1: " << keypoints.size() << std::endl;
            for (const cv::KeyPoint& kp : keypoints) {
                keypointsFile << std::fixed << std::setprecision(15)
                    << kp.pt.x << " " << kp.pt.y << " "
                    << std::setprecision(2)
                    << kp.size << " "
                    << std::fixed << std::setprecision(15)
                    << kp.angle << " "
                    << std::setprecision(2)
                    << kp.response << " " << kp.octave << " " << kp.class_id << endl;
            }


            keypointsFile.close();
        }
    }

    // Iterate over keypoints and descriptors to calculate matches
    for (size_t i = 0; i < keypointsList.size(); i++) {
        for (size_t j = i + 1; j < keypointsList.size(); j++) {
            vector<KeyPoint>& keypoints1 = keypointsList[i];
            Mat& descriptors1 = descriptorsList[i];

            vector<KeyPoint>& keypoints2 = keypointsList[j];
            Mat& descriptors2 = descriptorsList[j];

            // Find matches between the descriptors of the two images
            BFMatcher matcher(NORM_HAMMING, false);
            Matches matches;
            vector<Matches> dupMatches;
            matcher.knnMatch(descriptors1, descriptors2, dupMatches, 2);
            ratioTest(dupMatches, ratio, matches);

            // Save the matches to a file with the suffix "im1_number" and "im2_number"
            string baseFileName1 = fs::path(filePaths[i]).stem().string();
            string baseFileName2 = fs::path(filePaths[j]).stem().string();

            cout << "Processing matches for files: " << baseFileName1 << " and " << baseFileName2 << " ... ";
            cout << "Matches: " << matches.size() << endl;


            // Get the original directory path
            //string directoryPath = fs::path(filePaths[i]).parent_path().string();

            // Create a path object for the "matches" subdirectory
            //fs::path matchesSubdirectory = fs::path(directoryPath) / "matches";
            fs::path matchesSubdirectory = fs::path(matches_dir_path);
            // Create the "matches" subdirectory if it doesn't exist
            //fs::create_directory(matchesSubdirectory);

            // Create a subdirectory within "matchesSubdirectory" using baseFileName1
            fs::path subdirectory = matchesSubdirectory / baseFileName1;
            fs::create_directory(subdirectory);

            // Create the matches file path with the original directory
            string matchesFilePath = (subdirectory / (baseFileName1 + "_" + baseFileName2 + ".txt")).string();

            ofstream matchesFile(matchesFilePath);

            if (!matchesFile.is_open()) {
                cerr << "Error: Could not create matches files for " << filePaths[i] << endl;
                continue;
            }

            matchesFile << "Matches: " << matches.size() << std::endl;
            for (const cv::DMatch& match : matches) {
                matchesFile << match.queryIdx << " " << match.trainIdx << " " << std::fixed << std::setprecision(2) << match.distance << std::endl;
            }
            matchesFile.close();
        }
    }

    return 0;
}
