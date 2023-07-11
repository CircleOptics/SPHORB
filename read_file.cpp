#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "SPHORB.h"
#include "utility.h"

using namespace std;
using namespace cv;

void readKeypointsMatchesFromFile(const string& filename, vector<KeyPoint>& keypoints1, vector<KeyPoint>& keypoints2, Matches& matches) {
    ifstream file(filename);
    if (!file.is_open()) {
        cout << "Failed to open " << filename << endl;
        return;
    }

    string line;
    // Read keypoints
    getline(file, line); // Read the line "Keypoints1: <numKeypoints1>"
    string numKeypointsStr = line.substr(line.find(":") + 2); // Skip "Keypoints1: "
    int numKeypoints1 = stoi(numKeypointsStr);

    cout << "Number of keypoints " << numKeypoints1 << endl;
    for (int i = 0; i < numKeypoints1; i++) {
    	getline(file, line);
    	istringstream iss(line);
        float x, y, size, angle, response, octave, class_id;
        iss >> x >> y >> size >> angle >> response >> octave >> class_id;
        keypoints1.push_back(KeyPoint(x, y, size, angle, response, octave, class_id));
    }

    // Read keypoints2
    getline(file, line); // Read empty line
    getline(file, line); // Read the line "Keypoints2: <numKeypoints2>"
    cout << "Line: " << line << endl;
    numKeypointsStr = line.substr(line.find(":") + 2); // Skip "Keypoints2: "
    int numKeypoints2 = stoi(numKeypointsStr);
    for (int i = 0; i < numKeypoints2; i++) {
        getline(file, line);
    	istringstream iss(line);
        float x, y, size, angle, response, octave, class_id;
        iss >> x >> y >> size >> angle >> response >> octave >> class_id;
        keypoints2.push_back(KeyPoint(x, y, size, angle, response, octave, class_id));
    }

    // Read matches
    getline(file, line); // Read empty line
    getline(file, line); // Read the line "Matches: <numMatches>"
    string numMatchesStr = line.substr(line.find(":") + 2); // Skip "Matches: "
    int numMatches = stoi(numMatchesStr);
    for (int i = 0; i < numMatches; i++) {
    	getline(file, line);
    	istringstream iss(line);
        int queryIdx, trainIdx;
        float distance;
        iss >> queryIdx >> trainIdx >> distance;
        matches.push_back(DMatch(queryIdx, trainIdx, distance));
        cout << "Line: " << line << endl;
    }

    file.close();
}