
#ifndef _WIN32
#include <unistd.h>
#define _wgetcwd getcwd
#ifndef MAX_PATH
#define MAX_PATH 4096
#endif
#endif

#include "common.h"

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

const char* projectPath;


void extractFeatures(const Mat_<uchar>& img, vector<KeyPoint>& keypoints, Mat_<uchar>& descriptors)
{
    Ptr<ORB> orb=ORB::create(1000);

    // whole image analyzed, extraction of keypoints and descriptor calc done in one
    orb->detectAndCompute(
        img,
        noArray(),
        keypoints,
        descriptors
    );
}

vector<DMatch> matchDescriptors(const Mat_<uchar>& descriptors1, const Mat_<uchar>& descriptors2)
{
    // hamming distance = nr of different bits between two binary descriptors
    // mutual matching enforced through crossCheck=true
    BFMatcher matcher(NORM_HAMMING, true);

    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    return matches;
}

vector<DMatch> filterBestMatches(vector<DMatch> matches, int maxMatches)
{
    sort(matches.begin(), matches.end(),
        [](DMatch& a, DMatch& b) {
            return a.distance < b.distance;
        });

    if(matches.size()>maxMatches)
        matches.resize(maxMatches);

    return matches;
}

void showKeypoints(const string& windowName, const Mat_<uchar>& img,const vector<KeyPoint>& keypoints)
{
    Mat_<Vec3b> output;

    drawKeypoints(
        img,
        keypoints,
        output,
        Scalar::all(-1),
        DrawMatchesFlags::DRAW_RICH_KEYPOINTS
    );

    imshow(windowName,output);
}

void showMatches(
    const Mat_<uchar>& img1,
    const vector<KeyPoint>& keypoints1,
    const Mat_<uchar>& img2,
    const vector<KeyPoint>& keypoints2,
    const vector<DMatch>& matches
)
{
    Mat_<Vec3b> output;

    drawMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        matches,
        output,
        Scalar::all(-1),
        Scalar(),
        vector<char>(),
        DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS
    );

    imshow("Final matches",output);
}


int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    string path1="../Images/carte1.jpeg";
    string path2="../Images/carte2.jpeg";

    Mat_<uchar> img1=imread(path1, IMREAD_GRAYSCALE);
    Mat_<uchar> img2=imread(path2, IMREAD_GRAYSCALE);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat_<uchar> descriptors1, descriptors2;

    extractFeatures(img1, keypoints1, descriptors1);
    extractFeatures(img2, keypoints2, descriptors2);

    cout<<"Image 1 keypoints: "<<keypoints1.size()<<endl;
    cout<<"Image 2 keypoints: "<<keypoints2.size()<<endl;

    if (descriptors1.empty()||descriptors2.empty()) {
        cout<<"No descriptors found."<<endl;
        return -1;
    }

    auto matches=matchDescriptors(descriptors1, descriptors2);
    auto goodMatches=filterBestMatches(matches, 30);

    cout<<"Total matches: "<<matches.size()<<endl;
    cout<<"Displayed matches: "<<goodMatches.size()<<endl;

    showKeypoints("Keypoints image 1",img1,keypoints1);
    showKeypoints("Keypoints image 2",img2,keypoints2);

    showMatches(
        img1,
        keypoints1,
        img2,
        keypoints2,
        goodMatches
    );

    waitKey(0);
    return 0;
}