
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



//-----------------------------------------MATCHING------------------------------------


vector<DMatch> matchDescriptorsBuiltIn(const Mat_<uchar>& descriptors1, const Mat_<uchar>& descriptors2)
{
    // hamming distance = nr of different bits between two binary descriptors
    // mutual matching enforced through crossCheck=true
    BFMatcher matcher(NORM_HAMMING, true);

    vector<DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    return matches;
}

// for ORB, descriptor is always 32 bytes, but keep function general
int hammingDistance(const uchar* d1, const uchar* d2, int length) {
    int distance = 0;
    for (int i = 0; i < length; i++) {
        uchar xorVal = (uchar)d1[i] ^ (uchar)d2[i]; // bits set to 1 where the descriptors differ

        // count the different bits
        int cnt=0;
        for (int j = 0; j < 8; j++)
            if ((xorVal & (1<<j))!=0)
                cnt++;

        distance+=cnt;
    }
    return distance;
}

// for each descriptor in image1, find the closest descriptor in image2
// one-directional: multiple descriptors from image1 may match the same descriptor in image2.
vector<DMatch> matchNearestNeighbor(const Mat_<uchar>& descriptors1, const Mat_<uchar>& descriptors2) {
    vector<DMatch>  matches;

    for (int i = 0; i < descriptors1.rows; i++) {
        int matchIndex=-1;
        int matchDist=INT_MAX;

        for (int j = 0; j < descriptors2.rows; j++) {
            int dist=hammingDistance( descriptors1.ptr<uchar>(i),descriptors2.ptr<uchar>(j),descriptors1.cols);

            if (dist<matchDist) {
                matchDist = dist;
                matchIndex = j;
            }
        }

        if (matchIndex>=0)
            matches.push_back(DMatch(i, matchIndex, matchDist));

    }
   return matches;
}

// cross-check
// bidirectional (INTERSECTION op) - both matches coming form both images must agree
vector<DMatch> matchCrossCheck(const Mat_<uchar>& descriptors1, const Mat_<uchar>& descriptors2) {
    auto matches1=matchNearestNeighbor(descriptors1,descriptors2);
    auto matches2=matchNearestNeighbor(descriptors2,descriptors1);

    vector<DMatch> matches;

    for ( auto& m1 : matches1) {
        int i = m1.queryIdx;
        int j = m1.trainIdx;

        for (const auto& m2 : matches2) {
            if (m2.queryIdx == j && m2.trainIdx == i) {
                matches.push_back(m1);
                break;
            }
        }
    }

    return matches;
}

// for each descriptor in image2, keep only the closest match from image1. (DIFFERENCE op considering hamming distance)
vector<DMatch> matchUnique(const Mat_<uchar>& descriptors1, const Mat_<uchar>& descriptors2) {
    map<int, DMatch> matches;

    for (int i = 0; i < descriptors1.rows; i++) {
        int matchIndex=-1;
        int matchDist=INT_MAX;

        for (int j = 0; j < descriptors2.rows; j++) {
            int dist=hammingDistance( descriptors1.ptr<uchar>(i),descriptors2.ptr<uchar>(j),descriptors1.cols);

            if (dist<matchDist) {
                matchDist = dist;
                matchIndex = j;
            }
        }

        if (matchIndex>=0) {
            // keep only smallest distance s.t. no keypoint (from img2) is in 2 pairs
            DMatch currentMatch(i, matchIndex, matchDist);

            if (matches.find(matchIndex) == matches.end() || currentMatch.distance < matches[matchIndex].distance) {
                matches[matchIndex]=currentMatch;
            }

        }
    }

    vector<DMatch> trueMatches;

    for (auto& pair : matches) {
        trueMatches.push_back(pair.second);
    }

    return trueMatches;
}


vector<DMatch> differenceMatches(const vector<DMatch>& A, const vector<DMatch>& B)
{
    vector<DMatch> diff;

    for (const auto& m1 : A) {
        bool found = false;

        for (const auto& m2 : B) {
            if (m1.queryIdx == m2.queryIdx && m1.trainIdx == m2.trainIdx) {
                found = true;
                break;
                }
        }

        if (!found) {
            diff.push_back(m1);
        }
    }

    return diff;
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


// ------------------------ drawing -------------------------


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
    const string& windowName,
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

    imshow(windowName,output);
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

    // ------------------------------------------ MATCHING ---------------------------------------------

    int maxMatches=50;
    auto matchesBuiltIn=matchDescriptorsBuiltIn(descriptors1, descriptors2);
    auto goodMatchesBuiltIn=filterBestMatches(matchesBuiltIn, maxMatches);

    // cout<<"Total matches: "<<matches.size()<<endl;
    // cout<<"Displayed matches: "<<goodMatches.size()<<endl;

    // showKeypoints("Keypoints image 1",img1,keypoints1);
    // showKeypoints("Keypoints image 2",img2,keypoints2);

    showMatches(
        "OpenCV BFMatcher crossCheck",
        img1,
        keypoints1,
        img2,
        keypoints2,
        goodMatchesBuiltIn
    );

    auto matches1=matchUnique(descriptors1, descriptors2);
    auto goodMatches1=filterBestMatches(matches1, maxMatches);
    showMatches("Manual unique-on-image2 matching",img1,keypoints1,img2,keypoints2,goodMatches1);

    auto matches2=matchCrossCheck(descriptors1, descriptors2);
    auto goodMatches2=filterBestMatches(matches2, maxMatches);
    showMatches("Manual mutual matching",img1,keypoints1,img2,keypoints2,goodMatches2);

    auto onlyIn1 = differenceMatches(goodMatches1, goodMatches2);
    auto onlyIn2 = differenceMatches(goodMatches2, goodMatches1);

    showMatches("Only in match1", img1, keypoints1, img2, keypoints2, onlyIn1);
    showMatches("Only in match2", img1, keypoints1, img2, keypoints2, onlyIn2);

    waitKey(0);
    return 0;
}