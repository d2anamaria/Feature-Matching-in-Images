
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

// extraction: FAST + rotated BRIEF for description
void extractFeatures(const Mat_<uchar>& img, vector<KeyPoint>& keypoints, Mat_<uchar>& descriptors)
{
    Ptr<ORB> orb=ORB::create(3000);

    // whole image analyzed, extraction of keypoints and descriptor calc done in one
    orb->detectAndCompute(
        img,
        noArray(),
        keypoints,
        descriptors
    );
}

//----------------------------------------- DESCRIPTORS DEF ------------------------------------

// descriptors that store intensities, not relativity
Mat_<uchar> computeDescriptors(const Mat_<uchar>& img, const vector<KeyPoint>& keypoints, int patchSize = 16)
{

    int half=patchSize/2;
    Mat_<uchar> descriptors( (int)keypoints.size(), patchSize * patchSize);

    for (int k = 0; k < keypoints.size(); k++) {
        Point2f pt = keypoints[k].pt;

        for (int u = -half; u < half; u++) {
            for (int v = -half; v < half; v++) {
                int y = pt.y + u;
                int x = pt.x + v;

                int idx = (u + half) * patchSize + (v + half); //idx inside patch

                if (y >= 0 && y < img.rows && x >= 0 && x < img.cols)
                    descriptors(k, idx) = img(y, x);
                else
                    descriptors(k, idx) = 0;
            }
        }
    }

    return descriptors;
}

bool isInside2(const Mat_<uchar>& img, int i, int j) {
    return 0 <= i && i < img.rows &&
           0 <= j && j < img.cols;
}

// binary descriptors storing relative intensity comparisons
// each bit says: is current patch pixel brighter than its neighbor?
Mat_<uchar> computeBinaryDescriptors(const Mat_<uchar>& img, const vector<KeyPoint>& keypoints, int patchSize = 16)
{
    int half = patchSize / 2;

    // compare each patch pixel with right and down neighbor
    int nrComparisons = patchSize * patchSize * 2;

    // 8 comparisons are packed into 1 byte
    int descriptorBytes = (nrComparisons + 7) / 8;

    Mat_<uchar> descriptors((int)keypoints.size(), descriptorBytes);
    descriptors.setTo(0);

    for (int k = 0; k < keypoints.size(); k++) {
        Point2f pt = keypoints[k].pt;

        int bitIdx = 0;

        for (int u = -half; u < half; u++) {
            for (int v = -half; v < half; v++) {
                int y = (int)pt.y + u;
                int x = (int)pt.x + v;

                // comparison 1: current pixel vs right neighbor
                if (isInside2(img, y, x) && isInside2(img, y, x + 1) && v + 1 < half)
                    if (img(y, x) > img(y, x + 1))
                        descriptors(k, bitIdx / 8) |= (1 << (bitIdx % 8));
                bitIdx++;

                // comparison 2: current pixel vs down neighbor
                if (isInside2(img, y, x) && isInside2(img, y + 1, x) && u + 1 < half)
                    if (img(y, x) > img(y + 1, x))
                        descriptors(k, bitIdx / 8) |= (1 << (bitIdx % 8));
                bitIdx++;
            }
        }
    }

    return descriptors;
}

// use ORB to calc descriptors for already manually detected keypoints
Mat_<uchar> computeORBDescriptors(const Mat_<uchar>& img, vector<KeyPoint>& keypoints, int patchSize = 16)
{
    Ptr<ORB> orb = ORB::create();
    Mat_<uchar> descriptors;

    orb->compute(
        img,
        keypoints,
        descriptors
    );

    return descriptors;
}



//----------------------------------------- EXTRACTION ------------------------------------
float harrisValue(float Ixx, float Iyy, float Ixy)
{
    float k = 0.04f; // suppress trace to penalize edges (det grows much faster than trace^2)

    // det large when strong intensity variation exists in TWO independent directions
    // Ixy² penalizes highly correlated gradients (e.g. diagonal edges)
    float det = Ixx * Iyy - Ixy * Ixy;

    // total intensity variation, alone cannot distinguish between corner or edge
    float trace = Ixx + Iyy;

    // Harris response:
    // large positive -> corner
    // negative/small -> edge or flat region
    return det - k * trace * trace;
}


void harrisCornerDetection(
    const Mat_<uchar>& img,
    vector<KeyPoint>& keypoints,
    Mat_<uchar>& descriptors
) {

    Mat_<uchar> smooth;
    GaussianBlur(img, smooth, Size(5, 5), 1.0); //?? best kernel?

    Mat_<float> Ix, Iy;
    Sobel(smooth, Ix, CV_32F, 1, 0, 3); //first derivative on x axis
    Sobel(smooth, Iy, CV_32F, 0, 1, 3); //first derivative on y axis


    Mat_<float> R(img.size());
    R.setTo(0);

    int window=3;
    int k=window/2;
    float maxR = 0;

    // E(u,v) = Σ [ I(x+u,y+v)-I(x,y) ]²
    // Taylor expansion : I(x+u,y+v) ≈ I(x,y) + Ix*u + Iy*v

    // => E(u,v) = ≈ Σ [ Ix*u + Iy*v ]² = = Σ (Ix²u² + 2IxIyuv + Iy²v²) = = u²ΣIx² + 2uvΣIxIy + v²ΣIy²
    // this way we avoid computation overhead from  shifting

    for (int i=k;i<img.rows-k;i++) {
        for (int j=k;j<img.cols-k;j++) {
            float Ixx = 0;
            float Iyy = 0;
            float Ixy = 0;

            for (int u=-k; u<=k; u++) {
                for (int v=-k; v<=k; v++) {
                    float gx = Ix(i + u, j + v);
                    float gy = Iy(i + u, j + v);

                    Ixx += gx * gx;
                    Iyy += gy * gy;
                    Ixy += gx * gy; // higher value if gradients are strongly correlated
                }
            }

            R(i, j) = harrisValue(Ixx, Iyy, Ixy);
            maxR = max(maxR, R(i, j));
        }
    }

    // filter redundant dupes of features - keep only pixel that best represents the corner
    // ?should compare to average in window?
    float T = 0.005f * maxR;
    k=8;

    for (int i=k;i<img.rows-k;i++) {
        for (int j=k;j<img.cols-k;j++) {
            if (R(i,j)<T) continue;

            bool localMax = true;

            for (int u=-k; u<=k; u++) {
                for (int v=-k; v<=k; v++) {
                    if (R(i + u, j + v) > R(i,j)) {
                        localMax = false;
                        break;
                    }
                }
            }

            if (localMax) {
                keypoints.push_back(KeyPoint(Point2f(j, i), 16));
            }
        }
    }

    descriptors = computeORBDescriptors(img, keypoints);
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
// ONE-DIRECTIONAL: multiple descriptors from image1 may match the same descriptor in image2.
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

// ONE-DIRECTIONAL
// removes inconclusive matches (best match too similar to second best)
vector<DMatch> matchRatioTest(const Mat_<uchar>& descriptors1, const Mat_<uchar>& descriptors2, float ratioThreshold = 0.75f) {

    vector<DMatch>  matches;

    for (int i = 0; i < descriptors1.rows; i++) {
        int matchIndex=-1;
        int matchDist=INT_MAX;
        int secondBestDist=INT_MAX;

        for (int j = 0; j < descriptors2.rows; j++) {
            int dist=hammingDistance( descriptors1.ptr<uchar>(i),descriptors2.ptr<uchar>(j),descriptors1.cols);

            if (dist<matchDist) {
                secondBestDist = matchDist;
                matchDist = dist;
                matchIndex = j;
            }
            else if (dist<secondBestDist)
                secondBestDist = dist;

        }

        if (matchIndex>=0 && (secondBestDist==INT_MAX || (float)matchDist / secondBestDist < ratioThreshold)) {
                matches.push_back(DMatch(i, matchIndex, matchDist));

        }

    }
    return matches;
}


vector<DMatch> matchRatioTest(const Mat_<uchar>& descriptors1, const Mat_<uchar>& descriptors2) {
    return matchRatioTest(descriptors1,descriptors2,0.75f);
}

using OneWayMatcher = vector<DMatch> (*)(
    const Mat_<uchar>&,
    const Mat_<uchar>&
);

// cross-check
// bidirectional (INTERSECTION op) - both matches coming form both images must agree
vector<DMatch> matchCrossCheck(const Mat_<uchar>& descriptors1, const Mat_<uchar>& descriptors2, OneWayMatcher oneWayMatcher) {
    auto matches1=oneWayMatcher(descriptors1,descriptors2);
    auto matches2=oneWayMatcher(descriptors2,descriptors1);

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

// ------------------------------------ MATCHING EVALUATION -------------------------------------------

Mat evaluateMatchesWithAffine(
    const vector<KeyPoint>& keypoints1,
    const vector<KeyPoint>& keypoints2,
    const vector<DMatch>& matches
) {
    vector<Point2f> pts1, pts2;

    for (const auto& m : matches) {
        pts1.push_back(keypoints1[m.queryIdx].pt);
        pts2.push_back(keypoints2[m.trainIdx].pt);
    }

    if (pts1.size() < 3) {
        cout << "Not enough matches for affine estimation." << endl;
        return Mat();
    }

    Mat inlierMask;

    Mat affine = estimateAffinePartial2D(
        pts1,
        pts2,
        inlierMask,
        RANSAC,
        7.0
    );

    if (affine.empty()) {
        cout << "Affine estimation failed." << endl;
        return Mat();;
    }

    int inliers = countNonZero(inlierMask);
    double inlierRatio = (double)inliers / matches.size();

    cout << "Affine inliers: " << inliers << " / " << matches.size() << endl;
    cout << "Inlier ratio: " << inlierRatio * 100 << "%" << endl;

    return inlierMask;
}

// estimateAffinePartial2D manual implement
// highloght correct vs incorrect (red) after affine eval



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


void showMatchesWithInliers(
    const string& windowName,
    const Mat_<uchar>& img1,
    const vector<KeyPoint>& keypoints1,
    const Mat_<uchar>& img2,
    const vector<KeyPoint>& keypoints2,
    const vector<DMatch>& matches,
    const Mat& inlierMask
) {
    Mat output;
    hconcat(img1, img2, output);
    cvtColor(output, output, COLOR_GRAY2BGR);

    int offsetX = img1.cols;

    for (int i = 0; i < matches.size(); i++) {
        Point2f p1 = keypoints1[matches[i].queryIdx].pt;
        Point2f p2 = keypoints2[matches[i].trainIdx].pt;

        p2.x += offsetX;

        Scalar color;

        if (inlierMask.at<uchar>(i))
            color = Scalar(0, 255, 0);
        else
            color = Scalar(0, 0, 255);

        line(output, p1, p2, color, 1);
        circle(output, p1, 3, color, FILLED);
        circle(output, p2, 3, color, FILLED);
    }

    imshow(windowName, output);
}




int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

    string path1="../Images/carte1.jpeg";
    string path2="../Images/carte2.jpeg";

    Mat_<uchar> img1=imread(path1, IMREAD_GRAYSCALE);
    Mat_<uchar> img2=imread(path2, IMREAD_GRAYSCALE);


    //--------------------------------------------------------------- BUILT IN EXTRACTION
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
    showKeypoints("Keypoints image 1",img1,keypoints1);
    showKeypoints("Keypoints image 2",img2,keypoints2);

    vector<KeyPoint> keypointsHarris1, keypointsHarris2;
    Mat_<uchar> descriptorsHarris1, descriptorsHarris2;


    // ------------------------------------------------------------------------HARRIS CORNER DETECTION
    // harrisCornerDetection(img1, keypointsHarris1, descriptorsHarris1);
    // harrisCornerDetection(img2, keypointsHarris2, descriptorsHarris2);
    //
    // cout << "Harris keypoints 1: " << keypointsHarris1.size() << endl;
    // cout << "Harris keypoints 2: " << keypointsHarris2.size() << endl;
    //
    // showKeypoints("Harris keypoints image 1", img1, keypointsHarris1);
    // showKeypoints("Harris keypoints image 2", img2, keypointsHarris2);


    // ------------------------------------------ MATCHING ---------------------------------------------

    //----------------------------------------------------------------------------------- BUILT IN MATCHING
    int maxMatches=30;
    auto matchesBuiltIn=matchDescriptorsBuiltIn(descriptors1, descriptors2);
    auto goodMatchesBuiltIn=filterBestMatches(matchesBuiltIn, maxMatches);

    // cout<<"Total matches: "<<matchesBuiltIn.size()<<endl;
    // cout<<"Displayed matches: "<<goodMatchesBuiltIn.size()<<endl;

    // showMatches(
    //     "OpenCV BFMatcher crossCheck",
    //     img1,
    //     keypoints1,
    //     img2,
    //     keypoints2,
    //     goodMatchesBuiltIn
    // );

    printf("Built In:\n");
    Mat inlierMask = evaluateMatchesWithAffine(keypoints1, keypoints2, goodMatchesBuiltIn);
    showMatchesWithInliers(
        "Affine checked matches Built-In",
        img1,
        keypoints1,
        img2,
        keypoints2,
        goodMatchesBuiltIn,
        inlierMask);

    // auto matchesHarrisBuiltIn = matchDescriptorsBuiltIn(descriptorsHarris1, descriptorsHarris2);
    // auto goodMatchesHarrisBuiltIn = filterBestMatches(matchesHarrisBuiltIn, maxMatches);
    //
    // cout<<"Total matches HARRIS: "<<matchesHarrisBuiltIn.size()<<endl;
    // cout<<"Displayed matches HARRIS: "<<goodMatchesHarrisBuiltIn.size()<<endl;
    //
    // showMatches("HARRIS matches",
    //     img1,
    //     keypointsHarris1,
    //     img2,
    //     keypointsHarris2,
    //     goodMatchesHarrisBuiltIn
    //     );

    //------------------------------------------------------------------------------------- MANUAL MATCHING

    // auto matches1=matchUnique(descriptors1, descriptors2);
    // auto goodMatches1=filterBestMatches(matches1, maxMatches);
    // showMatches("Manual unique-on-image2 matching",img1,keypoints1,img2,keypoints2,goodMatches1);

    auto matchesClosest=matchCrossCheck(descriptors1, descriptors2,matchNearestNeighbor);
    auto goodMatchesClosest=filterBestMatches(matchesClosest, maxMatches);
    // showMatches("Manual mutual matching (closest neighb)",img1,keypoints1,img2,keypoints2,goodMatchesClosest);

    auto matchesRatio=matchCrossCheck(descriptors1, descriptors2,matchRatioTest);
    auto goodMatchesRatio=filterBestMatches(matchesRatio, maxMatches);
    // showMatches("Manual mutual matching (ratio test)",img1,keypoints1,img2,keypoints2,goodMatchesRatio);


    printf("\nNearest Neighb:\n");
    Mat inlierMaskClosest = evaluateMatchesWithAffine(keypoints1, keypoints2, goodMatchesClosest);
    showMatchesWithInliers(
        "Affine checked matches Closest",
        img1,
        keypoints1,
        img2,
        keypoints2,
        goodMatchesClosest,
        inlierMaskClosest);

    printf("\nRatio Test:\n");
    Mat inlierMaskRatio =evaluateMatchesWithAffine(keypoints1, keypoints2, goodMatchesRatio);
    showMatchesWithInliers(
        "Affine checked matches Ratio",
        img1,
        keypoints1,
        img2,
        keypoints2,
        goodMatchesRatio,
        inlierMaskRatio);


    //------------------------------------------------------------------------------------  COMPARISONS
    // compare diff vs union implementations
    // auto onlyIn1 = differenceMatches(goodMatches1, goodMatchesClosest);
    // auto onlyInClosest = differenceMatches(goodMatchesClosest, goodMatches1);
    // showMatches("{DIFF} \\ {UNION}", img1, keypoints1, img2, keypoints2, onlyIn1);
    // showMatches("{UNION} \\ {DIFF}", img1, keypoints1, img2, keypoints2, onlyInClosest);


    //compare closest neighb vs ratio test implementations
    // auto onlyInRatio= differenceMatches(goodMatchesRatio, goodMatchesClosest);
    // auto onlyInClosest2= differenceMatches(goodMatchesClosest, goodMatchesRatio);
    // showMatches("{Ratio Test} \\ {Closest neighb}",img1, keypoints1, img2, keypoints2, onlyInRatio);
    // showMatches("{Closest neighb} \\ {Ratio Test}",img1, keypoints1, img2, keypoints2, onlyInClosest2);


    waitKey(0);
    return 0;
}