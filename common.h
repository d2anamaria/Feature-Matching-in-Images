#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

#define PI 3.1415926535897932385
#define POS_INFINITY 1e30
#define NEG_INFINITY -1e30
#define max_(x,y) ((x) > (y) ? (x) : (y))
#define min_(x,y) ((x) < (y) ? (x) : (y))
#define isNan(x) ((x) != (x) ? 1 : 0)

#ifdef _WIN32
#include <windows.h>

class FileGetter{
    WIN32_FIND_DATAA found;
    HANDLE hfind;
    char folder[MAX_PATH];
    int chk;
    bool first;
    bool hasFiles;
public:
    FileGetter(char* folderin,char* ext);
    int getNextFile(char* fname);
    int getNextAbsFile(char* fname);
    char* getFoundFileName();
};

#else
#include <string>
#include <vector>

#ifndef MAX_PATH
#define MAX_PATH 4096
#endif

class FileGetter{
    std::string folderStr;
    std::vector<std::string> files;
    std::string foundName;
    size_t idx;
public:
    FileGetter(char* folderin,char* ext);
    int getNextFile(char* fname);
    int getNextAbsFile(char* fname);
    char* getFoundFileName();
};
#endif

int openFileDlg(char* fname);
int openFolderDlg(char* folderName);

void resizeImg(Mat src, Mat &dst, int maxSize, bool interpolate);