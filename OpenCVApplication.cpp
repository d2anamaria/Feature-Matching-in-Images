// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#ifndef _WIN32
#include <unistd.h>
#include <fstream>
#include <queue>
#include <random>
#include <unordered_map>


#define _wchdir chdir
#define _wgetcwd getcwd

#ifndef MAX_PATH
#define MAX_PATH 4096
#endif
#endif

#include "common.h"
#include <opencv2/core/utils/logger.hpp>


using namespace std;

const char* projectPath;



int main() 
{
	cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);


	return 0;
}