#include "common.h"

#include <cstring>
#include <cstdio>

#ifdef _WIN32
  #include "stdafx.h"
  #include <windows.h>
  #include <CommDlg.h>
  #include <ShlObj.h>

  FileGetter::FileGetter(char* folderin, char* ext) {
      strcpy(folder, folderin);
      char folderstar[MAX_PATH];
      if (!ext) strcpy(ext, (char*)"*");
      sprintf(folderstar, "%s\\*.%s", folder, ext);
      hfind = FindFirstFileA(folderstar, &found);
      hasFiles = !(hfind == INVALID_HANDLE_VALUE);
      first = true;
  }

  int FileGetter::getNextFile(char* fname) {
      if (!hasFiles) return 0;

      if (first) {
          strcpy(fname, found.cFileName);
          first = false;
          return 1;
      } else {
          chk = FindNextFileA(hfind, &found);
          if (chk) strcpy(fname, found.cFileName);
          return chk;
      }
  }

  int FileGetter::getNextAbsFile(char* fname) {
      if (!hasFiles) return 0;

      if (first) {
          sprintf(fname, "%s\\%s", folder, found.cFileName);
          first = false;
          return 1;
      } else {
          chk = FindNextFileA(hfind, &found);
          if (chk) sprintf(fname, "%s\\%s", folder, found.cFileName);
          return chk;
      }
  }

  char* FileGetter::getFoundFileName() {
      if (!hasFiles) return 0;
      return found.cFileName;
  }

  int openFileDlg(char* fname) {
      char *filter = (char*)"All Files (*.*)\0*.*\0";
      HWND owner = NULL;
      OPENFILENAME ofn;
      char fileName[MAX_PATH];
      strcpy(fileName, "");
      ZeroMemory(&ofn, sizeof(ofn));
      ofn.lStructSize = sizeof(OPENFILENAME);
      ofn.hwndOwner = owner;
      ofn.lpstrFilter = filter;
      ofn.lpstrFile = fileName;
      ofn.nMaxFile = MAX_PATH;
      ofn.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;
      ofn.lpstrDefExt = "";
      GetOpenFileName(&ofn);
      strcpy(fname, ofn.lpstrFile);
      return strcmp(fname, "");
  }

  int openFolderDlg(char* folderName) {
      BROWSEINFO bi;
      ZeroMemory(&bi, sizeof(bi));
      SHGetPathFromIDList(SHBrowseForFolder(&bi), folderName);
      return strcmp(folderName, "");
  }

#else
  // -------- macOS / Linux --------
  #include <filesystem>
  #include <string>
  #include <vector>

  namespace fs = std::filesystem;

  // We store state inside the object using "pimpl-like" static maps would be overkill.
  // Instead, we implement a simple vector-based iterator.
  // NOTE: This requires FileGetter in common.h to NOT contain Windows-only fields on non-Windows.
  // (Use the cross-platform common.h guard version you already have.)

  FileGetter::FileGetter(char* folderin, char* ext) {
      folderStr = folderin ? std::string(folderin) : std::string(".");
      idx = 0;
      files.clear();
      foundName.clear();

      std::string e = ext ? std::string(ext) : std::string("");
      // Accept "bmp", ".bmp", "*.bmp"
      if (!e.empty() && e.rfind("*.", 0) == 0) e = e.substr(1);   // "*.bmp" -> ".bmp"
      if (!e.empty() && e[0] != '.') e = "." + e;                 // "bmp" -> ".bmp"

      try {
          for (auto& p : fs::directory_iterator(folderStr)) {
              if (!p.is_regular_file()) continue;
              if (e.empty() || p.path().extension().string() == e) {
                  files.push_back(p.path().filename().string());
              }
          }
      } catch (...) {
          // folder unreadable -> keep empty
      }
  }

  int FileGetter::getNextFile(char* fname) {
      if (idx >= files.size()) return 0;
      foundName = files[idx++];
      if (fname) std::strcpy(fname, foundName.c_str());
      return 1;
  }

  int FileGetter::getNextAbsFile(char* fname) {
      if (idx >= files.size()) return 0;
      foundName = files[idx++];
      std::string abs = (fs::path(folderStr) / foundName).string();
      if (fname) std::strcpy(fname, abs.c_str());
      return 1;
  }

  char* FileGetter::getFoundFileName() {
      return foundName.empty() ? nullptr : foundName.data();
  }

  // "Dialog" replacement: prompt in terminal (minimal + works in CLion)
  int openFileDlg(char* fname) {
      if (!fname) return 0;
      std::printf("Enter file path (or empty to cancel): ");
      std::fflush(stdout);

      // read a whole line (spaces allowed)
      char buf[4096];
      if (!std::fgets(buf, sizeof(buf), stdin)) return 0;

      // strip newline
      size_t n = std::strlen(buf);
      while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';

      if (n == 0) return 0;
      std::strcpy(fname, buf);
      return 1;
  }

  int openFolderDlg(char* folderName) {
      if (!folderName) return 0;
      std::printf("Enter folder path (or empty to cancel): ");
      std::fflush(stdout);

      char buf[4096];
      if (!std::fgets(buf, sizeof(buf), stdin)) return 0;

      size_t n = std::strlen(buf);
      while (n > 0 && (buf[n-1] == '\n' || buf[n-1] == '\r')) buf[--n] = '\0';

      if (n == 0) return 0;
      std::strcpy(folderName, buf);
      return 1;
  }

#endif

void resizeImg(Mat src, Mat &dst, int maxSize, bool interpolate)
{
    double ratio = 1;
    double w = src.cols;
    double h = src.rows;
    if (w > h) ratio = w / (double)maxSize;
    else       ratio = h / (double)maxSize;

    int nw = (int)(w / ratio);
    int nh = (int)(h / ratio);

    Size sz(nw, nh);
    if (interpolate) resize(src, dst, sz);
    else            resize(src, dst, sz, 0, 0, INTER_NEAREST);
}