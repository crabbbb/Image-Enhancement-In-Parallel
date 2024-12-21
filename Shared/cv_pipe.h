#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h> // For pipe
#include <fcntl.h>  // For O_WRONLY // Open for writing only

#define reset_getopt()    (optind = 0)

// Mat = Matrix
int cv_imshow(cv::Mat &image);
int init_cv_pipe_comm(int argc, char *argv[], bool verbose=false);
int finalize_cv_pipe_comm();