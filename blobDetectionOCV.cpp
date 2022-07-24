//////////////////////////////////////////////////////////////////////////////
// Creates C++ MEX-file for OpenCV ORB feature detector. 
// ORB stands for Oriented FAST and Rotated BRIEF. It is basically a fusion of 
// FAST keypoint detector and BRIEF descriptor. 
//
// Copyright 2014-2016 The MathWorks, Inc.
//////////////////////////////////////////////////////////////////////////////

#include "opencvmex.hpp"
using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////////////////////
// Check inputs
//////////////////////////////////////////////////////////////////////////////
void checkInputs(int nrhs, const mxArray *prhs[])
{
//     if (nrhs != 2)
//     {
//         mexErrMsgTxt("Incorrect number of inputs. Function expects 2 inputs.");
//     }
//     
//     if (!mxIsUint8(prhs[0]))
//     {       
//         mexErrMsgTxt("Input image must be uint8.");
//     }
}

///////////////////////////////////////////////////////////////////////////
// Main entry point to a MEX function
///////////////////////////////////////////////////////////////////////////
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{  
    
    checkInputs(nrhs, prhs);
    
    // inputs
    cv::Ptr<cv::Mat> img = ocvMxArrayToImage_uint8(prhs[0], true);
  
    // Set up the detector with default parameters.
    SimpleBlobDetector::Params params;
    //    Change thresholds
//     params.minThreshold = 10;
//     params.maxThreshold = 200;
//     params.blobColor = 0;
//     params.minDistBetweenBlobs = 0;
    params.filterByColor = false;
    params.filterByArea = true;
    params.minArea = 50;
    params.filterByCircularity = false;
//     params.minCircularity = 0.1;
    params.filterByConvexity = false;
//     params.minConvexity = 0.87;
    params.filterByInertia = false;
//     params.minInertiaRatio = 0.01;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

    // prepare space for returned keypoints
    std::vector<KeyPoint> keypoints; 
    
    // invoke the detector
    detector->detect(*img, keypoints); 
        
    // populate the outputs
    plhs[0] = ocvKeyPointsToStruct(keypoints);  
}

