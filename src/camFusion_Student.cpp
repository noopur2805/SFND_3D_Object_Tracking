
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // calculate average euclidean distance between current and previous frames keypoints
    double avgEuclildeanDist;
    std::vector<double> distances;

    for (auto kptMatch: kptMatches)
    {
        cv::KeyPoint prevPoint = kptsPrev[kptMatch.queryIdx];
        cv::KeyPoint currPoint = kptsCurr[kptMatch.trainIdx];

        if (boundingBox.roi.contains(currPoint.pt))
            distances.push_back(cv::norm(currPoint.pt - prevPoint.pt));
    }
    avgEuclildeanDist = std::accumulate(distances.begin(), distances.end(), 0) / distances.size();

    // filter keypoints which are outside 1 standard deviation around mean
    double stdDev = 0.0;
    for (auto dist: distances)
        stdDev += pow(dist - avgEuclildeanDist, 2);
    stdDev = sqrt(stdDev / distances.size());

    for (int i = 0; i < kptMatches.size(); i++)
    {
        if (abs(distances[i] - avgEuclildeanDist) < stdDev)
            boundingBox.kptMatches.push_back(kptMatches[i]);
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }


    // take median distance ratio
    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence

    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
    
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // time difference between reading two frames
    double dT = 1.0 / frameRate;

    // to store median x values of lidar points in front of ego car
    std::vector<double> prevX;
    std::vector<double> currX;


    for (auto lidarPoint: lidarPointsPrev)
        prevX.push_back(lidarPoint.x);

    for (auto lidarPoint: lidarPointsCurr)
        currX.push_back(lidarPoint.x);

    sort(prevX.begin(), prevX.end());
    sort(currX.begin(), currX.end());
    
    double medianPrevX, medianCurrX;
    medianPrevX = prevX.size() % 2 == 0 ? (prevX[prevX.size()/2 - 1] + prevX[prevX.size()/2]) / 2.0 : prevX[prevX.size()/2];
    medianCurrX = currX.size() % 2 == 0 ? (currX[currX.size()/2 - 1] + currX[currX.size()/2]) / 2.0 : currX[currX.size()/2];
    
    // compute TTC on median X distances
    TTC = medianCurrX * dT / (medianPrevX - medianCurrX);

}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // get previous and current frame size
    int prevBoxSize = prevFrame.boundingBoxes.size(); 
    int currBoxSize = currFrame.boundingBoxes.size();

    // variable to store matching previous and current bounding boxes with same keypoints enclosed
    int matchCount[prevBoxSize][currBoxSize] = {0};
    
    for (auto match: matches)
    {
        cv::KeyPoint prevKeypoint = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKeypoint = currFrame.keypoints[match.trainIdx];

        cv::Point prevPoint = cv::Point(prevKeypoint.pt.x, prevKeypoint.pt.y);
        cv::Point currPoint = cv::Point(currKeypoint.pt.x, currKeypoint.pt.y);

        // store previous and current frames bounding box IDs
        std::vector <int> prevBoundingBoxIDs, currBoundingBoxIDs;
        for (int i = 0; i < prevBoxSize; ++i)
        {
            if (prevFrame.boundingBoxes[i].roi.contains(prevPoint))
            {
                prevBoundingBoxIDs.push_back(i);
            }
        }        
        for (int j = 0; j < currBoxSize; ++j)
        {
            if (currFrame.boundingBoxes[j].roi.contains(currPoint))
            {
                currBoundingBoxIDs.push_back(j);
            }
        }

        // update matchCount matrix with number of matched keypoints found in corresponding bounding boxes
        for (int i: prevBoundingBoxIDs)
        {
            for (int j: currBoundingBoxIDs)
            {
                matchCount[i][j] += 1;
            }
        }
    }
    
    // for each box in previous frame, find the box with highest number of keypolint correspondences in current frame
    for (int i = 0; i < prevBoxSize; i++)
    {
        int maxMatchCount = 0;
        int bestMatchIdx = 0;
        for (int j = 0; j < currBoxSize; j++)
        {
            if (matchCount[i][j] > maxMatchCount)
            {
                maxMatchCount = matchCount[i][j];
                bestMatchIdx = j;
            }
        }
        bbBestMatches[i] = bestMatchIdx; 
    } 

}
