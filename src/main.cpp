#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <string>
#include <algorithm>    // find
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <random>
#include <memory>

// OpenCV:
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp> // Viz & VTK visualization
#include <opencv2/viz/types.hpp>
#include <opencv2/calib3d.hpp>


// // Open3D:
// #include "open3d/Open3D.h"
// #include <pcl/point_types.h>
// #include <pcl/visualization/pcl_visualizer.h>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Geometry>

struct Options{
    // SIFT params
    int num_features = 1000;          // number of best features to retain
    int num_octave_layers = 3;        // number of layers in each octave
    double contrast_threshold = 0.04; // threshold used to filter out weak features in low-contrast regions; the larger, the less features
    double edge_threshold = 10;       // threshold used to filter out edge-like features; the larger, the more features
    double sigma = 1.6;               // sigma of the Gaussian applied to the input image; reduce for weak soft images
};

Options options;


// ----------------------------------------------------------------------------
int main() {

// camera interinsic params    
cv::Mat intrinsic_mat = cv::Mat(3,3, CV_64F);  // camera intrinsic matrix (aka calibration matrix)
cv::Mat distortion_coeff = cv::Mat::zeros(5,1, CV_64F); // camera distortion coefficients
intrinsic_mat.at<double>(0,0) = 359.035;
intrinsic_mat.at<double>(1,1) = 260.546;
intrinsic_mat.at<double>(0,2) = 352.779;
intrinsic_mat.at<double>(1,2) = 354.609;

// ----------------------------------------
// load images
int num_images = 3;

const std::string image_dir = "../data/"; // image paths
std::vector<std::string> image_names;
std::vector<cv::Mat> images;
for (int i = 0; i <3; ++i) {       
    std::string image_name = "img" + std::to_string(i) + ".png";
    image_names.emplace_back(image_name);

    std::string image_path = image_dir + image_names[i]; 
    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    images.emplace_back(image);

    // cv::imshow(image_name, image);
    // cv::waitKey(0);
}

// ----------------------------------------
// extract sift features

// Init sift things
auto detector = cv::SiftFeatureDetector::create(options.num_features,
                                                options.num_octave_layers,
                                                options.contrast_threshold,
                                                options.edge_threshold,
                                                options.sigma);
auto extractor = cv::SiftDescriptorExtractor::create();

// Create data structure to hold features
std::vector<cv::KeyPoint> keypoints[num_images];  // exteracted image features
cv::Mat descriptors[num_images];  // feature descriptors
cv::Mat image_keypoints[num_images];

// Detect features and compute descriptors
for (int i = 0; i < num_images; i++)
{
    detector->detect(images[i], keypoints[i]);
    extractor->compute(images[i], keypoints[i], descriptors[i]);

    // Debugging
    // cv::Mat image_keypoints;
    // cv::drawKeypoints(images[i], keypoints[i], image_keypoints);
    // cv::imshow("Image with SIFT Keypoints", image_keypoints);
    // cv::waitKey(0);
}


// ----------------------------------------
// find relative pose between images using ransac

// init BF matcher
cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_L2);
std::vector<std::vector<cv::DMatch>> matches;

// Find matches 
for (int i = 0; i < num_images - 1; i++) {
    for (int j = i + 1; j < num_images; j++) {
        std::vector<cv::DMatch> match;
        matcher->match(descriptors[i], descriptors[j], match);
        matches.push_back(match);
        
        // Debugging
        // cv::Mat img_matches;
        // cv::drawMatches(images[i], keypoints[i], images[j], keypoints[j], match, img_matches);
        // cv::imshow("Matches between Images", img_matches);
        // cv::waitKey(0);
    }
}

// put the matches into some vectors of points
std::vector<cv::Point2f> points1, points2;
for (const auto& match : matches[0]) {  
    points1.push_back(keypoints[0][match.queryIdx].pt);
    points2.push_back(keypoints[1][match.trainIdx].pt);
}

cv::Mat E, R, t, mask, triangulated;

E = cv::findEssentialMat(points1, points2, intrinsic_mat, cv::RANSAC, 0.999, 1.0, mask);
cv::recoverPose(E, points1, points2, intrinsic_mat, R, t,1, mask,triangulated);

printf("%ld\n",points1.size());
printf("%d,%d\n",mask.size().height,mask.size().width);
printf("%d,%d\n",triangulated.size().height,triangulated.size().width);

std::vector<cv::Vec3f> points;
for (int i =0; i < mask.size().height; i++)
{
    // if (mask.at<double>(i,0) == 0.0)
    if (true)
    {
        printf("%f,%f,%f,%f\n",triangulated.at<double>(0,i)
                              ,triangulated.at<double>(1,i)
                              ,triangulated.at<double>(2,i)
                              ,triangulated.at<double>(3,i));
        double x = triangulated.at<double>(0,i);
        double y = triangulated.at<double>(1,i);
        double z = triangulated.at<double>(2,i);
        double w = triangulated.at<double>(3,i);

        cv::Vec3f point = cv::Vec3f(x,y,z);
        // cv::Vec3f point = cv::Vec3f(x/w,y/w,z/w);
        // cv::Vec3f point = cv::Vec3f(x*w,y*w,z*w);
        points.push_back(point);
    }
}

// ----------------------------------------
// triangulate

// Create projection matrices for the first and second camera
// cv::Mat proj1(3, 4, CV_64FC1, cv::Scalar(0));
// cv::Mat proj2(3, 4, CV_64FC1, cv::Scalar(0));
// R.copyTo(proj2(cv::Rect(0, 0, 3, 3)));
// t.copyTo(proj2(cv::Rect(3, 0, 1, 3)));
// intrinsic_mat.copyTo(proj1(cv::Rect(0, 0, 3, 3)));
// intrinsic_mat.copyTo(proj2(cv::Rect(0, 0, 3, 3)));

// cv::Mat points4D;
// cv::triangulatePoints(proj1, proj2, points1, points2, points4D);

// // Convert from homogeneous coordinates
// cv::Mat points3D;
// cv::convertPointsFromHomogeneous(points4D.t(), points3D);


// ----------------------------------------
// match scale of landmarks
// ???


// ----------------------------------------
// display 3D landmarks
printf("here\n");
cv::viz::Viz3d window("Point Cloud");
printf("here\n");
    cv::Mat pointCloudMat = cv::Mat(points); // Reshape and convert to Mat
printf("here\n");
    cv::viz::WCloud cloud(pointCloudMat);
printf("here\n");
    // cv::viz::WCloud cloud(triangulated);
    window.showWidget("coordinate", cv::viz::WCoordinateSystem(1));
printf("here\n");
    window.showWidget("Point cloud", cloud);
printf("here\n");
    window.spin();
printf("here\n");
}