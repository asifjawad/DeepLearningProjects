// Project'4 Computer Vision 2022,
//Professor P.Zanuttigh,
// Image Stiching
//No Extra File
#include "opencv2/opencv.hpp"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


Mat Affine_Rotation(Mat straight, double orientation)   
{
    Mat rotate_img;      
    Point2f pt(straight.cols / 2, straight.rows / 2);            
    Mat r = getRotationMatrix2D(pt, orientation, 1.0);      
    warpAffine(straight, rotate_img, r, Size(straight.cols, straight.rows));  
    return rotate_img; 
}





Mat sift_des_stiching(Mat first_img, Mat sec_img)

{
    Ptr<cv::SIFT> detector = cv::SIFT::create(0, 3, 0.03, 10, 5);

    Ptr<cv::DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);

    vector<cv::KeyPoint> keypoint_1, keypoint_2;
    Mat Des_1, Des_2;

    vector<cv::DMatch > matches, resultmatches;
    int Direction = 0;

    detector->detectAndCompute(first_img, noArray(), keypoint_1, Des_1);
    detector->detectAndCompute(sec_img, noArray(), keypoint_2, Des_2);


    matcher->match(Des_1, Des_2, matches);

    vector<cv::Point2d> good_point1, good_point2;
    good_point1.reserve(matches.size());
    good_point2.reserve(matches.size());



    double max_dist = 0; double min_dist = 100;
    for (const auto& m : matches)
    {
        double dist = m.distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    for (const auto& m : matches)
    {
        if (m.distance <= 3 * min_dist)
        {
            good_point1.push_back(keypoint_1.at(m.queryIdx).pt);
            good_point2.push_back(keypoint_2.at(m.trainIdx).pt);
        }
    }

    Rect croppImg1(0, 0, first_img.cols, first_img.rows);
    Rect croppImg2(0, 0, sec_img.cols, sec_img.rows);

    int imgWidth = first_img.cols;
    for (int i = 0; i < good_point1.size(); ++i)
    {
        if (good_point1[i].x < imgWidth)
        {
            croppImg1.width = good_point1.at(i).x;
            croppImg2.x = good_point2[i].x;
            croppImg2.width = sec_img.cols - croppImg2.x;
            Direction = good_point1[i].y - good_point2[i].y;
            imgWidth = good_point1[i].x;
        }
    }
    first_img = first_img(croppImg1);
    sec_img = sec_img(croppImg2);


    int maxHeight = first_img.cols + sec_img.cols;
    int maxWidth = first_img.cols + sec_img.cols;
    cv::Mat result = cv::Mat::zeros(cv::Size(maxWidth, maxHeight + abs(Direction)), CV_8UC3);
    if (Direction > 0)
    {
        Mat half1(result, cv::Rect(0, 0, first_img.cols, first_img.rows));
        first_img.copyTo(half1);
        Mat half2(result, cv::Rect(first_img.cols, abs(Direction), sec_img.cols, sec_img.rows));
        sec_img.copyTo(half2);
    }
    else
    {
        Mat half1(result, cv::Rect(0, abs(Direction), first_img.cols, first_img.rows));
        first_img.copyTo(half1);
        Mat half2(result, cv::Rect(first_img.cols, 0, sec_img.cols, sec_img.rows));
        sec_img.copyTo(half2);
    }

    Mat img_matches;
    drawMatches(first_img, keypoint_1, sec_img, keypoint_2, matches, img_matches);
   
    imshow("Image Stiched ", result);
    imshow("Matches  ", img_matches);

    waitKey(0);
    return(result);
}





int main()
{
   
    string img_1 = "./T1/padova_patch00.jpg";
    string img_2 = "./T1/padova_patch01.jpg";
    string img_3 = "./T1/padova_patch02.jpg";

    string img_4 = "./T1/padova_patch10.jpg";
    string img_5 = "./T1/padova_patch11.jpg";
    string img_6 = "./T1/padova_patch12.jpg";

    string img_7 = "./T1/padova_patch20.jpg";
    string img_8 = "./T1/padova_patch21.jpg";
    string img_9 = "./T1/padova_patch22.jpg";
   
    /*
    string img_1 = "./T2/portello_patch00.jpg";
    string img_2 = "./T2/portello_patch01.jpg";
    string img_3 = "./T2/portello_patch02.jpg";

    string img_4 = "./T2/portello_patch10.jpg";
    string img_5 = "./T2/portello_patch11.jpg";
    string img_6 = "./T2/portello_patch12.jpg";

    string img_7 = "./T2/portello_patch20.jpg";
    string img_8 = "./T2/portello_patch21.jpg";
    string img_9 = "./T2/portello_patch22.jpg";

     string img_1 = "./RT1/padova_patch00.jpg";
    string img_2 = "./RT1/padova_patch01.jpg";
    string img_3 = "./RT1/padova_patch02.jpg";

    string img_4 = "./RT1/padova_patch10.jpg";
    string img_5 = "./RT1/padova_patch11.jpg";
    string img_6 = "./RT1/padova_patch12.jpg";

    string img_7 = "./RT1/padova_patch20.jpg";
    string img_8 = "./RT1/padova_patch21.jpg";
    string img_9 = "./RT1/padova_patch22.jpg";

     */


    Mat img1 = imread(img_1);
    Mat img2 = imread(img_2);
    Mat img3 = imread(img_3);
    Mat img4 = imread(img_4);
    Mat img5 = imread(img_5);
    Mat img6 = imread(img_6);
    Mat img7 = imread(img_7);
    Mat img8 = imread(img_8);
    Mat img9 = imread(img_9);



    Mat img1_2 = sift_des_stiching(img1, img2);
    Mat img123 = sift_des_stiching(img1_2, img3);

    Mat img4_5 = sift_des_stiching(img4, img5);
    Mat img456 = sift_des_stiching(img4_5, img6);

    Mat img7_8 = sift_des_stiching(img7, img8);
    Mat img789 = sift_des_stiching(img7_8, img9);

   
    
    
    Mat Aff_123 = Affine_Rotation(img123, 90);
    Mat Aff_456 = Affine_Rotation(img456, 90);
    Mat Aff_789 = Affine_Rotation(img789, 90);



    Mat two_img_sti = sift_des_stiching(Aff_123, Aff_456);
    Mat Entire_img = sift_des_stiching(two_img_sti, Aff_789);
    Mat IMAGE = Affine_Rotation(Entire_img, -90);


    IMAGE.resize(800, 800);
    imshow("Final Result", IMAGE);
      
    
    
    waitKey(0);
}

