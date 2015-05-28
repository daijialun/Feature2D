#include <stdio.h>
#include <iostream>
#include <core.hpp>
#include <features2d/features2d.hpp>
#include <xfeatures2d.hpp>
#include <highgui.hpp>
#include <cv.hpp>

using namespace cv;
using namespace cv::xfeatures2d;

void readme();

/** @function main */
int main( int argc, char** argv )
{
  if( argc != 3 )
  { readme(); return -1; }

  Mat img_object = imread( argv[1],0);
  Mat img_scene = imread( argv[2], 0);

//  cvtColor(img_object,img_object,CV_RGB2GRAY);
  //cvtColor(img_scene,img_scene,CV_RGB2GRAY);

  if( !img_object.data || !img_scene.data )
  { std::cout<< " --(!) Error reading images " << std::endl; return -1; }

  //-- Step 1: Detect the keypoints using SURF Detector
  int minHessian =370;
  //Hessian矩阵, 多元函数的二阶偏导数构成的方阵

  //SurfFeatureDetector detector( minHessian );
  Ptr<SURF> detector = SURF::create( minHessian ); //SURF is Speed Up Robust SIFT, minHessian  means Threshold for hessian keypoint detector used in SURF
 //SURF是类，成员类型有extended, upright, hessianThreshold, n0ctaves, n0ctaveLayers
//成员函数有SURF::SURF(), operator()
//Ptr是类模板

  std::vector<KeyPoint> keypoints_object, keypoints_scene;
  //KeyPoint是类，成员变量有pt，size,angle,response等
  //成员函数有KeyPoint::KeyPoint,conver,overlap

  detector -> detect( img_object, keypoints_object );
  detector -> detect( img_scene, keypoints_scene );

  //-- Step 2: Calculate descriptors (feature vectors)
// SurfDescriptorExtractor extractor;
 Ptr<SURF> extractor = SURF::create( );

  Mat descriptors_object, descriptors_scene;

  extractor->compute( img_object, keypoints_object, descriptors_object );
  extractor->compute( img_scene, keypoints_scene, descriptors_scene );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  FlannBasedMatcher matcher;
  std::vector< DMatch > matches;
  matcher.match( descriptors_object, descriptors_scene, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors_object.rows; i++ )
  { double dist = matches[i].distance;   //两个特征向量之间的欧氏距离，越小表明匹配度越高
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;

  for( int i = 0; i < descriptors_object.rows; i++ )
  { if( matches[i].distance < 2.5*min_dist )
     { good_matches.push_back( matches[i]); }
  }
  //该方法将一个或多个元素添加到矩阵的底部。元素为Mat时，其类型和列的数目必须和矩阵容器是相同的。

  Mat img_matches;
  drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
               std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

imshow("match",img_matches);
waitKey(0);

  //-- Localize the object
  std::vector<Point2f> obj;      //二维点坐标 typedef Point_<float> Point2f;
  std::vector<Point2f> scene;

  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypoints_object[ good_matches[i].queryIdx ].pt );	//queryIdx匹配对应的查询图像的特征描述子索引
    scene.push_back( keypoints_scene[ good_matches[i].trainIdx ].pt );
  }

  Mat H = findHomography( obj, scene, RANSAC );

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> obj_corners(4);
  obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_object.cols, 0 );
  obj_corners[2] = cvPoint( img_object.cols, img_object.rows ); obj_corners[3] = cvPoint( 0, img_object.rows );
  std::vector<Point2f> scene_corners(4);

  perspectiveTransform( obj_corners, scene_corners, H);

  //-- Draw lines between the corners (the mapped object in the scene - image_2 )
  line( img_matches, scene_corners[0] + Point2f( img_object.cols, 0), scene_corners[1] + Point2f( img_object.cols, 0), Scalar(0, 255, 0), 4 );
  line( img_matches, scene_corners[1] + Point2f( img_object.cols, 0), scene_corners[2] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[2] + Point2f( img_object.cols, 0), scene_corners[3] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );
  line( img_matches, scene_corners[3] + Point2f( img_object.cols, 0), scene_corners[0] + Point2f( img_object.cols, 0), Scalar( 0, 255, 0), 4 );

  //-- Show detected matches
  imshow( "Good Matches & Object detection", img_matches );

  waitKey(0);
  return 0;
  }

  /** @function readme */
  void readme()
  { std::cout << " Usage: ./SURF_descriptor <img1> <img2>" << std::endl; }
