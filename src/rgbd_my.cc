/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
// Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
//OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/core/eigen.hpp>
//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>

#include <System.h>
#include <boost/format.hpp>

using namespace std;

// 类型定义
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// 相机内参结构
struct CAMERA_INTRINSIC_PARAMETERS
{
    double cx, cy, fx, fy, scale;
};

// 定义生成点云函数
PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera );
//check keyframe
bool IsKeyFrame(cv::Mat& LastCamerapose,cv::Mat& CurrentCamerapose,double& max_norm, double& min_norm);


int main(int argc, char **argv)
{
    if(argc != 6)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence start_index end_index" << endl;
        return 1;
    }

      //Check settings file
        cv::FileStorage fsSettings("./config/kinect1.yaml", cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
           cerr << "Failed to open settings file at: " << argv[2]<< endl;
           exit(-1);
        }
    CAMERA_INTRINSIC_PARAMETERS camera;
    camera.cx = fsSettings["Camera.cx"];
    camera.cy = fsSettings["Camera.cy"];
    camera.fx = fsSettings["Camera.fx"];
    camera.fy = fsSettings["Camera.fy"];
    camera.scale = fsSettings["DepthMapFactor"];

    int     start_index = atoi( argv[4] );
    int     end_index = atoi( argv[5] );
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);

    // Main loop
    cv::Mat imRGB, imD;
    cv::Mat CameraPoseMatrix4_4(4,4,CV_32F);
    // cout<<CameraPoseMatrix4_4.type()<<endl;
    // 初始化点云
    PointCloud::Ptr output ( new PointCloud() ); //全局地图
    PointCloud::Ptr tmp ( new PointCloud() );
    PointCloud::Ptr newCloud ( new PointCloud() );
   pcl::visualization::CloudViewer viewer("viewer");
   // 点云滤波设置
   pcl::VoxelGrid<PointT> voxel; // 网格滤波器，调整地图分辨率
   double gridsize = 0.02;
   voxel.setLeafSize( gridsize, gridsize, gridsize );

   pcl::PassThrough<PointT> pass; // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
   pass.setFilterFieldName("z");
   pass.setFilterLimits( 0.0, 5.0 ); //4m以上就不要

  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  double max_norm = fsSettings["max_norm"];
  double min_norm = fsSettings["min_norm"];
  vector<cv::Mat> keyframe;
  double t,t1,LocationCost,MappingCost,TotalTime = 0;
  int keyframe_num=0;
    for ( int index = start_index; index < end_index; index ++ )
    {
        boost::format fmt ("./%s/rgb_png/%d.png");
        imRGB = cv::imread( (fmt%argv[3]%index).str(), CV_LOAD_IMAGE_UNCHANGED);
        if(imRGB.empty())
          {
            cerr << endl << "can't read the rgb image, please check the dir" << endl;
            cerr<<"may be the input dir :"<<(fmt%argv[3]%index).str() <<" is wrong"<<endl;
            return 1;
          }

        fmt = boost::format("./%s/depth_png/%d.png");
        imD = cv::imread( (fmt%argv[3]%index).str(), CV_LOAD_IMAGE_UNCHANGED);
        if(imD.empty())
          {
            cerr << endl << "can't read the depth image, please check the dir" << endl;
            cerr<<"may be the input dir :"<<(fmt%argv[3]%index).str() <<" is wrong"<<endl;
            return 1;
          }

        t = (double)cvGetTickCount();

        CameraPoseMatrix4_4 = SLAM.TrackRGBD( imRGB, imD, index  );

        LocationCost =  ((double)cvGetTickCount() - t)/((double)cvGetTickFrequency()*1000.);
        cout<<"Location cost time: "<<LocationCost<<"ms"<<endl;

         t1 = (double)cvGetTickCount();
         if(!CameraPoseMatrix4_4.empty())
       {
        if(index == start_index)
          keyframe.push_back(CameraPoseMatrix4_4);
        else
          {
            if( IsKeyFrame(keyframe.back(),CameraPoseMatrix4_4,max_norm, min_norm) )
              keyframe.push_back(CameraPoseMatrix4_4);
            else
              continue;
          }
        keyframe_num++;
       //生成点云并滤波
        newCloud = image2PointCloud(imRGB,imD,camera);
        voxel.setInputCloud( newCloud );
        voxel.filter( *tmp );
        pass.setInputCloud( tmp );
        pass.filter( *newCloud );
        //cout<<CameraPoseMatrix4_4.type()<<endl;

        // 把点云变换后加入全局地图中
        for(int i=0; i<CameraPoseMatrix4_4.rows; i++)
          {
           for(int j=0; j<CameraPoseMatrix4_4.cols; j++)
             {
                    T(i,j)=CameraPoseMatrix4_4.at<float>(i,j);
                     //cout<<CameraPoseMatrix4_4.at<float>(i,j)<<endl;//数据ok
             }
          }
            pcl::transformPointCloud( *newCloud,*tmp, T.inverse().matrix());
             
            *output += *tmp;
           tmp->clear();
           newCloud->clear();
           viewer.showCloud( output);

           }
         MappingCost =  ((double)cvGetTickCount() - t)/ ((double)cvGetTickFrequency()*1000.);
         cout<<"Mapping cost time: "<<MappingCost <<"ms"<<endl;

         TotalTime = LocationCost +MappingCost +TotalTime;
    }
     cout<<"Average FPS is: "<<(end_index-start_index+1)/(TotalTime/1000)<<endl;
     cout<<"the number of keyframe is :"<<keyframe_num<<endl;
     pcl::io::savePCDFile( "./data/result.pcd", *output );
      cout<<"Final map is saved."<<endl;
    // Stop all threads
    SLAM.Shutdown();
    return 0;
}

PointCloud::Ptr image2PointCloud( cv::Mat& rgb, cv::Mat& depth, CAMERA_INTRINSIC_PARAMETERS& camera )
{
    PointCloud::Ptr cloud ( new PointCloud );

    for (int m = 0; m < depth.rows; m++)
        for (int n=0; n < depth.cols; n++)
        {
            // 获取深度图中(m,n)处的值
            ushort d = depth.ptr<ushort>(m)[n];
            // d 可能没有值，若如此，跳过此点
            if (d == 0)
                continue;
            // d 存在值，则向点云增加一个点
            PointT p;

            // 计算这个点的空间坐标
            p.z = double(d) / camera.scale;
            p.x = (n - camera.cx) * p.z / camera.fx;
            p.y = (m - camera.cy) * p.z / camera.fy;

            // 从rgb图像中获取它的颜色
            // rgb是三通道的BGR格式图，所以按下面的顺序获取颜色
            p.b = rgb.ptr<uchar>(m)[n*3];
            p.g = rgb.ptr<uchar>(m)[n*3+1];
            p.r = rgb.ptr<uchar>(m)[n*3+2];

            // 把p加入到点云中
            cloud->points.push_back( p );
        }
    // 设置并保存点云
    cloud->height = 1;
    cloud->width = cloud->points.size();
    cloud->is_dense = false;

    return cloud;
}
bool IsKeyFrame(cv::Mat& LastCamerapose,cv::Mat& CurrentCamerapose,double& max_norm,double& min_norm)
{
  cv::Mat rvec, tvec;
  cv::Mat Tc1c2 = LastCamerapose * CurrentCamerapose.inv();
  cout<<LastCamerapose<<endl;
  cout<<CurrentCamerapose<<endl;
  cout<<CurrentCamerapose.inv()<<endl;
  cout<<Tc1c2<<endl;

  cv::Mat Tc1c2_3_3=Tc1c2(cv::Rect(0,0,3,3));
  cv::Rodrigues(Tc1c2_3_3,rvec);
  cout<<"rvec="<<rvec<<endl;
   tvec = Tc1c2(cv::Rect(3,0,1,3));
   cout<<"tvec="<<tvec<<endl;
  double change = fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
  cout<<"norm"<<change<<endl;
  if (change >= max_norm || change<=min_norm)
 // if (change<=min_norm)
    return false;
  else
    return true;
}

