#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include <stdlib.h>
#include <string>
#include <stdio.h>

#include <linux/input.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
//【1】
#include <XnCppWrapper.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

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
using namespace cv;

bool run = true ;
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
/*******非阻塞键盘输入*******/
/*static int get_char()
{
  fd_set rfds;
     struct timeval tv;
     int ch = 0;

     FD_ZERO(&rfds);
     FD_SET(0, &rfds);
     tv.tv_sec = 0;
     tv.tv_usec = 5000; //设置等待超时时间微秒。。

     //检测键盘是否有输入
     if (select(1, &rfds, NULL, NULL, &tv) > 0)
     {
         ch = getchar();
     }

     return ch;
}*/

void CheckOpenNIError( XnStatus result, string status )
{ 
	if( result != XN_STATUS_OK ) 
		cerr << status << " Error: " << xnGetStatusString( result ) << endl;
}

int main( int argc, char** argv )
{
	
 
  XnStatus result = XN_STATUS_OK;
	xn::DepthMetaData depthMD;
	xn::ImageMetaData imageMD;

	//OpenCV
	Mat depth(Size(640,480),CV_16UC1);
	Mat imgRGB8u(Size(640,480),CV_8UC3);
	Mat color(Size(640,480),CV_8UC3);

	//【2】
	// context 
	xn::Context context; 
	result = context.Init(); 
	CheckOpenNIError( result, "initialize context" );  

	// creategenerator  
	xn::DepthGenerator depthGenerator;  
	result = depthGenerator.Create( context ); 
	CheckOpenNIError( result, "Create depth generator" );  
	xn::ImageGenerator imageGenerator;

	result = imageGenerator.Create( context ); 
	CheckOpenNIError( result, "Create image generator" );

	//【3】
	//map mode  
	XnMapOutputMode mapMode; 
	mapMode.nXRes = 640;
	mapMode.nYRes = 480;
	mapMode.nFPS = 10;//低fds，避免过多的slam
	result = depthGenerator.SetMapOutputMode( mapMode ); 
	result = imageGenerator.SetMapOutputMode( mapMode );  

	//【4】
	// correct view port  
	depthGenerator.GetAlternativeViewPointCap().SetViewPoint( imageGenerator ); 

	//【5】
	//read data
	result = context.StartGeneratingAll();  
	//【6】
	result = context.WaitNoneUpdateAll();  
/*********** add **********/

  //Check settings file
  string orbVocFile = "/home/bobo/code/orbslam2_3D/config/ORBvoc.txt";
  string orbSetiingsFile = "/home/bobo/code/orbslam2_3D/config/kinect1.yaml";
  ORB_SLAM2::System orbslam( orbVocFile, orbSetiingsFile ,ORB_SLAM2::System::RGBD, true);

  cv::FileStorage fsSettings(orbSetiingsFile, cv::FileStorage::READ);
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
    double max_norm = fsSettings["max_norm"];
    double min_norm = fsSettings["min_norm"];
    int KF_per_submap = fsSettings["KF_per_submap"];//**key frames per submap
    double min_msploop=fsSettings["min_msploop"];//控制每次循环的最小用时

    cv::Mat  CameraPoseMatrix4_4(4,4,CV_32F);
     PointCloud::Ptr output ( new PointCloud() ); //全局地图
     PointCloud::Ptr tmp ( new PointCloud() );
     PointCloud::Ptr newCloud ( new PointCloud() );
     pcl::visualization::CloudViewer viewer("viewer");
     // 点云滤波设置
     pcl::VoxelGrid<PointT> voxel; // 网格滤波器，调整地图分辨率
     double gridsize = 0.01;
      voxel.setLeafSize( gridsize, gridsize, gridsize );

      pcl::PassThrough<PointT> pass; // z方向区间滤波器，由于rgbd相机的有效深度区间有限，把太远的去掉
      pass.setFilterFieldName("z");
      pass.setFilterLimits( 0.0, 4.0 ); //4m以上就不要

      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      vector<cv::Mat> keyframe;
      double t,t1,LocationCost,MappingCost,TotalTime = 0;
      double LocationTotalTime = 0;
      double frame_num = 0;
      int keyframe_num=0;
      int submap_num=0;
      int measure_point_th = 0;
      char Saverpcd[256],Savecampose[256];
      cv::Mat tvec,CameraPose;
      /**********非阻塞键盘输入**********/
      int keys_fd;
      struct input_event input;
      keys_fd=open("/dev/input/event4",O_RDONLY|O_NONBLOCK );//定义为非阻塞
      if(keys_fd<0)
        {
              printf("error\n");
              return -1;
         }
/**********************/

	//while( !(result = context.WaitNoneUpdateAll( ) )&& !(get_char()==10))
	//while( !(result = context.WaitNoneUpdateAll( ) ))
	while( !(result = context.WaitNoneUpdateAll( ) )&& !(input.code == KEY_ESC))
	{  
		//read the keyboad input
		read(keys_fd,&input,sizeof(struct input_event));
	 //'s':停，‘g’：启动
		if(input.code == KEY_S)
			{
					while(1)
						{
							read(keys_fd,&input,sizeof(struct input_event));
							if(input.code == KEY_G)
								break;
						}
			}
		//get meta data
		depthGenerator.GetMetaData(depthMD); 
		imageGenerator.GetMetaData(imageMD);

		//【7】
		//OpenCV output
		memcpy(depth.data,depthMD.Data(),640*480*2);
		memcpy(imgRGB8u.data,imageMD.Data(),640*480*3);
		cvtColor(imgRGB8u,color,CV_RGB2BGR);

	 /************ do something ***************/
		frame_num++;
		t = (double)cvGetTickCount();
//CameraPoseMatrix4_4 实为相机位姿矩阵的逆
		CameraPoseMatrix4_4 =  orbslam.TrackRGBD( color, depth, frame_num);
		LocationCost =  ((double)cvGetTickCount() - t)/((double)cvGetTickFrequency()*1000.);
		//cout<<"Location cost time: "<<LocationCost<<"ms"<<endl;
		LocationTotalTime = LocationCost +LocationTotalTime;
		/****calculate the translation distance*******
		CameraPose =CameraPoseMatrix4_4.inv();//求逆出错
		tvec = CameraPose(cv::Rect(3,0,1,3));
		cout<<"robot move"<<fabs(cv::norm(tvec))<<"meters"<<endl;
		cout<<"tvec="<<tvec<<endl;*/
	 //'s' 键，保存CameraPose,还有问题，每次会保存两个文件
		if(input.code == KEY_ENTER)
			{
				measure_point_th++;
				sprintf(Savecampose,"/home/bobo/code/orbslam2_3D/data/measure_point/%01d.txt",measure_point_th);
				ofstream fout(Savecampose);
				fout<<CameraPoseMatrix4_4;
			}
			t1 = (double)cvGetTickCount();
			if(!CameraPoseMatrix4_4.empty())
						{
								if(keyframe.size()==0)
												keyframe.push_back(CameraPoseMatrix4_4);
								 else
												{
													if( IsKeyFrame(keyframe.back(),CameraPoseMatrix4_4,max_norm, min_norm) )
														keyframe.push_back(CameraPoseMatrix4_4);
													else
														{
															TotalTime = LocationCost +TotalTime;
															if(LocationCost < min_msploop)
																usleep((min_msploop-LocationCost)*1000.);
															continue;  //结束本轮循环
														}
												}
											keyframe_num++;
											cout<<"keyframe_num is "<<keyframe_num<<endl;
										 //生成点云并滤波
											newCloud = image2PointCloud(color,depth,camera);
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
											if((keyframe_num%KF_per_submap) == 0)//整除KF_per_submap时
														{
														 submap_num++;
															sprintf(Saverpcd,"/home/bobo/code/orbslam2_3D/data/submap/%01d.pcd",submap_num);
															pcl::io::savePCDFile( Saverpcd, *output );
															output->clear();
														}
							}
			MappingCost =  ((double)cvGetTickCount() - t1)/ ((double)cvGetTickFrequency()*1000.);
			//cout<<"Mapping cost time: "<<MappingCost <<"ms"<<endl;
			TotalTime = LocationCost +MappingCost +TotalTime;
			//控制每次的循环的最小用时
			if((LocationCost +MappingCost)< min_msploop)
				usleep((min_msploop-LocationCost-MappingCost)*1000.);
	}

 /*************** end *********************/
  cout<<"Average FPS is: "<<frame_num/(TotalTime/1000)<<endl;
  cout<<"Average location FPS is: "<<frame_num/(LocationTotalTime/1000)<<endl;
  cout<<"the number of total frame is :"<<frame_num<<endl;
  cout<<"the number of keyframe is :"<<keyframe_num<<endl;
  orbslam.SaveTrajectoryTUM("/home/bobo/code/orbslam2_3D/data/CameraTrajectory.txt");
  for(int i=1;i<submap_num+1;i++)
    {
      sprintf(Saverpcd,"/home/bobo/code/orbslam2_3D/data/submap/%01d.pcd",i);
      pcl::io::loadPCDFile( Saverpcd, *tmp );
      *output += *tmp;
      tmp->clear();
    }
  pcl::io::savePCDFile( "/home/bobo/code/orbslam2_3D/data/result.pcd", *output );
  cout<<"Final map is saved."<<endl;
	//destroy
	context.StopGeneratingAll();
	context.Shutdown();
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
  //cout<<LastCamerapose<<endl;
  //cout<<CurrentCamerapose<<endl;
  //cout<<CurrentCamerapose.inv()<<endl;
  //cout<<Tc1c2<<endl;

  cv::Mat Tc1c2_3_3=Tc1c2(cv::Rect(0,0,3,3));
  cv::Rodrigues(Tc1c2_3_3,rvec);
  //cout<<"rvec="<<rvec<<endl;
   tvec = Tc1c2(cv::Rect(3,0,1,3));
   //cout<<"tvec="<<tvec<<endl;
  double change = fabs(min(cv::norm(rvec), 2*M_PI-cv::norm(rvec)))+ fabs(cv::norm(tvec));
  //cout<<"norm"<<change<<endl;
  if (change >= max_norm || change<=min_norm)
 // if (change<=min_norm)
    return false;
  else
    return true;
}
