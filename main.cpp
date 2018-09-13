#include "opencv2/opencv.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <cstdio>
#include <omp.h>
#include <time.h>
#include <fstream>
#include <math.h>
#include <algorithm>

#define NUM_SAMPLES 20
#define MIN_MATCHES 2
#define RADIUS 20
#define SUBSAMPLE_FACTOR 16
#define RADIUS_FORE 10


using namespace std;
using namespace cv;
long unsigned int count2_3=0,count2_23=0,count1_21=0,count2_21=0,count1_23;
const int c_xoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};   
const int c_yoff[9] = {-1,  0,  1, -1, 1, -1, 0, 1, 0};  //随机数种子
class ViBe_BGS
{
public:
	ViBe_BGS(void){};
	~ViBe_BGS(void){};
	void init(Mat _image);
	void readFirstFrame(Mat _image);
	void testAndUpdate(Mat _image);
	void testAndUpdate1(Mat _image,Mat _image_Prev1,Mat _image_Prev2,Mat _mask_Prev);
	Mat getMask(void);
private:
	Mat m_samples[NUM_SAMPLES];
	Mat m_foregroundMatchCount;
	Mat m_mask;
};

void viBe();
Mat ViBe_BGS::getMask(void)
{
	return m_mask;
}

void ViBe_BGS::testAndUpdate(Mat _image)
{
	RNG rng;
	int matches(0),count(0);
	double distance;
	int rand;

	for (int i=0;i<_image.rows;i++)
	{
		for (int j=0;j<_image.cols;j++)
		{
			matches=count=0;
			while (matches<MIN_MATCHES&&count<NUM_SAMPLES)
			{
				distance=abs(m_samples[count].at<uchar>(i,j)-_image.at<uchar>(i,j));
				if (distance<RADIUS)
				{
					matches++;
				}
				count++;
			}
			if (matches>=MIN_MATCHES)
			{
				//It is a backgroundPoint
				m_foregroundMatchCount.at<uchar>(i,j)=0;
				m_mask.at<uchar>(i,j)=0;
				//按照一定概率更新20个背景模型其中的一个 的i，j点的值
				rand=rng.uniform(0,SUBSAMPLE_FACTOR);
				if (rand==0)
				{
					rand=rng.uniform(0,NUM_SAMPLES);
					m_samples[rand].at<uchar>(i,j)=_image.at<uchar>(i,j);

				}
				//按照一定概率更新20个背景模型中其中一个的 i，j点 8个邻居其中一个邻居的值
				rand=rng.uniform(0,SUBSAMPLE_FACTOR);
				if (rand==0)
				{
					int row,col;
					rand=rng.uniform(0,9);
					row=i+c_yoff[rand];
					if (row<0)
					{
						row=0;
					}
					if (row>=_image.rows)
					{
						row=_image.rows-1;
					}
					rand=rng.uniform(0,9);
					col=j+c_xoff[rand];
					if (col<0)
					{
						col=0;
					}
					if (col>=_image.cols)
					{
						col=_image.cols-1;
					}
					rand=rng.uniform(0,NUM_SAMPLES);
					m_samples[rand].at<uchar>(row,col)=_image.at<uchar>(i,j);

				}
			}
			else{
				m_foregroundMatchCount.at<uchar>(i,j)++;
				m_mask.at<uchar>(i,j)=255;
				if (m_foregroundMatchCount.at<uchar>(i,j)>50)
				{
					int rand=rng.uniform(0,SUBSAMPLE_FACTOR);
					if (rand==0)
					{
						rand=rng.uniform(0,NUM_SAMPLES);
						m_samples[rand].at<uchar>(i,j)=_image.at<uchar>(i,j);
					}
				}
			}
		}
	}
}
long unsigned int sum_pi=0;
void ViBe_BGS::testAndUpdate1(Mat _image,Mat _image_Prev1,Mat _image_Prev2,Mat _mask_Prev)
{
	RNG rng;
	int matches(0),count(0);
	double distance;
	int rand;
	for (int i=0;i<_image.rows;i++)
	{
		for (int j=0;j<_image.cols;j++)
		{
			sum_pi++;
			//一级判断
			if(_mask_Prev.at<uchar>(i,j)<125){
				//前一个是背景
				matches=0;
				//相似度判断
				distance=abs(_image.at<uchar>(i,j)-_image_Prev1.at<uchar>(i,j));
				if(distance<RADIUS_FORE) matches++;
				distance=abs(_image.at<uchar>(i,j)-_image_Prev2.at<uchar>(i,j));
				if(distance<RADIUS_FORE) matches++;
				if(matches>0){
					//It is a Background
					count2_3++;
					//It is a backgroundPoint
					m_foregroundMatchCount.at<uchar>(i,j)=0;
					m_mask.at<uchar>(i,j)=0;
					//按照一定概率更新20个背景模型其中的一个 的i，j点的值
					rand=rng.uniform(0,SUBSAMPLE_FACTOR);
					if (rand==0)
					{
						rand=rng.uniform(0,NUM_SAMPLES);
						m_samples[rand].at<uchar>(i,j)=_image.at<uchar>(i,j);

					}
					//按照一定概率更新20个背景模型中其中一个的 i，j点 8个邻居其中一个邻居的值
					rand=rng.uniform(0,SUBSAMPLE_FACTOR);
					if (rand==0)
					{
						int row,col;
						rand=rng.uniform(0,9);
						row=i+c_yoff[rand];
						if (row<0)
						{
							row=0;
						}
						if (row>=_image.rows)
						{
							row=_image.rows-1;
						}
						rand=rng.uniform(0,9);
						col=j+c_xoff[rand];
						if (col<0)
						{
							col=0;
						}
						if (col>=_image.cols)
						{
							col=_image.cols-1;
						}
						rand=rng.uniform(0,NUM_SAMPLES);
						m_samples[rand].at<uchar>(row,col)=_image.at<uchar>(i,j);

					}
				}
				else{		
					matches=count=0;
					while (matches<MIN_MATCHES&&count<NUM_SAMPLES)
					{
						distance=abs(m_samples[count].at<uchar>(i,j)-_image.at<uchar>(i,j));
						if (distance<RADIUS)
						{
							matches++;
						}
						count++;
					}
					if (matches>=MIN_MATCHES)
					{
						//It is a backgroundPoint
						count2_23++;
						m_foregroundMatchCount.at<uchar>(i,j)=0;
						m_mask.at<uchar>(i,j)=0;
						//按照一定概率更新20个背景模型其中的一个 的i，j点的值
						rand=rng.uniform(0,SUBSAMPLE_FACTOR);
						if (rand==0)
						{
							rand=rng.uniform(0,NUM_SAMPLES);
							m_samples[rand].at<uchar>(i,j)=_image.at<uchar>(i,j);

						}
						//按照一定概率更新20个背景模型中其中一个的 i，j点 8个邻居其中一个邻居的值
						rand=rng.uniform(0,SUBSAMPLE_FACTOR);
						if (rand==0)
						{
							int row,col;
							rand=rng.uniform(0,9);
							row=i+c_yoff[rand];
							if (row<0)
							{
								row=0;
							}
							if (row>=_image.rows)
							{
								row=_image.rows-1;
							}
							rand=rng.uniform(0,9);
							col=j+c_xoff[rand];
							if (col<0)
							{
								col=0;
							}
							if (col>=_image.cols)
							{
								col=_image.cols-1;
							}
							rand=rng.uniform(0,NUM_SAMPLES);
							m_samples[rand].at<uchar>(row,col)=_image.at<uchar>(i,j);

						}
					}
					else{
						count1_23++;
						m_foregroundMatchCount.at<uchar>(i,j)++;
						m_mask.at<uchar>(i,j)=255;
						if (m_foregroundMatchCount.at<uchar>(i,j)>50)
						{
							int rand=rng.uniform(0,SUBSAMPLE_FACTOR);
							if (rand==0)
							{
								rand=rng.uniform(0,NUM_SAMPLES);
								m_samples[rand].at<uchar>(i,j)=_image.at<uchar>(i,j);
							}
						}
					}
				}
			}
			else{
				//前一个是前景
				matches=count=0;
				while (matches<MIN_MATCHES&&count<NUM_SAMPLES)
				{
					distance=abs(m_samples[count].at<uchar>(i,j)-_image.at<uchar>(i,j));
					if (distance<RADIUS)
					{
						matches++;
					}
					count++;
				}
				if (matches>=MIN_MATCHES)
				{
					count2_21++;
					//It is a backgroundPoint
					m_foregroundMatchCount.at<uchar>(i,j)=0;
					m_mask.at<uchar>(i,j)=0;
					//按照一定概率更新20个背景模型其中的一个 的i，j点的值
					rand=rng.uniform(0,SUBSAMPLE_FACTOR);
					if (rand==0)
					{
						rand=rng.uniform(0,NUM_SAMPLES);
						m_samples[rand].at<uchar>(i,j)=_image.at<uchar>(i,j);

					}
					//按照一定概率更新20个背景模型中其中一个的 i，j点 8个邻居其中一个邻居的值
					rand=rng.uniform(0,SUBSAMPLE_FACTOR);
					if (rand==0)
					{
						int row,col;
						rand=rng.uniform(0,9);
						row=i+c_yoff[rand];
						if (row<0)
						{
							row=0;
						}
						if (row>=_image.rows)
						{
							row=_image.rows-1;
						}
						rand=rng.uniform(0,9);
						col=j+c_xoff[rand];
						if (col<0)
						{
							col=0;
						}
						if (col>=_image.cols)
						{
							col=_image.cols-1;
						}
						rand=rng.uniform(0,NUM_SAMPLES);
						m_samples[rand].at<uchar>(row,col)=_image.at<uchar>(i,j);

					}
				}
				else{
					count1_21++;
					m_foregroundMatchCount.at<uchar>(i,j)++;
					m_mask.at<uchar>(i,j)=255;
					if (m_foregroundMatchCount.at<uchar>(i,j)>50)
					{
						int rand=rng.uniform(0,SUBSAMPLE_FACTOR);
						if (rand==0)
						{
							rand=rng.uniform(0,NUM_SAMPLES);
							m_samples[rand].at<uchar>(i,j)=_image.at<uchar>(i,j);
						}
					}
				}
			}
			
		}
	}
}
void ViBe_BGS::readFirstFrame(Mat _image)
{
	RNG rng;
	int row,col;
	for (int i=0;i<_image.rows;i++)
	{
		for (int j=0;j<_image.cols;j++)
		{
			for (int k=0;k<NUM_SAMPLES;k++)
			{
				int random=rng.uniform(0,9);//随机抽取
				row=i+c_yoff[random];
				col=j+c_xoff[random];
				//限定范围
				if (row<0) row=0;
				if (row>=_image.rows) row=_image.rows-1;
				if (col<0) col=0;
				if (col>=_image.cols) col=_image.cols-1;

				m_samples[k].at<uchar>(i,j)=_image.at<uchar>(row,col);
			}
		}		
	}
}

void ViBe_BGS::init(Mat _image)
{
	for(int i=0;i<NUM_SAMPLES;i++){
		m_samples[i]=Mat::zeros(_image.size(),CV_8UC1);
	}
	m_mask=Mat::zeros(_image.size(),CV_8UC1);
	m_foregroundMatchCount=Mat::zeros(_image.size(),CV_8UC1);
}

void viBe()
{
	Mat frame,mask,grayPrev1,grayPrev2,maskPrev1;
	VideoCapture capture;
	capture.open("1.avi");
	if(!capture.isOpened())
	{
		cout<<"not opened\n"<<endl;
		return ;
	}
	int count=0;
	ViBe_BGS ViBe_bgs;
	clock_t start_time,end_time;
	ofstream in;
	CvSize imgSize;
	capture>>grayPrev2;
	imgSize=grayPrev2.size();
	if(grayPrev2.empty())
		return;
	cvtColor(grayPrev2,grayPrev2,CV_RGB2GRAY);
	ViBe_bgs.init(grayPrev2);
	ViBe_bgs.readFirstFrame(grayPrev2);
	cout<<"Init Completed!"<<endl;
	cv::VideoWriter writer;
	writer=VideoWriter("111.avi",CV_FOURCC('X','V','I','D'),30,imgSize,0);

	capture>>grayPrev1;
	if(grayPrev1.empty())
		return;
	cvtColor(grayPrev1,grayPrev1,CV_RGB2GRAY);
	ViBe_bgs.testAndUpdate(grayPrev1);
	mask=ViBe_bgs.getMask();
	morphologyEx(mask,mask,MORPH_OPEN,Mat());
	//imshow("mask",mask);
	maskPrev1=mask.clone();

	while(1)
	{
		start_time=clock();
		count++;
		capture>>frame;
		if(frame.empty())
			break;
		cvtColor(frame,frame,CV_RGB2GRAY);
		ViBe_bgs.testAndUpdate1(frame,grayPrev1,grayPrev2,maskPrev1);
		mask=ViBe_bgs.getMask();
		//medianBlur(mask,mask,3);
		//morphologyEx(mask,mask,MORPH_OPEN,Mat());
		
		maskPrev1=mask;
		grayPrev2=grayPrev1;
		grayPrev1=frame;
		//end_time1=clock();
		end_time=clock();
		//cout << "Running time is: "<<static_cast<double>(end_time1-start_time1)/CLOCKS_PER_SEC*1000<<"ms"<<endl;//输出运行时间
		//in <<static_cast<double>(end_time-start_time)/CLOCKS_PER_SEC*1000<<" "<<endl;//输出运行时间
		imshow("mask",mask);
		imshow("input",frame);
		writer.write(mask);
		//imshow("prevMask",maskPrev1);
		//imshow("grayPrev1",grayPrev1);
		//imshow("grayPrev2",grayPrev2);
		if (cvWaitKey(10)=='q')
		{
			break;
		}
	}
}
int main(int argc,char* argv[])
{
	viBe();
}