// OpenCV_ColorFilter.cpp : Defines the entry point for the application.
//

#include "OpenCV_ColorFilter.h"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//SVD::compute(A, S, U, Vt);

Vec3f myDefV(const Mat& im)
{
    Matx33f Dsum = Matx33f::zeros();

    for (int i = 0; i < im.rows; i++)
    {
        const Vec3b* row = im.ptr<Vec3b>(i);

        for (int j = 0; j < im.cols; j++)
        {
            Vec3b di = row[j];
            Matx33f D = di * di.t();

            Dsum += D;

        }
    }


    //  Mat A = Mat::zeros(Size(5,4), CV_32F);
    //  A.at<float>(0, 0) = 1.;
    //  A.at<float>(0, 4) = 2.;
    //  A.at<float>(1, 2) = 3.;
    //  A.at<float>(3, 1) = 4.;
    //  cout <<endl<< A << endl << endl;

    Mat U, S, Vt;
    SVD::compute(Dsum, S, U, Vt);

    float porog = 0.01;
    int ind = S.at<float>(2) > porog ? 2 : S.at<float>(1) > porog ? 1 : S.at<float>(0) > porog ?0 : -1;

    Vec3f v = Vt(Range(ind,ind+1),Range::all());

    
    return v;
}

void boundCylinder(const Mat& im )
{
    Mat hist;

    int histSize = 5;
    int hist_w = 512, hist_h = 400;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    Mat gray;
    cvtColor(im, gray, COLOR_RGB2GRAY);

    calcHist(&gray, 1, 0, Mat(), hist, 1, & histSize, histRange, uniform, accumulate);
    normalize(hist, hist, 0, hist_h, NORM_MINMAX, -1, Mat());
    cout << hist << endl;
    
    imshow("hist", hist);
    waitKey(0);
}


int main()
{
	Mat im(Size(2, 3), CV_8UC3, Scalar(1,3,2));
    
    
    cout << im << endl;
    Scalar p0 = mean(im);
    Vec3f v = myDefV(im);


    boundCylinder(im);

	system("pause");
	return 0;
}
