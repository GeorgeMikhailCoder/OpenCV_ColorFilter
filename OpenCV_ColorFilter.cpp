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
    
    v = v / norm(v);
    
    return v;
}

const int histSize = 256;
const int hist_w = 512, hist_h = 256;
Mat img2hist(Mat img)
{
    vector<Mat> bgr_planes;
    split(img, bgr_planes);
       

    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    // Mat b_hist, g_hist, r_hist;

    vector<Mat> brg_hists;
    brg_hists.resize(img.channels());
    for (int i = 0; i < img.channels(); i++)
    {
        calcHist(&(bgr_planes[i]), 1, 0, Mat(), brg_hists[i], 1, &histSize, histRange, uniform, accumulate);
        normalize(brg_hists[i], brg_hists[i], 0, hist_h, NORM_MINMAX, -1, Mat());
    }

    //  #define CV_8U   0
    //  #define CV_8S   1
    //  #define CV_16U  2
    //  #define CV_16S  3
    //  #define CV_32S  4
    //  #define CV_32F  5
    //  #define CV_64F  6
    //  #define CV_16F  7


    Mat res;
    merge(brg_hists,res);
    

    return res;
}

Mat img2hist_old(Mat img)
{
    vector<Mat> bgr_planes;
    split(img, bgr_planes);


    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;

    Mat b_hist, g_hist, r_hist;

    calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, histRange, uniform, accumulate);



    normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
    normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
    normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());

    vector<Mat> rgb_hist = { b_hist, g_hist, r_hist };

    Mat res;
    merge(rgb_hist, res);
    return res;
}

Mat hist2mat(Mat rgb_hist)
{
    vector<Mat> rgb_planes;
    split(rgb_hist, rgb_planes);

    //  Mat b_hist, g_hist, r_hist;
    //  r_hist = rgb_planes[0];
    //  g_hist = rgb_planes[1];
    //  b_hist = rgb_planes[2];


    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    for (int i = 1; i < histSize; i++)
    {
        for (int ch = 0; ch < rgb_hist.channels(); ch++)
        {
            Scalar color = 0;
            color(ch) = 255;
            line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(rgb_planes[ch].at<float>(i - 1))),
                Point(bin_w * (i), hist_h - cvRound(rgb_planes[ch].at<float>(i))),
                color, 2, 8, 0);
        }

     //   line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(b_hist.at<float>(i - 1))),
     //       Point(bin_w * (i), hist_h - cvRound(b_hist.at<float>(i))),
     //       Scalar(255, 0, 0), 2, 8, 0);
     //   line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(g_hist.at<float>(i - 1))),
     //       Point(bin_w * (i), hist_h - cvRound(g_hist.at<float>(i))),
     //       Scalar(0, 255, 0), 2, 8, 0);
     //   line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
     //       Point(bin_w * (i), hist_h - cvRound(r_hist.at<float>(i))),
     //       Scalar(0, 0, 255), 2, 8, 0);
    }
    return histImage;
}



Mat boundCylinder(const Mat& im, Vec3f v, Scalar p0)
{
    cout <<"p0 = "<< p0 << endl;
    
    Mat dMat = im - p0;
    //cout << dMat << endl;
    Mat Tmass;
    for (int y = 0; y < im.rows; y++)
        for (int x = 0; x < im.cols; x++)
        {
            const Vec3b di = dMat.at<uchar>(y, x);
            float ti = di.dot(v)/ norm(v);  
            Tmass.push_back(ti);
        }
    Mat sT;
    cv::sort(Tmass, sT, SORT_EVERY_COLUMN + SORT_ASCENDING);
    return sT;
}

bool myCompareMat(const Mat& m1, const Mat& m2)
{
    Mat res;
    absdiff(m1, m2, res);
    cout << res << endl;
    double minEl, maxEl;
    minMaxLoc(res, &minEl, &maxEl);
    return maxEl == 0;
}

int main()
{
	Mat im(Size(300, 500), CV_8UC3, Scalar(200,50,40));
    im(Rect(20, 20, 200, 300)) = Scalar(100, 150, 0);
    imshow("im", im);
    
    //cout << im << endl;
    Scalar p0 = mean(im);
    Vec3f v = myDefV(im);
    Mat sT = boundCylinder(im,v,p0);
    //cout << sT << endl;
    Mat hist = img2hist(sT);


    imshow("hist_new", hist2mat(hist));
    waitKey();
	system("pause");
	return 0;
}
