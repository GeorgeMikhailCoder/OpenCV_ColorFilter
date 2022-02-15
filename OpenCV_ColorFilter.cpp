// OpenCV_ColorFilter.cpp : Defines the entry point for the application.
//

#include "OpenCV_ColorFilter.h"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//SVD::compute(A, S, U, Vt);

#define MatPair Vec<Mat_<float>,2>
#define CFpar ColorFilterParams

struct ColorFilterParams
{
    Vec3f p0;
    Vec3f v;
    float t1;
    float t2;
    float R;

    ColorFilterParams(Vec3f p0n, Vec3f vn, float t1n, float t2n, float Rn)
    {
        p0 = p0n;
        v = vn;
        t1 = t1n;
        t2 = t2n;
        R = Rn;
    }
};

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
    cout << S << endl;
    cout << Vt << endl;
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

MatPair getTRmass(const Mat& im, Vec3f v, Vec3f p0)
{
    cout <<"p0 = "<< p0 << endl;
    Mat dMat;
    im.convertTo(dMat, CV_32FC3);
    dMat = dMat - Scalar(p0);
    Mat Tmass;
    Mat Rmass;
    for (int y = 0; y < im.rows; y++)
        for (int x = 0; x < im.cols; x++)
        {
            const Vec3f di = dMat.at<Vec3f>(y, x);
            float ti = di.dot(v)/ norm(v);  
            Tmass.push_back(ti);
            float R = sqrt(norm(di) * norm(di) - norm(ti * v) * norm(ti * v));
            Rmass.push_back(R);
        }
    Mat sT;
    cv::sort(Tmass, sT, SORT_EVERY_COLUMN + SORT_ASCENDING);
    sT += 255;
    sT = sT.mul(0.5);

    Mat sR;
    cv::sort(Rmass, sR, SORT_EVERY_COLUMN + SORT_ASCENDING);
    
    return MatPair(sT, sR);
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

Vec2f getTparam(Mat sT) 
{
    float alpha = 0.01;
    float curSumT(0), sumT(sum(sT)[0]), curPercent(0);
    int indT1(-1), indT2(-1);
    float t1(0), t2(0);
    for (int i = 0; i < sT.rows; i++)
    {
        curSumT += sT.at<float>(i);
        curPercent = curSumT / sumT;
        if (indT1 == -1 && curPercent >= alpha)
            indT1 = i;
        if (indT2 == -1 && curPercent >= (1 - alpha))
            indT2 = i;
    }
    t1 = indT1 == -1 ? 0 : sT.at<float>(indT1);
    t2 = indT2 == -1 ? 0 : sT.at<float>(indT2);
    return Vec2f(t1, t2);
}

Mat applyFilter(const Mat& src, CFpar filt, Vec3b background = Vec3b(255,255,255))
{
    Vec3f p0 = filt.p0;
    Vec3f v = filt.v;
    float t1 = filt.t1;
    float t2 = filt.t2;
    float R = filt.R;


    Mat im;
    src.convertTo(im, CV_32FC3);
    im = im - Scalar(p0);
    Mat res(src);

    for(int y=0;y<im.rows;y++)
        for (int x = 0; x < im.cols; x++)
        {
            res.at<Vec3b>(y, x);
            Vec3f di = im.at<Vec3f>(y, x);
            float ti = di.dot(v) / norm(v);
            float ri = sqrt(norm(di) * norm(di) - norm(ti * v) * norm(ti * v));

            if (t1 < ti && ti < t2
                && ri < R)
                res.at<Vec3b>(y, x) = src.at<Vec3b>(y, x);
            else 
            {
                res.at<Vec3b>(y, x) = background;
            }
        }
    return res;
}

int main()
{
//  	Mat im(Size(30, 30), CV_8UC3, Scalar(0,100,0));
//      im(Rect(10, 10, 15, 15)) = Scalar(0,200,0);
//      imshow("im", im);

    Mat trainImage = imread("../../../img/redtemplate1.jpg", IMREAD_COLOR);
    if (trainImage.empty()) {
        cout << "Error: can not load train image" << endl;
        char a;
        cin >> a;
        exit(0);
    }
    imshow("train image", trainImage);

    Mat realImage = imread("../../../img/redcar1.jpg", IMREAD_COLOR);
    if (realImage.empty()) {
        cout << "Error: can not load real image" << endl;
        char a;
        cin >> a;
        exit(0);
    }
    imshow("real image", realImage);
    
    Scalar tmp = mean(trainImage);
    Vec3f p0(tmp[0],tmp[1],tmp[2]);
    Vec3f v = myDefV(trainImage);


    MatPair TRmass = getTRmass(trainImage, v, p0);
    
    imshow("histT", hist2mat(img2hist(TRmass[0])));
    imshow("histR", hist2mat(img2hist(TRmass[1])));

    Vec2f Tparams = getTparam(TRmass[0]);
    Vec2f Rparams = getTparam(TRmass[1]);
    
    float t1 = Tparams[0];
    float t2 = Tparams[1];
    float R = Rparams[1];

    cout << "p0 = " << p0 << endl << "v = " << v << endl;
    cout << "t1 = " << t1 << "  t2 = " << t2 << endl;
    cout << "R = " << R << endl;

    CFpar filt = CFpar(p0, v,t1, t2, R);
    Mat filteredImage = applyFilter(realImage, filt);

    imshow("filtered image", filteredImage);
    waitKey();

	system("pause");
	return 0;
}
