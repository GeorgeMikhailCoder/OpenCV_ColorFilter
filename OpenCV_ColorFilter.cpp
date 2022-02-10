// OpenCV_ColorFilter.cpp : Defines the entry point for the application.
//

#include "OpenCV_ColorFilter.h"
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main()
{
	
	Vec<double, 4> di; 
	di << 1, 2, 3, 4;
	cout << di << endl;
	cout<<di.t() << endl;
	cout << di * di.t() << endl;
	cout << di * di.t() << endl;
	Matx22d D;//= di * di.t();
	D << 1, 0, 0, 1;
	Matx22d U, S, Vt;
	SVD().compute(D, S, U, Vt);
	//  double v = Vt(4);
	//  cout << v << endl;


	system("pause");
	return 0;
}
