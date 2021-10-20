#pragma once
#include <numeric>
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

class Plate{
public:
	std::vector<cv::Point> Contour;

	cv::Rect contRect;

	std::vector<cv::Point> centerPos;

	double AspectRatio;
	double DiagSize;


	bool isNewPlateOrMatch;
	bool isTracked;

	int nFramesNoTracked;
	cv::Point NextPos;


	Plate(std::vector<cv::Point> _cnt);
	void predictPosition(void);

};

