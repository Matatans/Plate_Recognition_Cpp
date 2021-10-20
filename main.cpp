#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <numeric>

#include "Function.cpp"
#include "Plate.h"

using namespace cv;
using namespace std;

int main() {

	VideoCapture cap;

	Mat Frame1;
	Mat Frame2;

	vector<Plate> Plates;

	cap.open("Vid_B.mp4");


	if ((!cap.isOpened() || (cap.get(CAP_PROP_FRAME_COUNT) < 2))) {
		cout << "Error en el video, invalido o no cumple con la cantidad min de frames" << endl;
		return(0);
	}

	cap.read(Frame1);
	resize(Frame1, Frame1, Size(1024, 720));
	cap.read(Frame2);
	resize(Frame2, Frame2, Size(1024, 720));
	Mat Background;

	///////////////////////////////////////Parametros ajustables mediantes Trackbars/////////////////////////////////////////////////
	float u = 0.1;
	int th = 15;
	int aspMin = 180;
	int aspMax = 320;
	int widMax = 35;
	int widMin = 10;
	int hMax = 20;
	int hMin = 7;
	int th1 = 60;

	namedWindow("Trackbars", (250, 300));
	createTrackbar("th", "Trackbars", &th, 100);
	createTrackbar("th1", "Trackbars", &th1, 150);
	createTrackbar("aspect min", "Trackbars", &aspMin, 300);
	createTrackbar("aspect max", "Trackbars", &aspMax, 400);
	createTrackbar("widMax", "Trackbars", &widMax, (int)(Frame1.size().width * 0.05));
	createTrackbar("widMin", "Trackbars", &widMin, (int)(Frame1.size().width * 0.05));
	createTrackbar("hMax", "Trackbars", &hMax, (int)(Frame1.size().height * 0.05));
	createTrackbar("hMin", "Trackbars", &hMin, (int)(Frame1.size().height * 0.05));

	/// Variables/flag utiles///////////////////////////////////////////////////////////////////////////

	int n_plates_prev = 0;
	bool firstFrame = true;
	char key = 0;


	HOGDescriptor hog;
	hog.winSize = Size(128, 64);
	hog.blockSize = Size(hog.winSize.width / 4, hog.winSize.height / 4);
	hog.blockStride = Size(16, 32);
	hog.cellSize = Size(16, 32);

	vector<float> HOG_true_descriptor;
	vector<float> HOG_false_descriptor;
	vector<float> HOG_desc;

	HOG_true_descriptor = HOGDescriptorGenerator(1, Mat()); //Vector de caracteristicas HOG promedio de muestras de patente
	HOG_false_descriptor = HOGDescriptorGenerator(0, Mat()); //Vector de caracteristicas HOG promedio de muestras falsas

	while (cap.isOpened() && key != 27) {

		Mat CloneFrame1 = Frame1.clone();
		Mat CloneFrame2 = Frame2.clone();

		cvtColor(CloneFrame1, CloneFrame1, COLOR_BGR2GRAY);
		cvtColor(CloneFrame2, CloneFrame2, COLOR_BGR2GRAY);
		imshow("CloneFrame1", CloneFrame1);


		if (firstFrame) {
			Background = CloneFrame1.clone();
		}
		else {
			Background = Background * (0.9) + CloneFrame2 * 0.1;
		}

		Mat imgForeground;

		imgForeground = CloneFrame2 - Background;
		imshow("Foreground", imgForeground);
		medianBlur(imgForeground, imgForeground, 3);
		imshow("Foreground_blur", imgForeground);

		Mat imgBinary;
		threshold(imgForeground, imgBinary, std::max(10, th), 255.0, THRESH_BINARY);
		imshow("ImgBin", imgBinary);

		Mat Struct3x3 = getStructuringElement(MORPH_RECT, Size(3, 3));
		Mat Struct5x5 = getStructuringElement(MORPH_RECT, Size(5, 5));

		dilate(imgBinary, imgBinary, Struct3x3);
		imshow("Post-Morph -dilate 1", imgBinary);
		erode(imgBinary, imgBinary, Struct3x3);
		imshow("Post-Morph", imgBinary);

		vector<vector<Point> > contours;

		findContours(imgBinary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		PrintContours(imgBinary.size(), contours, "contours");

		/* Pruebas con convexhulls
		vector<vector<Point>> convHulls(contours.size());

		for (int i = 0; i < contours.size(); i++) {
			convexHull(contours[i], convHulls[i]);
		}
		PrintContours(imgBinary.size(), convHulls, "Hulls");
		/*/ 

		vector<Rect> conRect(contours.size());
		CloneFrame2 = Frame2.clone();
		Mat imgRectsinFiltrar = Frame2.clone();
		float faspMin = ((float)aspMin / 100.0);
		float faspMax = ((float)aspMax / 100.0);
		vector<Plate> possiblePlate;

		for (auto& contour : contours) {
			Plate PlateType(contour);

			if (PlateType.AspectRatio > faspMin &&
				PlateType.AspectRatio < faspMax &&
				PlateType.contRect.width > widMin &&
				PlateType.contRect.width < widMax &&
				PlateType.contRect.height > hMin &&
				PlateType.contRect.height < hMax) {
				possiblePlate.push_back(PlateType);

			}

			rectangle(imgRectsinFiltrar, PlateType.contRect.tl(), PlateType.contRect.br(), Scalar(255, 0, 0), 2);
		}

		//Carga de modelo de patente

		Mat plate_tmp = imread("img/tmp_plate2.PNG");
		resize(plate_tmp, plate_tmp, Size(64, 32));
 		cv::imshow("plate tmp", plate_tmp);
		Mat corr;

		//Destrucción de ventanas de posibles patentes
		for (int i = 1; i <= n_plates_prev; i++) {
			destroyWindow("posPlate " + to_string(i));
		}

		int cont = 0;
		vector<Plate> PlatesinFrame;

		for (auto& posPlate : possiblePlate) {

			Mat imgposPlate = Frame2(posPlate.contRect);
			resize(imgposPlate, imgposPlate, Size(64, 32));

			matchTemplate(imgposPlate, plate_tmp, corr, TM_CCORR_NORMED);
			
			rectangle(CloneFrame2, posPlate.contRect.tl(), posPlate.contRect.br(), Scalar(255, 0, 0), 1);

			if (corr.at<float>(0, 0) >= 0.9) {
				vector<float> HOG_desc = HOGDescriptorGenerator(2, imgposPlate);
				float truePlateflg = isPlate(HOG_true_descriptor, HOG_false_descriptor, HOG_desc);
				
				if (truePlateflg >= 0.3) {
					
					cont += 1;
					imshow("posPlate " + to_string(cont), imgposPlate);
					PlatesinFrame.push_back(posPlate);
 					cout << "Plate " << cont << ": (" << posPlate.contRect.width << "," << posPlate.contRect.height <<
						"); aspect =" << posPlate.AspectRatio << " tiene una corr =" << corr.at<float>(0, 0) << " HOG =" << truePlateflg << endl;

				}
				else {
					rectangle(CloneFrame2, posPlate.contRect.tl(), posPlate.contRect.br(), Scalar(0, 255, 0), 2);
				}
			}
		}

		

		if (firstFrame) {
			// Si es el primer frame, agregar toda la lista de patentes en el frame.
			for (auto& FramePlate : PlatesinFrame) {
				Plates.push_back(FramePlate);
			}
		}
		else {// Si no, hacer match entre las patentes en el frame y las patentes extraidas.
			matchFramePlateToPlates(PlatesinFrame, Plates);
		}

		printPlateInfo(Plates, CloneFrame2);

		n_plates_prev = cont;

		cv::imshow("con REct", CloneFrame2);
		cv::imshow("imgRectsinFiltrar", imgRectsinFiltrar);





		Frame1 = Frame2.clone();
		if ((cap.get(CAP_PROP_POS_FRAMES) + 1) < cap.get(CAP_PROP_FRAME_COUNT)) {
			cap.read(Frame2);
			resize(Frame2, Frame2, Size(1024, 720));
		}
		else {
			std::cout << "end of video\n";
			break;
		}


		key = (char)waitKey(10);
		if (key == 32) { waitKey(0); }
		firstFrame = false;

	}

	cap.release();
	cv::destroyAllWindows;
}
