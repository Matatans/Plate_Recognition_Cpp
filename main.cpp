#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include "Function.hpp"
//#include "Header.hpp"
#include <numeric>
#include "Plate.h"

using namespace cv;
using namespace std;



int main() {

	VideoCapture cap;

	Mat Frame1;
	Mat Frame2;

	//vector<Plate> plates;
	cap.open("Vid_B.mp4");

	//cap.open("DSCF0001.AVI");

	if ((!cap.isOpened() || (cap.get(CAP_PROP_FRAME_COUNT) < 2))) {
		cout << "Error en el video, invalido o no cumple con la cantidad min de frames" << endl;
		return(0);
	}

	//for (int i = 0; i < cap.get(CAP_PROP_FRAME_COUNT)/3; i++) {
	//	cap.read(Frame1);
	//}

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

	HOG_true_descriptor = HOGDescriptorGenerator(1, Mat());
	HOG_false_descriptor = HOGDescriptorGenerator(0, Mat());

	vector<Plate> Plates;



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
		//cvtColor(plate_tmp, plate_tmp, COLOR_BGR2GRAY);
		resize(plate_tmp, plate_tmp, Size(64, 32));
		//threshold(plate_tmp, plate_tmp, 60, 255, THRESH_BINARY_INV);
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
			//cvtColor(imgposPlate, imgposPlate, COLOR_BGR2GRAY);
			resize(imgposPlate, imgposPlate, Size(64, 32));
			//threshold(imgposPlate, imgposPlate, th1, 255, THRESH_BINARY_INV);
			matchTemplate(imgposPlate, plate_tmp, corr, TM_CCORR_NORMED);
			
			rectangle(CloneFrame2, posPlate.contRect.tl(), posPlate.contRect.br(), Scalar(255, 0, 0), 1);

			if (corr.at<float>(0, 0) >= 0.8) {
				vector<float> HOG_desc = HOGDescriptorGenerator(2, imgposPlate);
				float truePlateflg = isPlate(HOG_true_descriptor, HOG_false_descriptor, HOG_desc);
				
				if (truePlateflg >= 0.3) {
					
					cont += 1;
					imshow("posPlate " + to_string(cont), imgposPlate);
					PlatesinFrame.push_back(posPlate);
					//rectangle(CloneFrame2, posPlate.contRect.tl(), posPlate.contRect.br(), Scalar(0, 0, 255), 2);
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

/*
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include "Function.hpp"

using namespace cv;
using namespace std;




int main(int argc, char** argv)
{
    //Captura de video + error en caso de video vacio o no encontrado.
    VideoCapture vid("DSCF0003.AVI");
    if (!vid.isOpened()) {
        cout << "Error video stream" << endl;
        return -1;
    }

    //Declaración y inicialización de variables importantes.
    Mat bg, old_frame;
    vid >> bg;
    vid >> old_frame;
    resize(old_frame, old_frame, Size(640, 480));
    resize(bg, bg, Size(640, 480));
    cvtColor(old_frame, old_frame, COLOR_BGR2GRAY);
    cvtColor(bg, bg, COLOR_BGR2GRAY);
    Mat frame, fg, copyFrame, gray_frame;

    float u = 0.1;
    int th = 40;
    int aspMin = 240;
    int aspMax = 300;
    int widMax = 40;
    int widMin = 8;
    int hMax = 20;
    int hMin = 5;

    //Trackbars para perillar los parametros de detección.
    namedWindow("Trackbars", (250, 300));
    createTrackbar("th", "Trackbars", &th, 100);
    createTrackbar("aspect min", "Trackbars", &aspMin, 300);
    createTrackbar("aspect max", "Trackbars", &aspMax, 350);
    createTrackbar("widMax", "Trackbars", &widMax, (int)(bg.size().width * 0.05));
    createTrackbar("widMin", "Trackbars", &widMin, (int)(bg.size().width * 0.05));
    createTrackbar("hMax", "Trackbars", &hMax, (int)(bg.size().height * 0.05));
    createTrackbar("hMin", "Trackbars", &hMin, (int)(bg.size().height * 0.05));
    int _n_plates = 0;

    while (vid.read(frame)) {

        if (frame.empty()) break;
        resize(frame, frame, Size(640, 480));
        Mat original_frame;

        frame.copyTo(original_frame);
        frame.copyTo(copyFrame);
        frame.copyTo(gray_frame);
        cvtColor(gray_frame, gray_frame, COLOR_BGR2GRAY);
        //Preprocesamiento, bgsus + erode + dilate (parametros th para la binarización)
        BgSus(frame, &fg, u, &bg, th);

        //Declaración de arreglos para guardar los contornos y patentes.
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;

        //Detección de contornos.
        findContours(fg, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        drawContours(frame, contours, -1, Scalar(0, 0, 255), 1);

        //Vectores de almacenamiento
        vector<Rect> cont_rect(contours.size());
        //vector<vector<Point>> filter_cont(contours.size());

        //Filtrado de contornos por ratio y por tamaño.
       // int n_plates = FilterContours(contours, &cont_rect, &filter_cont,aspMin, aspMax, widMin, widMax, hMin, hMax);
        vector<vector<Point>> filter_cont = FilterContours(contours, &cont_rect, aspMin, aspMax, widMin, widMax, hMin, hMax);
        cout << filter_cont.size() << endl;
        if (!filter_cont.empty()) drawContours(copyFrame, filter_cont, -1, Scalar(0, 255, 0), 2);

        for (size_t i = 0; (i < cont_rect.size()) & !cont_rect.empty(); i++) {
            //drawContours(copyFrame, filter_cont, -1, Scalar(0, 255, 0), 2);
            //rectangle(copyFrame, cont_rect[i].tl(), cont_rect[i].br(), Scalar(255, 0, 0), 1);
        }
        //Crop de posibles patentes para correlación y confirmación de F+.
        Mat plate_crop, crop_ref;
        for (int i = 0; i < _n_plates; i++) {
            destroyWindow("plate" + to_string(i));
        }
        Mat plate_template = imread("pat_ext.PNG");
        cvtColor(plate_template, plate_template, COLOR_BGR2GRAY);
        Mat corr;
        for (int i = 0; i < filter_cont.size(); i++) {
            crop_ref = original_frame(cont_rect[i]);
            crop_ref.copyTo(plate_crop);
            cvtColor(plate_crop, plate_crop, COLOR_BGR2GRAY);
            resize(plate_crop, plate_crop, plate_template.size());
            matchTemplate(plate_crop, plate_template, corr, TM_CCORR_NORMED );
            imshow("plate" + to_string(i), plate_crop);

            cout << "plate n°" << i << "tiene una corr de " << corr.at<float>(0, 0) << endl;
            rectangle(copyFrame, cont_rect[i].tl(), cont_rect[i].br(), Scalar(255, 0, 0), 2);
        }

        /*vector<uchar> status;
        vector<float> err;
        vector<Point2f> prev_plate, new_plate;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT)+(TermCriteria::EPS), 10, 0.03);
      

        vector<Point2f> p0, p1;
        goodFeaturesToTrack(old_frame, p0, 5, 0.3, 7, Mat(), 7, false, 0.04);

        calcOpticalFlowPyrLK(old_frame, gray_frame, p0, p1, status, err, Size(15, 15), 2, criteria);

        vector<Point2f> good_new;
        vector<Scalar> colors;
        RNG rng;
        for (int i = 0; i < 100; i++)
        {
            int r = rng.uniform(0, 256);
            int g = rng.uniform(0, 256);
            int b = rng.uniform(0, 256);
            colors.push_back(Scalar(r, g, b));
        }

        for (uint i = 0; i < p0.size(); i++)
        {
            // Select good points
            if (status[i] == 1) {
                good_new.push_back(p1[i]);
                // Draw the tracks
                line(copyFrame, p1[i], p0[i], colors[i], 2);
                circle(copyFrame, p1[i], 5, colors[i], -1);
            }
        }


        //Guardar los datos anteriores.

        gray_frame.copyTo(old_frame);
        p0 = good_new;
        //
        _n_plates = filter_cont.size();
        imshow("copia filtrada", copyFrame);
        imshow("copia ", frame);
        imshow("foreground", fg);

        char c = (char)waitKey(1);
        if (c == 27) break;
        else if (c == 32) { waitKey(0); }
    }
    
    vid.release();
    cv::destroyAllWindows;
    return 0;
}
*/