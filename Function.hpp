#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <vector>
#include <numeric>
#include "Plate.h"

using namespace cv;
using namespace std;
//Prototipos de función.

//Funciones variadas.
void PrintContours(Size, vector<vector<Point> >, string);
double euclideanDistance(Point, Point);

//Funciones para detección via clasificador de media con HOG features.
int HOGDescriptorGenerator(HOGDescriptor, vector<float>*, int, Mat);
vector<float> HOGDescriptorGenerator(char, Mat);
float isPlate(vector<float>, vector<float>, vector<float>);

//Funciones para Match y manejo de vector de patentes.
void addPlateToMatch(vector<Plate>&, Plate&, int);
void addNewPlate(vector<Plate>&, Plate&);
void matchFramePlateToPlates(vector<Plate>&, vector<Plate>&);
void printPlateInfo(std::vector<Plate>&, cv::Mat&);
//Declaraciones de funciones.

void printPlateInfo(std::vector<Plate>& Plate, cv::Mat& imgCopy) {
    for (int i = 0; i < Plate.size(); i++) {

        if (Plate[i].isTracked == true) {
            rectangle(imgCopy, Plate[i].contRect, Scalar(0, 0, 255), 2);
            putText(imgCopy, to_string(i), Plate[i].contRect.tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }
    }
}

float isPlate(vector<float> trueVector, vector<float> falseVector, vector<float> testVector) {

    vector<float> dist_t, dist_f;

    for (int i = 0; i < testVector.size(); i++) {

        dist_t.push_back(pow(testVector[i] - trueVector[i], 2));
        dist_f.push_back(pow(testVector[i] - falseVector[i], 2));

    }
    float sum_dist_t, sum_dist_f;

    sum_dist_t = sqrt(accumulate(dist_t.begin(), dist_t.end(), 0.0f));
    sum_dist_f = sqrt(accumulate(dist_f.begin(), dist_f.end(), 0.0f));
    return (sum_dist_f - sum_dist_t);
}

void PrintContours(Size imageSize, vector<vector<Point> > contours, string strImageName) {
    cv::Mat image(imageSize, CV_8UC3, Scalar(0.0, 0.0, 0.0));

    cv::drawContours(image, contours, -1, Scalar(255.0, 255.0, 255.0), -1);

    cv::imshow(strImageName, image);
}

int HOGDescriptorGenerator(HOGDescriptor hog, vector<float>* descriptor, int type, Mat plate) {

    vector<Mat> plate_list;

    switch (type) {
    case 0://Calculo de descriptores de Falsos
        for (int i = 1; i < 7; ++i) {
            Mat img = imread("Falso" + to_string(i) + ".PNG");
            cvtColor(img, img, COLOR_BGR2GRAY);
            resize(img, img, Size(128, 64));
            cv::rotate(img, img, ROTATE_90_CLOCKWISE);
            //img.convertTo(img, CV_32F, 1 / 255.0);
            plate_list.push_back(img);
        }
    case 1://Calculo de descriptores de Verdaderos
        for (int i = 1; i < 10; ++i) {
            Mat img = imread("True" + to_string(i) + ".PNG");
            cvtColor(img, img, COLOR_BGR2GRAY);
            resize(img, img, Size(128, 64));
            cv::rotate(img, img, ROTATE_90_CLOCKWISE);
            //img.convertTo(img, CV_32F, 1 / 255.0);
            plate_list.push_back(img);
        }
    default://Calculo de descriptores de la imagen 
        if (plate.empty()) return -1;
        cvtColor(plate, plate, COLOR_BGR2GRAY);
        resize(plate, plate, Size(128, 64));
        cv::rotate(plate, plate, ROTATE_90_CLOCKWISE);
        plate_list.push_back(plate);
    }

    vector<Point> loc;
    vector<float> desc((*descriptor).size());

    for (int i = 0; i < plate_list.size(); i++) {
        hog.compute(plate_list[i], desc, plate_list[0].size(), Size(0, 0), loc);
        if (i == 0) {
            (*descriptor) = desc;
        }
        else {
            transform((*descriptor).begin(), (*descriptor).end(), desc.begin(), (*descriptor).begin(), std::plus<float>());
        }
    }

    for (int i = 0; i < (*descriptor).size(); ++i) {
        (*descriptor)[i] = (*descriptor)[i] / plate_list.size();
    }

    return 0;
}

vector<float> HOGDescriptorGenerator(char type, Mat plate) {

    vector<Mat> plate_list;
    HOGDescriptor hog;



    if (type == 0) {
        for (int i = 1; i < 11; ++i) {
            Mat img = imread("Falso" + to_string(i) + ".PNG");
            cvtColor(img, img, COLOR_BGR2GRAY);
            resize(img, img, Size(128, 64));
            cv::rotate(img, img, ROTATE_90_CLOCKWISE);
            //img.convertTo(img, CV_32F, 1 / 255.0);
            plate_list.push_back(img);
        }
    }
    else if (type == 1) {
        for (int i = 1; i < 12; ++i) {
            Mat img = imread("True" + to_string(i) + ".PNG");
            cvtColor(img, img, COLOR_BGR2GRAY);
            resize(img, img, Size(128, 64));
            cv::rotate(img, img, ROTATE_90_CLOCKWISE);
            //img.convertTo(img, CV_32F, 1 / 255.0);
            plate_list.push_back(img);
        }
    }
    else if (type == 2) {
        if (plate.empty()) return vector<float>();
        //cvtColor(plate, plate, COLOR_BGR2GRAY);
        resize(plate, plate, Size(128, 64));
        cv::rotate(plate, plate, ROTATE_90_CLOCKWISE);
        plate_list.push_back(plate);
    }
    else {
        cout << "ERROR TYPE" << endl;
    }

    vector<Point> loc;
    vector<float> descriptors;
    vector<float> desc_acum(descriptors.size());

    hog.winSize = plate_list[0].size();
    hog.blockSize = Size(hog.winSize.width / 4, hog.winSize.height / 4);
    hog.blockStride = Size(16, 32);
    hog.cellSize = Size(16, 32);


    for (int i = 0; i < plate_list.size(); i++) {
        hog.compute(plate_list[i], descriptors, Size(128, 64), Size(0, 0), loc);
        if (i == 0) {
            desc_acum = descriptors;
        }
        else {
            transform(desc_acum.begin(), desc_acum.end(), descriptors.begin(), desc_acum.begin(), std::plus<float>());
        }
    }

    for (int i = 0; i < desc_acum.size(); ++i) {
        desc_acum[i] = desc_acum[i] / plate_list.size();
    }

    return desc_acum;
}

void matchFramePlateToPlates(vector<Plate>& FramePlate, vector<Plate>& ExistingPlate) {

    for (auto& exPlate : ExistingPlate) {
        //Preparación de todas las patentes en el pasadas para comparación con las patentes del frame. 
        exPlate.isNewPlateOrMatch = false;
        exPlate.predictPosition();
    }

    for (auto& framePlate : FramePlate) {

        int idxBestMatch = 0;
        double MinDist = 999999.9;
        //For para encontrar la distancia minima entre la prediccion de todas las patentes a las patentes del frame
        for (int i = 0; i < ExistingPlate.size(); i++) {
            if (ExistingPlate[i].isTracked == true) {
                double plateDist = euclideanDistance(ExistingPlate[i].NextPos, framePlate.centerPos.back());

                if (plateDist < MinDist) {
                    MinDist = plateDist;
                    idxBestMatch = i;
                }
            }
        }

        if (MinDist < framePlate.DiagSize * 1.2) {
            addPlateToMatch(ExistingPlate, framePlate, idxBestMatch);
        }
        else {
            addNewPlate(ExistingPlate, framePlate);
        }
    }

    for (auto& exPlate : ExistingPlate) {

        if (exPlate.isNewPlateOrMatch == false) {
            exPlate.nFramesNoTracked++;
        }
        if (exPlate.nFramesNoTracked >= 5) {
            exPlate.isTracked = false;
        }
    }
}

void addPlateToMatch(vector<Plate>& ExistingPlate, Plate& FramePlate, int idx) {
    ExistingPlate[idx].Contour = FramePlate.Contour;
    ExistingPlate[idx].contRect = FramePlate.contRect;
    ExistingPlate[idx].centerPos.push_back(FramePlate.centerPos.back());
    ExistingPlate[idx].AspectRatio = FramePlate.AspectRatio;
    ExistingPlate[idx].DiagSize = FramePlate.DiagSize;

    ExistingPlate[idx].isTracked = true;
    ExistingPlate[idx].isNewPlateOrMatch = true;
}

void addNewPlate(vector<Plate>& ExistingPlate, Plate& FramePlate) {

    FramePlate.isNewPlateOrMatch = true;
    ExistingPlate.push_back(FramePlate);

}

double euclideanDistance(Point p1, Point p2) {

    int dx, dy;

    dx = abs(p1.x - p2.x);
    dy = abs(p1.y - p2.y);

    return sqrt(pow(dx, 2) + pow(dy, 2));
}


/*#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <vector>

using namespace cv;
using namespace std;

int BgSus(Mat , Mat* , float , Mat* , int );
vector<vector<Point>> FilterContours(vector<vector<Point>> , vector<Rect>*, int, int,int , int , int , int );
int HOGDescriptorGenerator(HOGDescriptor, vector<float>*, int, Mat);

int BgSus(Mat img_in, Mat* fg, float u, Mat* bg, int th) {
    if (img_in.empty()) return -1;
    Mat grayFrame;
    cvtColor(img_in, grayFrame, COLOR_BGR2GRAY);

    *bg = (*bg) * (1.0 - u) + u * grayFrame;
    *fg = grayFrame - (*bg);
    medianBlur(*fg, *fg, 3);
    imshow("fg", *fg);
    if (th > 15) { threshold(*fg, *fg, th, 255, THRESH_BINARY); }
    else { threshold(*fg, *fg, 40, 255, THRESH_BINARY); }

    erode(*fg, *fg, getStructuringElement(MORPH_RECT, Size(3, 3)));
    dilate(*fg, *fg, getStructuringElement(MORPH_RECT, Size(3, 3)));

    return 0;
}


//int FilterContours(vector<vector<Point>> cont, vector<Rect>* filter_plate, vector<vector<Point>>* filter_cont, int aspMin, int aspMax,
vector<vector<Point>> FilterContours(vector<vector<Point>> cont, vector<Rect>* filter_plate, int aspMin, int aspMax,
                    int widMin, int widMax, int hMin, int hMax) {

    if (cont.empty()) { return {}; }

    vector<vector<Point>> cnPoly(cont.size());
    vector<Rect> cnRect(cont.size());
    vector<vector<Point>> filter_cont;
    vector<int> idx_selected;

    for (int i = 0; i < cont.size(); i++) {
        //Generación de rectangulos que encierren los contornos
        approxPolyDP(cont[i], cnPoly[i], 1, true);
        cnRect[i] = boundingRect(cnPoly[i]);
        //Cálculo de caracteristicas de los contornos para próximo filtrado.
        float aspect = (float)cnRect[i].width / (float)cnRect[i].height;
        float faspMin = ((float)aspMin / 100.0);
        float faspMax = ((float)aspMax / 100.0);
        //Filtrado por aspect ratio y tamaño.
        if ((aspect > faspMin) & (aspect < faspMax)
            & (cnRect[i].width > widMin) & (cnRect[i].width < widMax)
            & (cnRect[i].height > hMin) & (cnRect[i].height < hMax)) {
            cout << "width: " << cnRect[i].width << " height: " << cnRect[i].height << " aspect: " << aspect << endl;
            cout << "aspect min " << faspMin << "aspect max " << faspMax << endl;
            idx_selected.push_back(i);//Guardar los indices de los contornos que cumplen con las restricciones.
        }
    }

    for (size_t i = 0; i < idx_selected.size(); i++) {
        filter_cont.push_back(cont.at(idx_selected[i]));
        (*filter_plate)[i] = cnRect[idx_selected[i]];
    }

    return filter_cont;

}



int HOGDescriptorGenerator(HOGDescriptor hog, vector<float>* descriptor, int type, Mat plate) {

    vector<Mat> plate_list;

    switch (type) {
    case 0://Calculo de descriptores de Falsos
        for (int i = 1; i < 7; ++i) {
            Mat img = imread("Falso" + to_string(i) + ".PNG");
            cvtColor(img, img, COLOR_BGR2GRAY);
            resize(img, img, Size(128, 64));
            cv::rotate(img, img, ROTATE_90_CLOCKWISE);
            //img.convertTo(img, CV_32F, 1 / 255.0);
            plate_list.push_back(img);
        }
    case 1://Calculo de descriptores de Verdaderos
        for (int i = 1; i < 10; ++i) {
            Mat img = imread("True" + to_string(i) + ".PNG");
            cvtColor(img, img, COLOR_BGR2GRAY);
            resize(img, img, Size(128, 64));
            cv::rotate(img, img, ROTATE_90_CLOCKWISE);
            //img.convertTo(img, CV_32F, 1 / 255.0);
            plate_list.push_back(img);
        }
    default://Calculo de descriptores de la imagen 
        if (plate.empty()) return -1;
        cvtColor(plate, plate, COLOR_BGR2GRAY);
        resize(plate, plate, Size(128, 64));
        cv::rotate(plate, plate, ROTATE_90_CLOCKWISE);
        plate_list.push_back(plate);
    }

    vector<Point> loc;
    vector<float> desc((*descriptor).size());

    for (int i = 0; i < plate_list.size(); i++) {
        hog.compute(plate_list[i], desc, plate_list[0].size(), Size(0, 0), loc);
        if (i == 0) {
            (*descriptor) = desc;
        }
        else {
            transform((*descriptor).begin(), (*descriptor).end(), desc.begin(), (*descriptor).begin(), std::plus<float>());
        }
    }

    for (int i = 0; i < (*descriptor).size(); ++i) {
        (*descriptor)[i] = (*descriptor)[i] / plate_list.size();
    }

    return 0;
}*/