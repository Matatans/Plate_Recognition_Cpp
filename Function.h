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

//Prototipos de función.

//Funciones variadas.
void PrintContours(cv::Size, std::vector<std::vector<cv::Point> >, std::string);
double euclideanDistance(cv::Point, cv::Point);

//Funciones para detección via clasificador de media con HOG features.
int HOGDescriptorGenerator(cv::HOGDescriptor, std::vector<float>*, int, cv::Mat);
std::vector<float> HOGDescriptorGenerator(char, cv::Mat);
float isPlate(std::vector<float>, std::vector<float>, std::vector<float>);

//Funciones para Match y manejo de vector de patentes.
void addPlateToMatch(std::vector<Plate>&, Plate&, int);
void addNewPlate(std::vector<Plate>&, Plate&);
void matchFramePlateToPlates(std::vector<Plate>&, std::vector<Plate>&);
void printPlateInfo(std::vector<Plate>&, cv::Mat&);