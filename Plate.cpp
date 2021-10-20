#include "Plate.h"
#include <iostream>

Plate::Plate(std::vector<cv::Point> _cnt) {

	Contour = _cnt;

	contRect = cv::boundingRect(Contour);

	cv::Point _centerPos;

	_centerPos.x = (contRect.x + (contRect.x + contRect.width) ) / 2;
	_centerPos.y = (contRect.y + (contRect.y + contRect.height)) / 2;

	centerPos.push_back(_centerPos);

	AspectRatio = (float)contRect.width / (float)contRect.height;

	DiagSize = sqrt( pow(contRect.width, 2) + pow(contRect.height, 2) );
	

	isNewPlateOrMatch = true;
	isTracked = true;

	int nFramesNoTracked = 0;
}
//El tracker precide la posición con un promedio ponderado de las diferencias de las coordenadas del centro de todas las muestras.
void Plate::predictPosition() {
	int numPos = (int)centerPos.size();

	if (numPos == 1) {
		NextPos.x = centerPos.back().x;
		NextPos.y = centerPos.back().y;
	}
	else if(numPos == 2){
		int dx, dy;
		dx = centerPos.back().x - centerPos[0].x;
		dy = centerPos.back().y - centerPos[0].y;

		NextPos.x = centerPos.back().x + dx;
		NextPos.y = centerPos.back().y + dy;
	}
	else if (numPos == 3) {
		int dx, dy;
		int sumaPonderadaX = ((centerPos[2].x - centerPos[1].x) * 2) +
							((centerPos[1].x - centerPos[0].x) * 1);
		
		int sumaPonderadaY = ((centerPos[2].y - centerPos[1].y) * 2) +
							((centerPos[1].y - centerPos[0].y) * 1);

		dx = (int)round((float)sumaPonderadaX / 3.0);
		dy = (int)round((float)sumaPonderadaY / 3.0);

		NextPos.x = centerPos.back().x + dx;
		NextPos.y = centerPos.back().y + dy;
	}
	else if (numPos >= 4) {
		int dx, dy;
		int sumaPonderadaX = ((centerPos[numPos - 1].x - centerPos[numPos - 2].x) * 3) +
							((centerPos[numPos - 2].x - centerPos[numPos - 3].x) * 2) +
							((centerPos[numPos - 3].x - centerPos[numPos - 4].x) * 1);

		int sumaPonderadaY = ((centerPos[numPos - 1].y - centerPos[numPos - 2].y) * 3) +
							((centerPos[numPos - 2].y - centerPos[numPos - 3].y) * 2) +
							((centerPos[numPos - 3].y - centerPos[numPos - 4].y) * 1);

		dx = (int)round((float)sumaPonderadaX / 4.0);
		dy = (int)round((float)sumaPonderadaY / 4.0);

		NextPos.x = centerPos.back().x + dx;
		NextPos.y = centerPos.back().y + dy;
	}
	else {
		std::cout << "Error Numero de posiciones invalida" << std::endl;
	}
	
}