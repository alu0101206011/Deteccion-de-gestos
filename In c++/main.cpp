#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;


double angle(Point s, Point e, Point f) {
	double v1[2], v2[2];
	v1[0] = s.x - f.x;
	v1[1] = s.y - f.y;
	v2[0] = e.x - f.x;
	v2[1] = e.x - f.x;
	double ang1 = atan2(v1[1], v1[0]);
	double ang2 = atan2(v2[1], v2[0]);

	double ang = ang1 - ang2;
	if (ang > CV_PI) ang -= 2 * CV_PI;
	if (ang < -CV_PI) ang += 2 * CV_PI;

	return ang * 180 / CV_PI;
}

size_t maxSizeContours(vector<vector<Point> > contours) {
	unsigned max = contours[0].size();
	size_t contourNumber = 0;
	for (size_t i = 0; i < contours.size(); i++)
		if (contours[i].size() > max) 
			contourNumber = i;
	return contourNumber;
}


int main(int argc, char* argv[]) {
	Mat frame, roi, fgMask;
	vector<vector<Point> > contours;
	VideoCapture cap;
	cap.open(0);

	Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2();  // Para quitar el fondo

	if (!cap.isOpened()) {
		printf("Error opening cam\n");
		return -1;
	}
	namedWindow("Frame");
	namedWindow("ROI");  // Ventana con la mano
	namedWindow("Foreground Mask");

	int l_rate = -1;

	Rect rect(400, 100, 200, 200);
	int numDefects = 0;
	while (true) {

		cap>>frame;
		flip(frame, frame, 1);
		
		frame(rect).copyTo(roi);  // Muestra todo lo que tenga el rectangulo que se encuentra en frame en ROI

		rectangle(frame, rect, Scalar(255, 0, 0));

		pBackSub->apply(roi, fgMask, l_rate);  // Aplicamos la mascara al fondo

		findContours(fgMask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		vector<int> hull;
		if (contours.size() != 0) {
			size_t i = maxSizeContours(contours);  // ES PELIGROSO SOLO TENER EN CUENTA UN CONTORNO

			drawContours(roi, contours, i, Scalar(0,255,0), 2);

			Rect boundRect = boundingRect(contours[i]);
			rectangle(roi, boundRect, Scalar(0,0,255), 1);

			const int heightRect = boundRect.size().height;

			convexHull(contours[i], hull, false, false);
			sort(hull.begin(), hull.end(), greater <int>());

			std::cout << "Area: " << contourArea(contours[i]) << "\n";

			vector<Vec4i> defects;
			convexityDefects(contours[i], hull, defects);
			for (int j = 0; j < defects.size(); j++) {
				Point s = contours[i][defects[j][0]];  // Punto inicial
				Point e = contours[i][defects[j][1]];  // Punto final
				Point f = contours[i][defects[j][2]];  // Punto más lejano
				float depth = (float)defects[j][3] / 256.0;  // Distancia en pixeles desde la malla hasta el punto más lejano a ella
				double ang = angle(s, e, f);
				if (0.3*heightRect < depth && ang < 90) {
					circle(roi, f, 5, Scalar(0, 0, 255), -1);
					line(roi, s, e, Scalar(255, 0, 0), 2);
					numDefects++;
				}
			}

	  double perimeter = arcLength(contours[i]);
	  std::cout << "Perimeter: " << arcLength(contours[i],True) << std::endl;
	  std::cout << "Area: " << contourArea(contours[i]) << std::endl;
	  std::cout << "Height: " << boundRect.size().height << std::endl;


      int substract = boundRect.size().height - boundRect.size().width;
			// Gestos
			if (numDefects == 2 && contourArea(contours[i])/boundRect.size().height < 100 && contourArea(contours[i])/boundRect.size().height > 75) {
				string text = "Spock"; 
				Point textPoint(frame.cols - getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 3, 0).width, 
				                frame.rows - getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 3, 0).height);
				putText(frame, text, textPoint, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar::all(255), 3, 8);
			}
            else if(substract < boundRect.size().height * 0.3 && !numDefects) { // Contar dedos
				string text = "Dedos levantados: " + to_string(numDefects);
				Point textPoint(frame.cols - getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 3, 0).width, 
				                frame.rows - getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 3, 0).height);
				putText(frame, text, textPoint, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar::all(255), 3, 8);
			} else {
				string text = "Dedos levantados: " + to_string(numDefects+1);
				Point textPoint(frame.cols - getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 3, 0).width, 
				                frame.rows - getTextSize(text, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 3, 0).height);
				putText(frame, text, textPoint, FONT_HERSHEY_SCRIPT_SIMPLEX, 1, Scalar::all(255), 3, 8);
			}
			numDefects = 0;
		}
		imshow("Frame",frame);
		imshow("ROI", roi);
		imshow("Foreground Mask", fgMask);

		int c = waitKey(40);
		if ((char)c == 'v') {
			l_rate = 0;
		}
		if ((char)c == 'q') break;
	}
	cap.release();
	destroyAllWindows();
}