#ifndef _CH_H_
#define _CH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <conio.h>
#include <math.h>
#include <cuda.h>
#include <windows.h>
//#include <helper_cuda.h>
#include <cuda_runtime.h>
#define MAX_DEPTH       24 //Maximum nesting depth of dynamic parallelism is limited to 24
#define INSERTION_SORT  32

#define SIZE 1048576

using namespace std;

typedef struct coord{
	double x;
	double y;
}point_t;

void convex_hull(point_t *points, int npoints, point_t *out_hull, int *out_hullsize);
void sort_points(point_t *points, int npoints);
void runOnGPU(point_t *points, int npoints, point_t *out_hull, int *out_hullsize);
void quickSortOnCPU(point_t* points, int left, int right);

#endif
