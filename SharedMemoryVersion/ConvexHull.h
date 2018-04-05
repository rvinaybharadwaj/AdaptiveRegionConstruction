#ifndef _CH_H_
#define _CH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <conio.h>
#include <math.h>
#include <cuda.h>
#include <windows.h>
//#include <string.h>
//#include <stdlib.h>
//#include <helper_cuda.h>
#include <cuda_runtime.h>
#include "ThreadTimer.h"
#define MAX_DEPTH       24 //Maximum nesting depth of dynamic parallelism is limited to 24
#define INSERTION_SORT  32

#define SIZE 1048576
#define THREADS_BLOCK 128	//can go only till 512 threads/block due to shared memory limitations
#define NUM_BLOCKS 2048
#define SMEM SIZE/NUM_BLOCKS
#define ELEM_THREAD SMEM/THREADS_BLOCK

using namespace std;

typedef struct coord{
	float x;
	float y;
}point_t;

void convex_hull(point_t *points, int npoints, point_t *out_hull_cpu, int *out_hullsize_cpu);
void sort_points(point_t *points, int npoints);
void runOnGPU(point_t *points, int npoints, point_t *out_hull_gpu, int *out_hullsize_gpu, double **et);
void quickSortOnCPU(point_t* points, int left, int right);
void verify_convexHull(point_t *out_hull_cpu, int *out_hullsize_cpu, point_t *out_hull_gpu, int *out_hullsize_gpu);

#endif
