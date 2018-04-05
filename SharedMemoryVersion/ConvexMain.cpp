#include "ConvexHull.h"

int main()
{
	int n_points, i;
	int out_hullsize_cpu, out_hullsize_gpu;
	point_t *points;
	point_t *out_hull_cpu, *out_hull_gpu;
	//LARGE_INTEGER frequency, t1, t2;
	//double elapsedTimeGPU[10], elapsedTimeCPU[10], avg_timeCPU, avg_timeGPU, std_dev;

	//Performance Measurement Data
	const int num_it = 1;
	const int num_alg = 5;
	double **et = NULL;

	et = ThreadTimer::SetupElapsedTimeMatrix(num_it, num_alg);
	ThreadTimer tt;

	//points = new point_t[SIZE];
	cudaError_t status = cudaMallocHost((void**)&points, SIZE*sizeof(point_t));
	if (status != cudaSuccess)
		printf("Error allocating pinned host memory\n");

	out_hull_cpu = new point_t[SIZE];
	out_hull_gpu = new point_t[SIZE];
	out_hullsize_cpu = 0;
	for (i = 0; i < 2; i++)
	{
		out_hull_cpu[i].x = 0;
		out_hull_cpu[i].y = 0;
	}

	//cout << "Enter the number of points:";
	//cin >> n_points;
	n_points = SIZE;
	
	ifstream file;
	char sbuff[10];
	char str[30];
	_itoa_s(SIZE, sbuff, 10, 10);
	strcpy_s(str, "Cvx_Pts");
	strcat_s(str, sbuff);
	strcat_s(str, ".txt");
	file.open(str);
	/*file.open("Cvx_Pts2097152.txt");*/
	if (!file)
	{
		cout << "Cannot open file.\n";
		_getche();
		exit(0);
	}

	/*cout << "Enter the points:";
	for (i = 0; i < n_points; i++)
	{
	cin >> points[i].x >> points[i].y;
	}*/


	// Read data from text file, convert it to float and store it
	for (i = 0; i < n_points; i++)
	{
		string line;
		getline(file, line);
		istringstream linestream(line);
		string item;
		int itemcount = 0;
		while (getline(linestream, item, '\t'))
		{
			itemcount++;

			if (itemcount == 1)
			{
				char *cstr = new char[item.length() + 1];	//allocate memory for character array
				strcpy_s(cstr, item.length() + 1, item.c_str());	//copy second item to char array
				points[i].x = atof(cstr);	//convert char to float
				delete[] cstr;	//release memory
			}

			if (itemcount == 2)
			{
				char *cstr = new char[item.length() + 1];	//allocate memory for character array
				strcpy_s(cstr, item.length() + 1, item.c_str());	//copy second item to char array
				points[i].y = atof(cstr);	//convert character to float
				delete[] cstr;	//release memory
				itemcount = 0;	//reset item counter
			}
		}
	}
	file.close();
	/*cout << "The points are:";
	for (i = 0; i < n_points; i++)
	{
		cout << points[i].x << "\t" << points[i].y << endl;
	}*/

	//*********************************************************CPU part begins****************************************************************
	//avg_timeCPU = 0;
	// get ticks per second
	//QueryPerformanceFrequency(&frequency);
	for (i = 0; i < num_it; i++)
	{
		//Selection sort on CPU
		//sort_points(points, n_points);
		quickSortOnCPU(points, 0, n_points - 1);
	
		/*FILE *fp;
		fopen_s(&fp, "Sort100_Ufm.txt", "w");
		if (fp == NULL) {
			fprintf(stderr, "Can't open input file!\n");
			exit(1);
		}
		else
		{
			for (int i = 0; i < n_points; i++)
			{
				fprintf(fp, "%f\t%f\n", points[i].x, points[i].y);
			}
		}*/

		/*cout << "Sorted points on CPU are:";
		for (i = 0; i < n_points; i++)
		{
		cout << points[i].x << "\t" << points[i].y << endl;
		}*/
		// Start timer
		//QueryPerformanceCounter(&t1);
		tt.Start();
		//Convex hull on CPU
		convex_hull(points, n_points, out_hull_cpu, &out_hullsize_cpu);

		// Stop timer
		tt.Stop();
		et[i][0] = tt.GetElapsedTime();
		//QueryPerformanceCounter(&t2);
		// Compute the elapsed time in millisec
		/*elapsedTimeCPU[i] = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		avg_timeCPU = avg_timeCPU + elapsedTimeCPU[i];*/
	}
	//// Average time
	//avg_timeCPU = avg_timeCPU / num_it;
	//// Compute standard deviation
	//std_dev = 0;
	//for (i = 0; i < num_it; i++)
	//{
	//	std_dev = std_dev + pow((elapsedTimeCPU[i] - avg_timeCPU), 2);
	//}
	//std_dev = sqrt(std_dev / num_it);

	//cout << "The size of convex hull on CPU is" << out_hullsize_cpu << endl;
	/*cout << "The convex hull calculated by CPU is:" << endl;
	for (i = 0; i < out_hullsize_cpu; i++)
	{
		cout << out_hull_cpu[i].x << "\t" << out_hull_cpu[i].y << "\n";
	}*/
	
	/*cout << "The average time taken by the CPU is: " << avg_timeCPU << " ms with a standard deviation of " << std_dev << " ms " << endl;*/
	//*********************************************************CPU part ends*******************************************************************

	//***********************************************************GPU Part begins**************************************************************
	// Run on GPU
	runOnGPU(points, n_points, out_hull_gpu, &out_hullsize_gpu, et);

	//***********************************************************GPU part ends****************************************************************
	//verification
	verify_convexHull(out_hull_cpu, &out_hullsize_cpu, out_hull_gpu, &out_hullsize_gpu);
	// deallocate memory
	//delete points;
	cudaFreeHost(points);
	delete out_hull_cpu;
	delete out_hull_gpu;
	cout << "Press any key to exit";
	_getche();
}