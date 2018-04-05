#include "ConvexHull.h"

int main()
{
	int n_points, i;
	int out_hullsize;
	int rep=1;			// Number of repetitions
	point_t *points;
	point_t *out_hull;
	LARGE_INTEGER frequency, t1, t2;
	double elapsedTimeGPU[10], elapsedTimeCPU[10], avg_timeCPU, avg_timeGPU, std_dev;

	points = new point_t[SIZE];
	out_hull = new point_t[SIZE];
	out_hullsize = 0;
	for (i = 0; i < 2; i++)
	{
		out_hull[i].x = 0;
		out_hull[i].y = 0;
	}

	//cout << "Enter the number of points:";
	//cin >> n_points;
	n_points = SIZE;
	
	ifstream file;
	file.open("Cvx_Pts1048576.txt");
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
	//cout << "The points are:";
	//for (i = 0; i < n_points; i++)
	//{
	//	cout << points[i].x << "\t" << points[i].y << endl;
	//}

	//*********************************************************CPU part begins****************************************************************
	avg_timeCPU = 0;
	// get ticks per second
	QueryPerformanceFrequency(&frequency);
	for (i = 0; i < rep; i++)
	{
		//Selection sort on CPU
		//sort_points(points, n_points);
		quickSortOnCPU(points, 0, n_points - 1);

		/*cout << "Sorted points on CPU are:";
		for (i = 0; i < n_points; i++)
		{
		cout << points[i].x << "\t" << points[i].y << endl;
		}*/
		// Start timer
		QueryPerformanceCounter(&t1);

		//Convex hull on CPU
		convex_hull(points, n_points, out_hull, &out_hullsize);

		// Stop timer
		QueryPerformanceCounter(&t2);
		// Compute the elapsed time in millisec
		elapsedTimeCPU[i] = (t2.QuadPart - t1.QuadPart) * 1000.0 / frequency.QuadPart;
		avg_timeCPU = avg_timeCPU + elapsedTimeCPU[i];
	}
	// Average time
	avg_timeCPU = avg_timeCPU / rep;
	// Compute standard deviation
	std_dev = 0;
	for (i = 0; i < rep; i++)
	{
		std_dev = std_dev + pow((elapsedTimeCPU[i] - avg_timeCPU), 2);
	}
	std_dev = sqrt(std_dev / rep);

	cout << "The convex hull calculated by CPU is:" << endl;
	for (i = 0; i < out_hullsize; i++)
	{
		cout << out_hull[i].x << "\t" << out_hull[i].y << "\n";
	}
	
	cout << "The average time taken by the CPU is: " << avg_timeCPU << " ms with a standard deviation of " << std_dev << " ms " << endl;
	//*********************************************************CPU part ends******************************************************************

	//***********************************************************GPU Part begins**************************************************************
	// Run on GPU
	runOnGPU(points, n_points, out_hull, &out_hullsize);

	//***********************************************************GPU part ends****************************************************************
	// deallocate memory
	delete points;
	delete out_hull;
	cout << "Press any key to exit";
	_getche();
}