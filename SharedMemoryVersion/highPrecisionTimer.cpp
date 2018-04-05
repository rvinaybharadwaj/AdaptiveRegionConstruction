//Timer
#ifdef _MSC_VER

#include <windows.h>

class highPrecisionTimer
{
	LARGE_INTEGER starttime;
public:
	void start();
	double end(); 
};

void highPrecisionTimer::start()
{
	QueryPerformanceCounter(&starttime);
}

double highPrecisionTimer::end()
{
	LARGE_INTEGER endtime,freq;
	QueryPerformanceCounter(&endtime);
	QueryPerformanceFrequency(&freq);

	return ((double)(endtime.QuadPart-starttime.QuadPart))/((double)(freq.QuadPart/1000.0));
}


#else
#include <sys/time.h>

class highPrecisionTimer
{
	struct timeval starttime;
public:
	void start();
	double end(); 
};

void highPrecisionTimer::start()
{
	gettimeofday(&starttime,0);
}

double highPrecisionTimer::end()
{
	struct timeval endtime;
	gettimeofday(&endtime,0);

	return (endtime.tv_sec - starttime.tv_sec)*1000.0 + (endtime.tv_usec - starttime.tv_usec)/1000.0;
}


#endif
