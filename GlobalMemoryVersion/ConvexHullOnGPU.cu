#include "ConvexHull.h"

/* Three points are a counter-clockwise turn if ccw > 0, clockwise if
ccw < 0, and collinear if ccw = 0 because ccw is a determinant that
gives the signed area of the triangle formed by p1, p2 and p3.
*/
__device__ float ccw(point_t* p1, point_t* p2, point_t* p3)
{
	return (p2->x - p1->x)*(p3->y - p1->y) - (p2->y - p1->y)*(p3->x - p1->x);
}

__global__ void lowerHullonGPU(point_t* points, int npoints, point_t* out_hull1, int* out_hullsize1)
{
	//Calculate global index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//Number of elements handled by each thread
	int elements_thread = npoints / (blockDim.x * gridDim.x);
	//each thread has a different offset
	int offset = idx*elements_thread;
	//initialize the count to zero
	out_hullsize1[idx] = 0;

	// lower hull 
	for (int i = offset; i < offset+elements_thread; ++i) 
	{
		/* while L contains at least two points and the sequence of last two points
		of L and the point P[i] does not make a counter-clockwise turn:
		remove the last point from L, append P[i] to L
		*/
		while (out_hullsize1[idx] >= 2 && ccw(&out_hull1[offset + out_hullsize1[idx] - 2], &out_hull1[offset + out_hullsize1[idx] - 1], &points[i]) <= 0)
		{
			--out_hullsize1[idx];
		}
		out_hull1[offset + (out_hullsize1[idx]++)] = points[i];
	}
	//out_hullsize1[idx]++;
}

__global__ void upperHullonGPU(point_t* points, int npoints, point_t* out_hull2, int* out_hullsize2)
{
	//Calculate global index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	//Number of elements handled by each thread
	int elements_thread = npoints / (blockDim.x * gridDim.x);
	//Each thread has a different offset
	int offset = idx*elements_thread;
	//initialize the count to zero
	out_hullsize2[idx] = 0;

	//out_hull2[0] = points[npoints - 1]; //first point for upper hull	

	// upper hull 
	// remove the last point of each list (it's the same as the first 
	// point of the other list, so start from npoints-2)
	/* while U contains at least two points and the sequence of last two points
	of U and the point P[i] does not make a counter-clockwise turn:
	remove the last point from U, append P[i] to U
	*/
	//t=k+1 to begin the upper hull - make a turn by considering the immediate point
	for (int i = npoints-offset-1; i >= npoints-offset-elements_thread; --i)
	{	
		while (out_hullsize2[idx] >= 2 && ccw(&out_hull2[offset + out_hullsize2[idx] - 2], &out_hull2[offset + out_hullsize2[idx] - 1], &points[i]) <= 0)
		{
			--out_hullsize2[idx];
		}
		out_hull2[offset + (out_hullsize2[idx]++)] = points[i];
	}
	//out_hullsize2[idx]++;
}

__global__ void mergeLowerHull(point_t *hull_part, int *part_size, int *i, int *j, int npoints)
{
	//Calculate the global index
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//Number of threads
	int num_threads = blockDim.x*gridDim.x;

	//Number of elements seen by each thread, but size of individual convex hulls may be smaller
	int elements_thread = npoints / num_threads;

	if (idx < num_threads - 1)
	{
		//Consider two sub-hulls in each thread
		int offset = idx*elements_thread;
		int next_offset = (idx + 1)*elements_thread;

		//Loop conditions
		bool condition1 = true, condition2 = true;

		//Points to construct tangent
		point_t *a, *b;

		//For lower hull
		a = &hull_part[offset + part_size[idx] - 1];//right most part of left hull
		b = &hull_part[next_offset];//left most part of right hull

		//Construct tangent
		while (condition1 || condition2)
		{
			condition1 = false;
			condition2 = false;
			while (ccw(b, a, (a - 1)) > 0)
			{
				a = (a - 1);
				i[idx]++;
				condition1 = true;
			}
			while (ccw(a, b, (b + 1)) <= 0)
			{
				b = (b + 1);
				j[idx]++;
				condition2 = true;
			}
		}
		//printf("idx = %d and i = %d and j = %d\n", idx, i[idx], j[idx]);
	}
}

__global__ void mergeUpperHull(point_t *hull_part, int *part_size, int *i, int *j, int npoints)
{
	//Calculate the global index
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int num_threads = blockDim.x * gridDim.x;
	//Number of elements seen by each thread, but size of individual convex hulls is smaller
	int elements_thread = npoints / num_threads;

	if (idx < num_threads - 1)
	{
		//Consider two sub-hulls in each thread
		int offset = idx*elements_thread;
		int next_offset = (idx + 1)*elements_thread;

		//Loop conditions
		bool condition1 = true, condition2 = true;

		//Points to construct tangent
		point_t *a, *b;

		//For upper hull
		a = &hull_part[offset + part_size[idx] - 1];//left most part of right hull
		b = &hull_part[next_offset];//right most part of left hull

		//Construct tangent
		while (condition1 || condition2)
		{
			condition1 = false;
			condition2 = false;
			while (ccw(b, a, (a - 1)) > 0)
			{
				a = (a - 1);
				i[idx]++;
				condition1 = true;
			}
			while (ccw(a, b, (b + 1)) <= 0)
			{
				b = (b + 1);
				j[idx]++;
				condition2 = true;
			}
		}
		//printf("idx = %d and i = %d and j = %d\n", idx, i[idx], j[idx]);
	}
}

// Selection sort used when depth gets too big or the number of elements drops
// below a threshold.
__device__ void selection_sort(point_t *points, int left, int right)
{
	for (int i = left; i <= right; ++i)
	{
		point_t min_val = points[i];
		int min_idx = i;

		// Find the smallest value in the range [left, right].
		for (int j = i + 1; j <= right; ++j)
		{
			point_t temp = points[j];
			if ((temp.x < min_val.x) || ((temp.x == min_val.x) && (temp.y < min_val.y)))
			{
				min_idx = j;
				min_val = temp;
			}
		}

		// Swap the values.
		if (i != min_idx)
		{
			points[min_idx] = points[i];
			points[i] = min_val;
		}
	}
}


// Very basic quicksort algorithm, recursively launching the next level.
__global__ void quickSortOnGPU(point_t *points, int left, int right, int depth)
{
	// If we're too deep or there are few elements left, we use an insertion sort...
	if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT)
	{
		selection_sort(points, left, right);
		return;
	}

	point_t *lptr = points + left;
	point_t *rptr = points + right;
	point_t  pivot = points[(left + right) / 2];

	// Do the partitioning.
	while (lptr <= rptr)
	{
		// Find the next left- and right-hand values to swap
		point_t lval = *lptr;
		point_t rval = *rptr;

		// Move the left pointer as long as the pointed element is smaller than the pivot.
		while ((lval.x < pivot.x) || ((lval.x == pivot.x) && (lval.y < pivot.y)))
		{
			lptr++;
			lval = *lptr;
		}

		// Move the right pointer as long as the pointed element is larger than the pivot.		
		while ((rval.x > pivot.x) || ((rval.x == pivot.x) && (rval.y > pivot.y)))
		{
			rptr--;
			rval = *rptr;
		}

		// If the swap points are valid, do the swap!
		if (lptr <= rptr)
		{
			*lptr++ = rval;
			*rptr-- = lval;
		}
	}

	// Now the recursive part
	int nright = rptr - points;
	int nleft = lptr - points;

	// Launch a new block to sort the left part.
	if (left < (rptr - points))
	{
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		quickSortOnGPU <<< 1, 1, 0, s >>>(points, left, nright, depth + 1);
		cudaStreamDestroy(s);
	}

	// Launch a new block to sort the right part.
	if ((lptr - points) < right)
	{
		cudaStream_t s1;
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		quickSortOnGPU <<< 1, 1, 0, s1 >>>(points, nleft, right, depth + 1);
		cudaStreamDestroy(s1);
	}
}

// Call the quicksort kernel from the host.
void runOnGPU(point_t *points, int npoints, point_t *out_hull, int *out_hullsize)
{
	// Get device properties
	int device_count = 0, device = -1;
	cudaGetDeviceCount(&device_count);
	for (int i = 0; i < device_count; ++i)
	{
		cudaDeviceProp properties; //instance of the structure
		cudaGetDeviceProperties(&properties, i);
		if (properties.major > 3 || (properties.major == 3 && properties.minor >= 5))
		{
			device = i;
			cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
			break;
		}
		cout << "GPU " << i << " (" << properties.name << ") does not support CUDA Dynamic Parallelism" << std::endl;
	}
	if (device == -1)
	{
		cerr << "Quicksort requires GPU devices with compute SM 3.5 or higher.  Exiting..." << std::endl;
		exit(EXIT_SUCCESS);
	}
	cudaSetDevice(device);//Set the device to run computations

	// Allocate GPU memory.
	point_t *dev_points;
	cudaMalloc((void **)&dev_points, npoints * sizeof(point_t));

	// Copy data to device memory
	cudaMemcpy(dev_points, points, npoints * sizeof(point_t), cudaMemcpyHostToDevice);

	// Prepare Cuda Dynamic Program for the maximum depth of MAX_DEPTH.
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

	// Launch on device
	int left = 0;
	int right = npoints - 1;
	//cout << "Launching kernel on the GPU" << endl;

	// Launch CUDA kernel to sort the points
	quickSortOnGPU <<< 1, 1 >>> (dev_points, left, right, 0);
	cudaDeviceSynchronize(); // Blocks until the device has completed all preceding requested tasks 

	/*cudaMemcpy(points, dev_points, npoints * sizeof(point_t), cudaMemcpyDeviceToHost);
	printf("The sorted points are:");
	for (int i = 0; i < npoints; i++)
	{
		printf("%d and %d\n", points[i].x, points[i].y);
	}*/

	// Kernel parameters
	int threads_block = 256;
	int num_blocks = 1024;
	int num_threads = threads_block*num_blocks;

	// Convex hull parameters
	int *out_hullSizeLower, *out_hullSizeUpper, *dev_out_hullSizeLower, *dev_out_hullSizeUpper;
	int *mergeLowerEnd, *mergeLowerBegin, *dev_mergeLowerEnd, *dev_mergeLowerBegin;
	int *mergeUpperEnd, *mergeUpperBegin, *dev_mergeUpperEnd, *dev_mergeUpperBegin;
	point_t *out_hullLower, *out_hullUpper, *dev_out_hullLower, *dev_out_hullUpper;

	//allocate memory on CPU
	//out_hullLower = new point_t[SIZE];
	//out_hullUpper = new point_t[SIZE];
	out_hullLower = (point_t*)calloc(SIZE,sizeof(point_t));
	out_hullUpper = (point_t*)calloc(SIZE,sizeof(point_t));

	out_hullSizeLower = new int[num_threads];
	out_hullSizeUpper = new int[num_threads];
	mergeLowerEnd = (int*)calloc(num_threads - 1, sizeof(int));
	mergeLowerBegin = (int*)calloc(num_threads - 1, sizeof(int));
	mergeUpperEnd = (int*)calloc(num_threads - 1, sizeof(int));
	mergeUpperBegin = (int*)calloc(num_threads - 1, sizeof(int));

	//allocate memory on GPU
	cudaMalloc((void **)&dev_out_hullLower, SIZE * sizeof(point_t));
	cudaMalloc((void **)&dev_out_hullUpper, SIZE * sizeof(point_t));
	cudaMalloc((void **)&dev_out_hullSizeLower, num_threads*sizeof(int));
	cudaMalloc((void **)&dev_out_hullSizeUpper, num_threads*sizeof(int));
	cudaMalloc((void **)&dev_mergeLowerEnd, (num_threads - 1)*sizeof(int));
	cudaMalloc((void **)&dev_mergeLowerBegin, (num_threads - 1)*sizeof(int));
	cudaMalloc((void **)&dev_mergeUpperEnd, (num_threads - 1)*sizeof(int));
	cudaMalloc((void **)&dev_mergeUpperBegin, (num_threads - 1)*sizeof(int));

	//Create events to measure time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Get starting tick
	cudaEventRecord(start);
	//copy host memory to device memory
	cudaMemcpy(dev_out_hullLower, out_hullLower, SIZE * sizeof(point_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out_hullUpper, out_hullUpper, SIZE * sizeof(point_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out_hullSizeLower, out_hullSizeLower, num_threads*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_out_hullSizeUpper, out_hullSizeUpper, num_threads*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mergeLowerEnd, mergeLowerEnd, (num_threads-1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mergeLowerBegin, mergeLowerBegin, (num_threads-1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mergeUpperEnd, mergeUpperEnd, (num_threads - 1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mergeUpperBegin, mergeUpperBegin, (num_threads - 1)*sizeof(int), cudaMemcpyHostToDevice);

	//Create streams 
	cudaStream_t st1, st2;
	cudaStreamCreateWithFlags(&st1, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking);

	// call the cuda kernels to compute convex hull
	lowerHullonGPU <<< num_blocks, threads_block, 0, st1 >>> (dev_points, npoints, dev_out_hullLower, dev_out_hullSizeLower);
	upperHullonGPU <<< num_blocks, threads_block, 0, st2 >>> (dev_points, npoints, dev_out_hullUpper, dev_out_hullSizeUpper);
	cudaDeviceSynchronize(); // Blocks until the device has completed all preceding requested tasks	


	cudaMemcpy(out_hullLower, dev_out_hullLower, npoints * sizeof(point_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_hullUpper, dev_out_hullUpper, npoints * sizeof(point_t), cudaMemcpyDeviceToHost);
	/*cout << "The lower hull computed on GPU is: " << endl;
	for (int i = 0; i < npoints; i++)
	{
		cout << out_hullLower[i].x << "\t" << out_hullLower[i].y << endl;
	}
	cout << "The upper hull computed on GPU is: " << endl;
	for (int i = 0; i < npoints; i++)
	{
		cout << out_hullUpper[i].x << "\t" << out_hullUpper[i].y << endl;
	}*/

	mergeLowerHull <<< num_blocks, threads_block, 0, st1 >>> (dev_out_hullLower, dev_out_hullSizeLower, dev_mergeLowerEnd, dev_mergeLowerBegin, npoints);
	mergeUpperHull <<< num_blocks, threads_block, 0, st2 >>> (dev_out_hullUpper, dev_out_hullSizeUpper, dev_mergeUpperEnd, dev_mergeUpperBegin, npoints);
	cudaDeviceSynchronize();

	//Destroy streams
	cudaStreamDestroy(st1);
	cudaStreamDestroy(st2);

	//Copy device memory to host memory
	cudaMemcpy(out_hullSizeLower, dev_out_hullSizeLower, num_threads*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_hullSizeUpper, dev_out_hullSizeUpper, num_threads*sizeof(int), cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < num_threads; i++)
	{
		cout << "The size of the Lower hull is: " << out_hullSizeLower[i] << endl;
		cout << "The size of the Upper hull is: " << out_hullSizeUpper[i] << endl;
	}*/

	/*cudaMemcpy(out_hullLower, dev_out_hullLower, *out_hullSizeLower * sizeof(point_t), cudaMemcpyDeviceToHost);
	cudaMemcpy(out_hullUpper, dev_out_hullUpper, *out_hullSizeUpper * sizeof(point_t), cudaMemcpyDeviceToHost);*/	
	cudaMemcpy(mergeLowerEnd, dev_mergeLowerEnd, (num_threads - 1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(mergeLowerBegin, dev_mergeLowerBegin, (num_threads - 1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(mergeUpperEnd, dev_mergeUpperEnd, (num_threads - 1)*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(mergeUpperBegin, dev_mergeUpperBegin, (num_threads - 1)*sizeof(int), cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < num_threads - 1; i++)
	{
		cout << "mergeLowerEnd = " << mergeLowerEnd[i] << endl;
		cout << "mergeLowerBegin = " << mergeLowerBegin[i] << endl;
		cout << "mergeUpperEnd = " << mergeUpperEnd[i] << endl;
		cout << "mergeUpperBegin = " << mergeUpperBegin[i] << endl;
	}*/

	//Write merged hull into an array
	int k = 0;
	int offset = npoints / num_threads;
	//Writing lower hull
	for (int i = 0; i < num_threads; i++)
	{
		if (i == 0)
		{
			for (int j = 0; j < (out_hullSizeLower[i] - mergeLowerEnd[i]); j++)
			{
				out_hull[k] = out_hullLower[j];
				k++;
			}
		}
		else if (i > 0 && i < num_threads-1)
		{
			for (int j = mergeLowerBegin[i-1]; j < (out_hullSizeLower[i] - mergeLowerEnd[i]); j++)
			{
				out_hull[k] = out_hullLower[i*offset + j];
				k++;
			}
		}
		else
		{
			for (int j = mergeLowerBegin[i-1]; j < out_hullSizeLower[i]; j++)
			{
				out_hull[k] = out_hullLower[i*offset + j];
				k++;
			}
		}
	}
	cout << "The lower convex hull computed on GPU is: " << endl;
	for (int i = 0; i < k; i++)
	{
		cout << out_hull[i].x << "\t" << out_hull[i].y << endl;
	}
	//Writing upper hull
	for (int i = 0; i < num_threads; i++)
	{
		if (i == 0)
		{		
			for (int j = 0; j < (out_hullSizeUpper[i] - mergeUpperEnd[i]); j++)
			{
				out_hull[k] = out_hullUpper[j];
				k++;
			}
		}
		else if (i > 0 && i < num_threads - 1)
		{
			for (int j = mergeUpperBegin[i-1]; j < (out_hullSizeUpper[i] - mergeUpperEnd[i]); j++)
			{
				out_hull[k] = out_hullUpper[i*offset + j];
				k++;
			}
		}
		else
		{
			for (int j = mergeUpperBegin[i-1]; j < out_hullSizeUpper[i]; j++)
			{
				out_hull[k] = out_hullUpper[i*offset + j];
				k++;
			}
		}
	}
	cout << "The value of k is: " << k << endl;
	cout << "The hull computed on GPU is: " << endl;
	for (int i = 0; i < k; i++)
	{
		cout << out_hull[i].x << "\t" << out_hull[i].y << endl;
	}

	point_t *final_out_hull;
	final_out_hull = new point_t[SIZE];
	int final_hull_size;
	quickSortOnCPU(out_hull, 0, k - 1);
	convex_hull(out_hull, k, final_out_hull, &final_hull_size);

	//get ending tick
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	//cout << "The size of final convex hull is: " << final_hull_size << endl;
	//calculate elapsed time in ms
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cout << "The convex hull computed on CPU+GPU is: " << endl;
	for (int i = 0; i < final_hull_size; i++)
	{
		cout << final_out_hull[i].x << "\t" << final_out_hull[i].y << endl;
	}

	cout << "The elapsed time is: " << milliseconds << " ms" << endl;

	// Free device memory
	cudaFree(dev_points);
	cudaFree(dev_out_hullLower);
	cudaFree(dev_out_hullUpper);
	cudaFree(dev_out_hullSizeLower);
	cudaFree(dev_out_hullSizeUpper);
	cudaFree(dev_mergeLowerEnd);
	cudaFree(dev_mergeLowerBegin);
	cudaFree(dev_mergeUpperEnd);
	cudaFree(dev_mergeUpperBegin);

	//Free host memory


	delete final_out_hull;
	delete out_hullSizeLower;
	delete out_hullSizeUpper;

	free(out_hullLower);
	free(out_hullUpper);

	free(mergeLowerEnd);
	free(mergeLowerBegin);
	free(mergeUpperEnd);
	free(mergeUpperBegin);
	
	// Reset the device
	cudaDeviceReset();

}