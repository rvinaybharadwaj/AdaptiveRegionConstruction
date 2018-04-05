#include "ConvexHull.h"
using namespace std;

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
	//Allocate shared memory
	__shared__ int smem_hullSize[THREADS_BLOCK];
	__shared__ point_t smem_outHull[SMEM];

	//Calculate global index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Number of elements handled by each thread
	int elements_thread = npoints / (blockDim.x * gridDim.x);

	//Each thread has a different offset
	int offset = idx*elements_thread;

	//Shared memory offset
	int localOffset = threadIdx.x*elements_thread;

	//initialize the count to zero
	//out_hullsize1[idx] = 0;
	smem_hullSize[threadIdx.x] = 0;

	// lower hull 
	for (int i = offset; i < offset+elements_thread; ++i) 
	{
		/* while L contains at least two points and the sequence of last two points
		of L and the point P[i] does not make a counter-clockwise turn:
		remove the last point from L, append P[i] to L
		*/
		//while (out_hullsize1[idx] >= 2 && ccw(&out_hull1[offset + out_hullsize1[idx] - 2], &out_hull1[offset + out_hullsize1[idx] - 1], &points[i]) <= 0)
		//while (smem_hullSize[threadIdx.x] >= 2 && ccw(&out_hull1[offset + smem_hullSize[threadIdx.x] - 2], &out_hull1[offset + smem_hullSize[threadIdx.x] - 1], &points[i]) <= 0)
		while (smem_hullSize[threadIdx.x] >= 2 && ccw(&smem_outHull[localOffset + smem_hullSize[threadIdx.x] - 2], &smem_outHull[localOffset + smem_hullSize[threadIdx.x] - 1], &points[i]) <= 0)
		{
			//--out_hullsize1[idx];
			--smem_hullSize[threadIdx.x];
		}
		//out_hull1[offset + (out_hullsize1[idx]++)] = points[i];
		//out_hull1[offset + (smem_hullSize[threadIdx.x]++)] = points[i];
		smem_outHull[localOffset + smem_hullSize[threadIdx.x]] = points[i];
		out_hull1[offset + smem_hullSize[threadIdx.x]] = smem_outHull[localOffset + smem_hullSize[threadIdx.x]];
		smem_hullSize[threadIdx.x]++;
	}
	out_hullsize1[idx] = smem_hullSize[threadIdx.x];
	//printf("%d\n", out_hullsize1[idx]);
	//printf("out_hullsize1 = %d\n", out_hullsize1[idx]++);
}

__global__ void upperHullonGPU(point_t* points, int npoints, point_t* out_hull2, int* out_hullsize2)
{
	//Allocate shared memory
	__shared__ int smem_hullSize[THREADS_BLOCK];
	__shared__ point_t smem_outHull[SMEM];

	//Calculate global index
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	//Number of elements handled by each thread
	int elements_thread = npoints / (blockDim.x * gridDim.x);

	//Each thread has a different offset
	int offset = idx*elements_thread;

	//Calculate local offset
	int localOffset = threadIdx.x*elements_thread;

	//initialize the count to zero
	//out_hullsize2[idx] = 0;
	smem_hullSize[threadIdx.x] = 0;

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
		//while (out_hullsize2[idx] >= 2 && ccw(&out_hull2[offset + out_hullsize2[idx] - 2], &out_hull2[offset + out_hullsize2[idx] - 1], &points[i]) <= 0)
		//while (smem_hullSize[threadIdx.x] >= 2 && ccw(&out_hull2[offset + smem_hullSize[threadIdx.x] - 2], &out_hull2[offset + smem_hullSize[threadIdx.x] - 1], &points[i]) <= 0)
		while (smem_hullSize[threadIdx.x] >= 2 && ccw(&smem_outHull[localOffset + smem_hullSize[threadIdx.x] - 2], &smem_outHull[localOffset + smem_hullSize[threadIdx.x] - 1], &points[i]) <= 0)
		{
			//--out_hullsize2[idx];
			--smem_hullSize[threadIdx.x];
		}
		//out_hull2[offset + (out_hullsize2[idx]++)] = points[i];
		//out_hull2[offset + (smem_hullSize[threadIdx.x]++)] = points[i];
		smem_outHull[localOffset + smem_hullSize[threadIdx.x]] = points[i];
		out_hull2[offset + smem_hullSize[threadIdx.x]] = smem_outHull[localOffset + smem_hullSize[threadIdx.x]];
		smem_hullSize[threadIdx.x]++;
	}
	out_hullsize2[idx] = smem_hullSize[threadIdx.x];
	//out_hullsize2[idx]++;
	//printf("out_hullsize2 = %d", out_hullsize2[idx]++);
}

__global__ void mergeLowerHull(point_t *hull_part, int *part_size, int *i, int *j, int npoints)
{
	//Allocate shared memory
	__shared__ int smem_hullSize[THREADS_BLOCK];
	__shared__ point_t smem_hullPart[SMEM + ELEM_THREAD];

	//Calculate the global index
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	//printf("idx=%d",idx);

	//Load part size to shared memory
	smem_hullSize[threadIdx.x] = part_size[idx];

	//Number of threads
	int num_threads = blockDim.x*gridDim.x;

	//Number of elements seen by each thread, but size of individual convex hulls may be smaller
	int elements_thread = npoints / num_threads;

	//Load hull parts to shared memory
	if ((idx % blockDim.x == blockDim.x - 1) && (idx < num_threads - 1))
	{
		for (int i = 0; i < 2 * elements_thread; i++)
		{
			smem_hullPart[threadIdx.x*elements_thread + i] = hull_part[idx*elements_thread + i];
		}
	}
	if ((idx % blockDim.x < blockDim.x - 1) || (idx == num_threads - 1))
	{
		for (int i = 0; i < elements_thread; i++)
		{
			smem_hullPart[threadIdx.x*elements_thread + i] = hull_part[idx*elements_thread + i];
		}
	}
	__syncthreads();

	if (idx < num_threads - 1)
	{
		//Consider two sub-hulls in each thread
		/*int offset = idx*elements_thread;
		int next_offset = (idx + 1)*elements_thread;*/

		/*int quotient = idx / blockDim.x;
		if (idx % blockDim.x == 0)
		{
			for (int i = 0; i < elements_thread*(blockDim.x + 1) && i < elements_thread*num_threads; i++)
			{
				smem_hullPart[i] = hull_part[quotient * elements_thread * blockDim.x + i];
			}
		}
		__syncthreads();*/

		int localOffset = threadIdx.x*elements_thread;
		int localNextOffset = (threadIdx.x + 1)*elements_thread;

		//Loop conditions
		bool condition1 = true, condition2 = true;

		//Initialize count to zero
		/*i[idx] = 0;
		j[idx] = 0;*/

		int local_i = 0;
		int local_j = 0;

		//Points to construct tangent
		//point_t *a, *b;

		//For lower hull
		//a = &hull_part[offset + part_size[idx] - 1];//right most part of left hull
		//b = &hull_part[next_offset];//left most part of right hull
	/*	while (i[idx] < part_size[idx])
		{
			printf("a.x=%f\ta.y=%f", hull_part[offset + part_size[idx] - 1 - i[idx]].x, hull_part[offset + part_size[idx] - 1 - i[idx]].y);
			i[idx]++;
		}*/


		//Construct tangent
		while (condition1 || condition2)
		{
			condition1 = false;
			condition2 = false;

			while ((local_i < smem_hullSize[threadIdx.x] - 1) && (ccw(&smem_hullPart[localNextOffset + local_j], &smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 1 - local_i], &smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 2 - local_i]) > 0))
			//while ((i[idx] < smem_hullSize[threadIdx.x] - 1) && (ccw(&smem_hullPart[localNextOffset + j[idx]], &smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 1 - i[idx]], &smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 2 - i[idx]]) > 0))
			//while ((i[idx] < smem_hullSize[threadIdx.x] - 1) && (ccw(&hull_part[next_offset + j[idx]], &hull_part[offset + smem_hullSize[threadIdx.x] - 1 - i[idx]], &hull_part[offset + smem_hullSize[threadIdx.x] - 2 - i[idx]]) > 0))
			//while ((i[idx] < part_size[idx]-1) && (ccw(&hull_part[next_offset + j[idx]], &hull_part[offset + part_size[idx] - 1 - i[idx]], &hull_part[offset + part_size[idx] - 2 - i[idx]]) > 0))
			//while ((i[idx] < part_size[idx] - 1) && (ccw(b, a, a - 1) > 0))
			{
				//printf("a.x=%f\ta.y=%f", hull_part[offset + part_size[idx] - 1 - i[idx]].x, hull_part[offset + part_size[idx] - 1 - i[idx]].y);
				//printf("a=%d", *a);
				//a = (a - 1);
				//i[idx]++;
				local_i++;
				condition1 = true;
			}
			
			while ((local_j < smem_hullSize[threadIdx.x] - 1) && (ccw(&smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 1 - local_i], &smem_hullPart[localNextOffset + local_j], (&smem_hullPart[localNextOffset + 1 + local_j])) <= 0))
			//while ((j[idx] < smem_hullSize[threadIdx.x] - 1) && (ccw(&smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 1 - i[idx]], &smem_hullPart[localNextOffset + j[idx]], (&smem_hullPart[localNextOffset + 1 + j[idx]])) <= 0))
			//while ((j[idx] < smem_hullSize[threadIdx.x] - 1) && (ccw(&hull_part[offset + smem_hullSize[threadIdx.x] - 1 - i[idx]], &hull_part[next_offset + j[idx]], (&hull_part[next_offset + 1 + j[idx]])) <= 0))
			//while ((j[idx]<part_size[idx]-1) && (ccw(&hull_part[offset + part_size[idx] - 1 - i[idx]], &hull_part[next_offset + j[idx]], (&hull_part[next_offset + 1 + j[idx]])) <= 0))
			//while ((j[idx] < part_size[idx] - 1) && (ccw(a, b, b + 1) <= 0))
			{
				//printf("b.x=%f\tb.y=%f", hull_part[next_offset + j[idx]].x, hull_part[next_offset + j[idx]].y);
				//printf("b=%d", *b);
				//b = (b + 1);
				//j[idx]++;
				local_j++;
				condition2 = true;
			}
		}
		i[idx] = local_i;
		j[idx] = local_j;
		//printf("idx = %d and i = %d and j = %d\n", idx, i[idx], j[idx]);
	}
	//printf("Part size=%d",part_size[idx]);
}

__global__ void mergeUpperHull(point_t *hull_part, int *part_size, int *i, int *j, int npoints)
{
	//Allocate shared memory
	__shared__ int smem_hullSize[THREADS_BLOCK];
	__shared__ point_t smem_hullPart[SMEM + ELEM_THREAD];

	//Calculate the global index
	int idx = blockIdx.x*blockDim.x + threadIdx.x;

	//Load part size to shared memory
	smem_hullSize[threadIdx.x] = part_size[idx];

	int num_threads = blockDim.x * gridDim.x;

	//Number of elements seen by each thread, but size of individual convex hulls is smaller
	int elements_thread = npoints / num_threads;

	//Load hull parts to shared memory
	if ((idx % blockDim.x == blockDim.x - 1) && (idx < num_threads - 1))
	{
		for (int i = 0; i < 2 * elements_thread; i++)
		{
			smem_hullPart[threadIdx.x*elements_thread + i] = hull_part[idx*elements_thread + i];
		}
	}
	if ((idx % blockDim.x < blockDim.x - 1) || (idx == num_threads - 1))
	{
		for (int i = 0; i < elements_thread; i++)
		{
			smem_hullPart[threadIdx.x*elements_thread + i] = hull_part[idx*elements_thread + i];
		}
	}
	__syncthreads();

	if (idx < num_threads - 1)
	{
		//Consider two sub-hulls in each thread
		/*int offset = idx*elements_thread;
		int next_offset = (idx + 1)*elements_thread;*/

		//Loop conditions
		bool condition1 = true, condition2 = true;

		//Initialize count to zero
		/*i[idx] = 0;
		j[idx] = 0;*/
		int local_i = 0;
		int local_j = 0;

		//Calculate local offset
		int localOffset = threadIdx.x*elements_thread;
		int localNextOffset = (threadIdx.x+1)*elements_thread;

		//Points to construct tangent
		//point_t *a, *b;

		//For upper hull
		//a = &hull_part[offset + part_size[idx] - 1];//left most part of right hull
		//b = &hull_part[next_offset];//right most part of left hull

		//Construct tangent
		while (condition1 || condition2)
		{
			condition1 = false;
			condition2 = false;

			while ((local_i < smem_hullSize[threadIdx.x] - 1) && (ccw(&smem_hullPart[localNextOffset + local_j], &smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 1 - local_i], &smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 2 - local_i]) > 0))
			//while ((i[idx] < smem_hullSize[threadIdx.x] - 1) && (ccw(&smem_hullPart[localNextOffset + j[idx]], &smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 1 - i[idx]], &smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 2 - i[idx]]) > 0))
			//while ((i[idx] < part_size[idx] - 1) && (ccw(&hull_part[next_offset + j[idx]], &hull_part[offset + part_size[idx] - 1 - i[idx]], &hull_part[offset + part_size[idx] - 2 - i[idx]]) > 0))
			//while ((i[idx] < part_size[idx] - 1) && (ccw(b, a, a - 1) > 0))
			{
				//printf("x=%d\ty=%d", hull_part[offset + part_size[idx] - 1 - i[idx]].x, hull_part[offset + part_size[idx] - 1 - i[idx]].y);
				//a = (a - 1);
				//i[idx]++;
				local_i++;
				condition1 = true;
			}

			while ((local_j < smem_hullSize[threadIdx.x] - 1) && (ccw(&smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 1 - local_i], &smem_hullPart[localNextOffset + local_j], &smem_hullPart[localNextOffset + 1 + local_j]) <= 0))
			//while ((j[idx] < smem_hullSize[threadIdx.x] - 1) && (ccw(&smem_hullPart[localOffset + smem_hullSize[threadIdx.x] - 1 - i[idx]], &smem_hullPart[localNextOffset + j[idx]], &smem_hullPart[localNextOffset + 1 + j[idx]]) <= 0))
			//while ((j[idx] < part_size[idx] - 1) && (ccw(&hull_part[offset + part_size[idx] - 1 - i[idx]], &hull_part[next_offset + j[idx]], &hull_part[next_offset + 1 + j[idx]]) <= 0))
			//while ((j[idx] < part_size[idx] - 1) && (ccw(a, b, b + 1) <= 0))
			{
				//b = (b + 1);
				//j[idx]++;
				local_j++;
				condition2 = true;
			}
		}
		i[idx] = local_i;
		j[idx] = local_j;
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

//write to global memory by rearanging
__global__ void writeMemory(point_t *lowerHull, point_t *upperHull, int *sizeLower, int *sizeUpper, int *lowerEnd, int *lowerBegin, int *upperEnd, int *upperBegin, int npoints, int *hullSize)
{
	int k = 0;
	int num_threads = THREADS_BLOCK * NUM_BLOCKS;
	int offset = npoints / num_threads;

	//Writing lower hull
	for (int i = 0; i < num_threads; i++)
	{
		if (i == 0)
		{
			for (int j = 0; j < (sizeLower[i] - lowerEnd[i]); j++)
			{
				lowerHull[k] = lowerHull[j];
				k++;
			}
		}
		else if (i > 0 && i < num_threads - 1)
		{
			for (int j = lowerBegin[i - 1]; j < (sizeLower[i] - lowerEnd[i]); j++)
			{
				lowerHull[k] = lowerHull[i*offset + j];
				k++;
			}
		}
		else
		{
			for (int j = lowerBegin[i - 1]; j < sizeLower[i]; j++)
			{
				lowerHull[k] = lowerHull[i*offset + j];
				k++;
			}
		}
	}

	//Writing upper hull
	for (int i = 0; i < num_threads; i++)
	{
		if (i == 0)
		{
			for (int j = 0; j < (sizeUpper[i] - upperEnd[i]); j++)
			{
				lowerHull[k] = upperHull[j];
				k++;
			}
		}
		else if (i > 0 && i < num_threads - 1)
		{
			for (int j = upperBegin[i - 1]; j < (sizeUpper[i] - upperEnd[i]); j++)
			{
				lowerHull[k] = upperHull[i*offset + j];
				k++;
			}
		}
		else
		{
			for (int j = upperBegin[i - 1]; j < sizeUpper[i]; j++)
			{
				lowerHull[k] = upperHull[i*offset + j];
				k++;
			}
		}
	}
	*hullSize = k;
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
void runOnGPU(point_t *points, int npoints, point_t *out_hull, int *out_hullsize, double **et)
{
	//Performance Measurement Data
	const int num_it = 1;
	const int num_alg = 5;
	ThreadTimer tt;

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
			std::cout << "Running on GPU " << i << " (" << properties.name << ")" << std::endl;
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
	cudaThreadSetCacheConfig(cudaFuncCachePreferShared);

	for (int a = 0; a < num_it; a++)
	{
		// Allocate GPU memory.
		point_t *dev_points;
		cudaMalloc((void **)&dev_points, npoints * sizeof(point_t));

		tt.Start();
		// Copy data to device memory
		cudaMemcpy(dev_points, points, npoints * sizeof(point_t), cudaMemcpyHostToDevice);
		tt.Stop();
		et[a][1] = tt.GetElapsedTime();
		// Prepare Cuda Dynamic Program for the maximum depth of MAX_DEPTH.
		cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);

		// Launch on device
		int left = 0;
		int right = npoints - 1;
		//cout << "Launching kernel on the GPU" << endl;

		// Launch CUDA kernel to sort the points
		//quickSortOnGPU << < 1, 1 >> > (dev_points, left, right, 0);
		//cudaDeviceSynchronize(); // Blocks until the device has completed all preceding requested tasks 
		quickSortOnCPU(points, left, right);

		/*cudaMemcpy(points, dev_points, npoints * sizeof(point_t), cudaMemcpyDeviceToHost);
		printf("The sorted points are:");
		for (int i = 0; i < npoints; i++)
		{
		printf("%d and %d\n", points[i].x, points[i].y);
		}*/

		// Kernel parameters
		int threads_block = THREADS_BLOCK;
		int num_blocks = NUM_BLOCKS;
		int num_threads = threads_block*num_blocks;

		// Convex hull parameters
		int *out_hullSizeLower, *out_hullSizeUpper;
		int *dev_out_hullSizeLower, *dev_out_hullSizeUpper;
		int *mergeLowerEnd, *mergeLowerBegin;
		int *dev_mergeLowerEnd, *dev_mergeLowerBegin;
		int *mergeUpperEnd, *mergeUpperBegin;
		int *dev_mergeUpperEnd, *dev_mergeUpperBegin;
		point_t *out_hullLower, *out_hullUpper;
		point_t *dev_out_hullLower, *dev_out_hullUpper;
		int *hullSize, *dev_hullSize;

		//allocate memory on CPU
		//out_hullLower = new point_t[SIZE];
		//out_hullUpper = new point_t[SIZE];
		/*out_hullLower = (point_t*)calloc(SIZE, sizeof(point_t));
		out_hullUpper = (point_t*)calloc(SIZE, sizeof(point_t));
		out_hullSizeLower = new int[num_threads];
		out_hullSizeUpper = new int[num_threads];
		mergeLowerEnd = (int*)calloc(num_threads - 1, sizeof(int));
		mergeLowerBegin = (int*)calloc(num_threads - 1, sizeof(int));
		mergeUpperEnd = (int*)calloc(num_threads - 1, sizeof(int));
		mergeUpperBegin = (int*)calloc(num_threads - 1, sizeof(int));*/

		//Allocate pinned memory on CPU
		cudaError_t status;
		status = cudaMallocHost((void**)&out_hullLower, SIZE*sizeof(point_t));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - out_hullLower\n");
		status = cudaMallocHost((void**)&out_hullUpper, SIZE*sizeof(point_t));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - out_hullUpper\n");
		status = cudaMallocHost((void**)&out_hullSizeLower, num_threads*sizeof(int));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - out_hullSizeLower\n");
		status = cudaMallocHost((void**)&out_hullSizeUpper, num_threads*sizeof(int));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - out_hullSizeUpper\n");

		status = cudaMallocHost((void**)&mergeLowerEnd, (num_threads-1)*sizeof(int));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - mergeLowerEnd\n");
		status = cudaMallocHost((void**)&mergeLowerBegin, (num_threads - 1)*sizeof(int));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - mergeLowerBegin\n");
		status = cudaMallocHost((void**)&mergeUpperEnd, (num_threads - 1)*sizeof(int));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - mergeUpperEnd\n");
		status = cudaMallocHost((void**)&mergeUpperBegin, (num_threads - 1)*sizeof(int));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - mergeUpperBegin\n");

		status = cudaMallocHost((void**)&hullSize, sizeof(int));
		if (status != cudaSuccess)
			printf("Error allocating pinned host memory - hullSize\n");
		

		//allocate memory on GPU
		cudaMalloc((void **)&dev_out_hullLower, SIZE * sizeof(point_t));
		cudaMalloc((void **)&dev_out_hullUpper, SIZE * sizeof(point_t));
		cudaMalloc((void **)&dev_out_hullSizeLower, num_threads*sizeof(int));
		cudaMalloc((void **)&dev_out_hullSizeUpper, num_threads*sizeof(int));
		cudaMalloc((void **)&dev_mergeLowerEnd, (num_threads - 1)*sizeof(int));
		cudaMalloc((void **)&dev_mergeLowerBegin, (num_threads - 1)*sizeof(int));
		cudaMalloc((void **)&dev_mergeUpperEnd, (num_threads - 1)*sizeof(int));
		cudaMalloc((void **)&dev_mergeUpperBegin, (num_threads - 1)*sizeof(int));
		cudaMalloc((void **)&dev_hullSize, sizeof(int));

		//initialize cconvex hulls
		status = cudaMemset(dev_out_hullLower, 0, SIZE * sizeof(point_t));
		if (status != cudaSuccess)
			printf("Error in memset - dev_out_hullLower\n");
		status = cudaMemset(dev_out_hullUpper, 0, SIZE * sizeof(point_t));
		if (status != cudaSuccess)
			printf("Error in memset - dev_out_hullUpper\n");

		//Create events to measure time
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);


		//copy host memory to device memory
		cudaMemcpy(dev_out_hullLower, out_hullLower, SIZE * sizeof(point_t), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_out_hullUpper, out_hullUpper, SIZE * sizeof(point_t), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_out_hullSizeLower, out_hullSizeLower, num_threads*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_out_hullSizeUpper, out_hullSizeUpper, num_threads*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mergeLowerEnd, mergeLowerEnd, (num_threads - 1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mergeLowerBegin, mergeLowerBegin, (num_threads - 1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mergeUpperEnd, mergeUpperEnd, (num_threads - 1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mergeUpperBegin, mergeUpperBegin, (num_threads - 1)*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_hullSize, hullSize, sizeof(int), cudaMemcpyDeviceToHost);

		//Create streams 
		cudaStream_t st1, st2;
		cudaStreamCreateWithFlags(&st1, cudaStreamNonBlocking);
		cudaStreamCreateWithFlags(&st2, cudaStreamNonBlocking);
		float milliseconds = 0;

		//Get starting tick
		cudaEventRecord(start);
		//Launch CUDA kernels to compute convex hull
		lowerHullonGPU <<< num_blocks, threads_block, 0, st1 >>> (dev_points, npoints, dev_out_hullLower, dev_out_hullSizeLower);
		upperHullonGPU <<< num_blocks, threads_block, 0, st2 >>> (dev_points, npoints, dev_out_hullUpper, dev_out_hullSizeUpper);
		//cudaDeviceSynchronize(); // Blocks until the device has completed all preceding requested tasks	
		////get ending tick
		//cudaEventRecord(stop);
		//cudaEventSynchronize(stop);
		////calculate elapsed time in ms
		//cudaEventElapsedTime(&milliseconds, start, stop);
		//et[a][2] = milliseconds;

		////Get starting tick
		//cudaEventRecord(start);
		//Launch CUDA kernels to merge consecutive convex hulls
		mergeLowerHull <<< num_blocks, threads_block, 0, st1 >>> (dev_out_hullLower, dev_out_hullSizeLower, dev_mergeLowerEnd, dev_mergeLowerBegin, npoints);
		mergeUpperHull <<< num_blocks, threads_block, 0, st2 >>> (dev_out_hullUpper, dev_out_hullSizeUpper, dev_mergeUpperEnd, dev_mergeUpperBegin, npoints);
		cudaDeviceSynchronize(); // Blocks until the device has completed all preceding requested tasks	
		//get ending tick
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		//calculate elapsed time in ms
		cudaEventElapsedTime(&milliseconds, start, stop);
		et[a][2] = milliseconds;
		
		//Destroy streams
		cudaStreamDestroy(st1);
		cudaStreamDestroy(st2);

		writeMemory <<< 1, 1 >>> (dev_out_hullLower, dev_out_hullUpper, dev_out_hullSizeLower, dev_out_hullSizeUpper, dev_mergeLowerEnd, dev_mergeLowerBegin, dev_mergeUpperEnd, dev_mergeUpperBegin, npoints, dev_hullSize);
		cudaDeviceSynchronize();

		tt.Start();
		//Copy device memory to host memory
		/*cudaMemcpy(out_hullSizeLower, dev_out_hullSizeLower, num_threads*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(out_hullSizeUpper, dev_out_hullSizeUpper, num_threads*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(out_hullLower, dev_out_hullLower, npoints * sizeof(point_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(out_hullUpper, dev_out_hullUpper, npoints * sizeof(point_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(mergeLowerEnd, dev_mergeLowerEnd, (num_threads - 1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(mergeLowerBegin, dev_mergeLowerBegin, (num_threads - 1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(mergeUpperEnd, dev_mergeUpperEnd, (num_threads - 1)*sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(mergeUpperBegin, dev_mergeUpperBegin, (num_threads - 1)*sizeof(int), cudaMemcpyDeviceToHost);*/

		cudaMemcpy(hullSize, dev_hullSize, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(out_hullLower, dev_out_hullLower, *hullSize * sizeof(point_t), cudaMemcpyDeviceToHost);

		tt.Stop();
		et[a][3] = tt.GetElapsedTime();
		//cout << "The lower hull computed on GPU is: " << endl;
		//for (int i = 0; i < npoints; i++)
		//{
		//	cout << out_hullLower[i].x << "\t" << out_hullLower[i].y << endl;
		//}
		//cout << "The upper hull computed on GPU is: " << endl;
		//for (int i = 0; i < npoints; i++)
		//{
		//	cout << out_hullUpper[i].x << "\t" << out_hullUpper[i].y << endl;
		//}

		//for (int i = 0; i < num_threads; i++)
		//{
		//	//cout << "out_hullSizeLower: " << out_hullSizeLower[i] << endl;
		//	cout << "out_hullSizeUpper: " << out_hullSizeUpper[i] << endl;
		//}



		/*cudaMemcpy(out_hullLower, dev_out_hullLower, *out_hullSizeLower * sizeof(point_t), cudaMemcpyDeviceToHost);
		cudaMemcpy(out_hullUpper, dev_out_hullUpper, *out_hullSizeUpper * sizeof(point_t), cudaMemcpyDeviceToHost);*/

		//for (int i = 0; i < num_threads - 1; i++)
		//{
		//	cout << "mergeLowerEnd = " << mergeLowerEnd[i] << endl;
		//	cout << "mergeLowerBegin = " << mergeLowerBegin[i] << endl;
		//	//cout << "mergeUpperEnd = " << mergeUpperEnd[i] << endl;
		//	//cout << "mergeUpperBegin = " << mergeUpperBegin[i] << endl;
		//}

#pragma region mergeOnCPU
		//Write merged hull into an array
		/*int k = 0;
		int offset = npoints / num_threads;*/
		tt.Start();

		////Writing lower hull
		//for (int i = 0; i < num_threads; i++)
		//{
		//	if (i == 0)
		//	{
		//		for (int j = 0; j < (out_hullSizeLower[i] - mergeLowerEnd[i]); j++)
		//		{
		//			out_hull[k] = out_hullLower[j];
		//			k++;
		//		}
		//	}
		//	else if (i > 0 && i < num_threads - 1)
		//	{
		//		for (int j = mergeLowerBegin[i - 1]; j < (out_hullSizeLower[i] - mergeLowerEnd[i]); j++)
		//		{
		//			out_hull[k] = out_hullLower[i*offset + j];
		//			k++;
		//		}
		//	}
		//	else
		//	{
		//		for (int j = mergeLowerBegin[i - 1]; j < out_hullSizeLower[i]; j++)
		//		{
		//			out_hull[k] = out_hullLower[i*offset + j];
		//			k++;
		//		}
		//	}
		//}
		///*cout << "The lower convex hull computed on GPU is: " << endl;
		//for (int i = 0; i < k; i++)
		//{
		//cout << out_hull[i].x << "\t" << out_hull[i].y << endl;
		//}*/
		////Writing upper hull
		//for (int i = 0; i < num_threads; i++)
		//{
		//	if (i == 0)
		//	{
		//		for (int j = 0; j < (out_hullSizeUpper[i] - mergeUpperEnd[i]); j++)
		//		{
		//			out_hull[k] = out_hullUpper[j];
		//			k++;
		//		}
		//	}
		//	else if (i > 0 && i < num_threads - 1)
		//	{
		//		for (int j = mergeUpperBegin[i - 1]; j < (out_hullSizeUpper[i] - mergeUpperEnd[i]); j++)
		//		{
		//			out_hull[k] = out_hullUpper[i*offset + j];
		//			k++;
		//		}
		//	}
		//	else
		//	{
		//		for (int j = mergeUpperBegin[i - 1]; j < out_hullSizeUpper[i]; j++)
		//		{
		//			out_hull[k] = out_hullUpper[i*offset + j];
		//			k++;
		//		}
		//	}
		//}
		cout << "The value of hullSize is: " << *hullSize << endl;
		/*cout << "The partial convex hull computed on GPU is: " << endl;
		for (int i = 0; i < *hullSize; i++)
		{
		cout << out_hullLower[i].x << "\t" << out_hullLower[i].y << endl;
		}*/

		point_t *final_out_hull;
		final_out_hull = new point_t[*hullSize];
		quickSortOnCPU(out_hullLower, 0, *hullSize - 1);
		convex_hull(out_hullLower, *hullSize, final_out_hull, out_hullsize);
		tt.Stop();
		et[a][4]=tt.GetElapsedTime();
#pragma endregion mergeOnCPU

		cout << "The size of final convex hull is: " << *out_hullsize << endl;
		//calculate elapsed time in ms
		/*float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);*/
		/*cout << "The convex hull computed on CPU+GPU is: " << endl;
		for (int i = 0; i < *out_hullsize; i++)
		{
			cout << final_out_hull[i].x << "\t" << final_out_hull[i].y << endl;
		}*/

		//cout << "The elapsed time is: " << milliseconds << " ms" << endl;

		//write to out_hull for verification
		for (int i = 0; i < *out_hullsize; i++)
		{
			out_hull[i] = final_out_hull[i];
		}

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
		/*delete out_hullSizeLower;
		delete out_hullSizeUpper;

		free(out_hullLower);
		free(out_hullUpper);

		free(mergeLowerEnd);
		free(mergeLowerBegin);
		free(mergeUpperEnd);
		free(mergeUpperBegin);*/
		/*cudaFreeHost(out_hullSizeLower);
		cudaFreeHost(out_hullSizeUpper);*/
		cudaFreeHost(out_hullLower);
		/*cudaFreeHost(out_hullUpper);
		cudaFreeHost(mergeLowerEnd);
		cudaFreeHost(mergeLowerBegin);
		cudaFreeHost(mergeUpperEnd);
		cudaFreeHost(mergeUpperBegin);*/
	}
	char sbuff[10];
	char str[30];
	_itoa_s(SIZE, sbuff, 10, 10);
	strcpy_s(str, "resUfm");
	strcat_s(str, sbuff);
	strcat_s(str, ".csv");
	tt.SaveElapsedTimeMatrix(str, et, num_it, num_alg);
	//tt.SaveElapsedTimeMatrix("resReg2097152.csv", et, num_it, num_alg);
	// Reset the device
	cudaDeviceReset();	
}