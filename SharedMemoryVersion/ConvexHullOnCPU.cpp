#include "ConvexHull.h"

//selection sort
void sort_points(point_t* points, int npoints)
{
	point_t temp;
	for (int i = 0; i < npoints; i++)
	{
		for (int j = i + 1; j < npoints; j++)
		{
			//sort the points by x-coordinate; in case of a tie, sort by y-coordinate
			if ((points[i].x > points[j].x) || ((points[i].x == points[j].x) && (points[i].y > points[j].y)))
			{
				temp = points[i];
				points[i] = points[j];
				points[j] = temp;
			}
		}
	}
}

// Quicksort
void quickSortOnCPU(point_t* points, int left, int right) {
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
		quickSortOnCPU(points, left, nright);
	}

	// Launch a new block to sort the right part.
	if ((lptr - points) < right)
	{
		quickSortOnCPU(points, nleft, right);
	}
}


/* Three points are a counter-clockwise turn if ccw > 0, clockwise if
ccw < 0, and collinear if ccw = 0 because ccw is a determinant that
gives the signed area of the triangle formed by p1, p2 and p3.
|x1 y1 1|
|x2 y2 1|
|x3 y3 1|
*/
static double ccw(point_t* p1, point_t* p2, point_t* p3)
{
	return (p2->x - p1->x)*(p3->y - p1->y) - (p2->y - p1->y)*(p3->x - p1->x);
}


/* Returns a list of points on the convex hull in counter-clockwise order.
Note: the last point in the returned list is the same as the first one.
*/
void convex_hull(point_t* points, int npoints, point_t* out_hull, int* out_hullsize)
{
	point_t* hull;
	int i, t, k = 0;

	hull = out_hull;

	// lower hull 
	for (i = 0; i < npoints; ++i) {
		/* while L contains at least two points and the sequence of last two points
		of L and the point P[i] does not make a counter-clockwise turn:
		remove the last point from L, append P[i] to L
		*/
		while (k >= 2 && ccw(&hull[k - 2], &hull[k - 1], &points[i]) <= 0)
		{
			--k;
		}
		hull[k++] = points[i];
	}

	// upper hull 
	// The last point (npoint-1th point) of each list is same as the first 
	// point of the other list, so start from npoints-2
	/* while U contains at least two points and the sequence of last two points
	of U and the point P[i] does not make a counter-clockwise turn:
	remove the last point from U, append P[i] to U
	*/
	//t=k+1 to begin the upper hull - make a turn by considering the immediate point
	for (i = npoints - 2, t = k + 1; i >= 0; --i) {
		while (k >= t && ccw(&hull[k - 2], &hull[k - 1], &points[i]) <= 0)
		{
			--k;
		}
		hull[k++] = points[i];
	}

	out_hull = hull;
	*out_hullsize = k;
}

void verify_convexHull(point_t *out_hull_cpu, int *out_hullsize_cpu, point_t *out_hull_gpu, int *out_hullsize_gpu)
{
	if (*out_hullsize_cpu != *out_hullsize_gpu)
		printf("FAIL: Hull sizes don't match :( \n");
	else
	{
		int flag = 0;
		for (int i = 0; i < *out_hullsize_cpu; i++)
		{
			if ((fabs(out_hull_cpu[i].x - out_hull_gpu[i].x) >= 0.0001) || (fabs(out_hull_cpu[i].y - out_hull_gpu[i].y) >= 0.0001))
				flag = 1;
		}
		if (flag != 0)
			printf("FAIL: Values don't match :( \n");
		else
			printf("PASS!\n");
	}
}