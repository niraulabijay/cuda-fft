#include <bits/stdc++.h>
#include <opencv2/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
#include <chrono> 

#include <opencv4/opencv2/core/core_c.h>
#include <opencv2/highgui/highgui_c.h>
#include <cuda_runtime.h>
// #include <crt/device_functions.h>
#include <device_launch_parameters.h>
#include <cuComplex.h>
#include <opencv4/opencv2/imgcodecs.hpp>
// #include <opencv2/core/mat.hpp>
// #include <opencv2>s


using namespace std::chrono;
using namespace std;
using namespace cv;

typedef complex<float> base;
typedef float2 Complex_my;

template <typename T>
ostream &operator<<(ostream &o, vector<T> v)
{
    if (v.size() > 0)
        o << v[0];
    for (unsigned i = 1; i < v.size(); i++)
        o << " " << v[i];
    return o << endl;
}
static __device__ __host__ inline Complex_my Add(Complex_my A, Complex_my B)
{
    Complex_my result;
    result.x = A.x + B.x;
    result.y = A.y + B.y;
    return result;
}

/**
 *  Inverse of Complex_my Number
 */
static __device__ __host__ inline Complex_my Inverse(Complex_my A)
{
    Complex_my result;
    result.x = -A.x;
    result.y = -A.y;
    return result;
}

/**
 *  Multipication of Complex_my Numbers
 */
static __device__ __host__ inline Complex_my Multiply(Complex_my A, Complex_my B)
{
    Complex_my result;
    result.x = A.x * B.x - A.y * B.y;
    result.y = A.y * B.x + A.x * B.y;
    return result;
}

/* 
* This function subtracts the real and imaginary parts of b from a separately and 
* returns the result as a new Complex_my object.
*/
__device__ Complex_my operator+(const Complex_my& a, const Complex_my& b) {
    Complex_my result;
    result.x = a.x + b.x;
    result.y = a.y + b.y;
    return result;
}

__device__ Complex_my operator-(const Complex_my& a, const Complex_my& b) {
    Complex_my result;
    result.x = a.x - b.x;
    result.y = a.y - b.y;
    return result;
}

__device__ Complex_my operator*(const Complex_my& a, const Complex_my& b) {
    Complex_my result;
    result.x = a.x * b.x - a.y * b.y;
    result.y = a.x * b.y + a.y * b.x;
    return result;
}

/**
* Parallel Functions for performing various tasks
*/

/**
*  Dividing by constant for inverse fft transform
*/
__global__ void inplace_divide_invert(Complex_my *A, int n, int threads)
{
    int i = blockIdx.x * threads + threadIdx.x;
    if (i < n)
    {
        // printf("in divide");
        A[i].x /= n;
        A[i].y /= n;
    }
    else
    {
        // printf("else in divide");
        // printf("i=%d, n=%d", i, n);
    }
}

/**
* Reorders array by bit-reversing the indexes.
*/
__global__ void bitrev_reorder(Complex_my *__restrict__ r, Complex_my *__restrict__ d, int s, size_t nthr, int n)
{
    int id = blockIdx.x * nthr + threadIdx.x;
    //r[id].x = -1;
    if (id < n and __brev(id) >> (32 - s) < n)
        r[__brev(id) >> (32 - s)] = d[id];
}


/*
* The bitrev_reorder function is performing a bit-reversal operation on the indices of the input array for a radix-2 FFT. 
* For a radix-4 FFT, you would need to perform a base-4 digit reversal operation instead.
*/
__global__ void bitrev_reorder_radix4(Complex_my *A, int s, size_t nthr, int n) {
    int id = blockIdx.x * nthr + threadIdx.x;
    if (id < n) {
        int rev = 0;
        int temp = id;
        for (int i = 0; i < s; i++) { 
            rev <<= 1;
            rev |= (temp & 1);
            temp >>= 1;
        }

        // In-place swap
        if (rev > id) { 
            Complex_my temp = A[id];
            A[id] = A[rev];
            A[rev] = temp;
        }
    }
}



/**
* Inner part of the for loop
*/
__device__ void inplace_fft_inner(Complex_my *__restrict__ A, int i, int j, int len, int n, bool invert)
{
    if (i + j + len / 2 < n and j < len / 2)
    {
        Complex_my u, v;

        float angle = (2 * M_PI * j) / (len * (invert ? 1.0 : -1.0));
        v.x = cos(angle);
        v.y = sin(angle);

        u = A[i + j];
        v = Multiply(A[i + j + len / 2], v);
        // printf("i:%d j:%d u_x:%f u_y:%f    v_x:%f v_y:%f\n", i, j, u.x, u.y, v.x, v.y);
        A[i + j] = Add(u, v);
        A[i + j + len / 2] = Add(u, Inverse(v));
    }
}

// CUDA Kernel function to generate the Hanning window
__global__ void hanningWindow(float *d_in, int N, float scale_factor = 1.0) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float scale = 2.0f * 3.14159265358979323846f / (N - 1);
        d_in[idx] *= scale_factor * 0.5f * (1.0f - cosf(scale * idx));
    }
}

/**
* FFT if number of threads are sufficient.
*/
__global__ void inplace_fft(Complex_my *__restrict__ A, int i, int len, int n, int threads, bool invert)
{
    int j = blockIdx.x * threads + threadIdx.x;
    inplace_fft_inner(A, i, j, len, n, invert);
}


__global__ void inplace_fft_radix4(Complex_my *__restrict__ A, int start, int len, int n, size_t nthr, bool invert)
{
    int id = blockIdx.x * nthr + threadIdx.x;
    if (id < len / 4)
    {
        // float angle = 2.0f * M_PI / len * (invert ? -1 : 1); // first iter
        // float angle = 2.0f * M_PI / (4 * len) * (invert ? -1 : 1) * id; // 2nd iter
        float angle = 2.0f * M_PI / (4 * len) * (invert ? -1 : 1) * id; 

        Complex_my t1, t2, t3, t4;
        t1 = A[start + id] + A[start + id + len];
        t2 = A[start + id + 2 * len] + A[start + id + 3 * len];
        t3 = A[start + id] - A[start + id + len];
        t4.x = 0;
        t4.y = invert ? -1 : 1;

        t4.x = cos(id * angle);
        t4.y = invert ? -sin(id * angle) : sin(id * angle);
        t4 = (A[start + id + 2 * len] - A[start + id + 3 * len]) * t4; // 2nd iter 
        // t4 = (A[start + id + 2 * len] - A[start + id + 3 * len]) * Complex_my(cos(id * angle), invert ? -sin(id * angle) : sin(id * angle)); //3rd iter
        // t4 = (A[start + id + 2 * len] - A[start + id + 3 * len]) * Complex_my(0, invert ? -1 : 1); //1st iter
        A[start + id] = t1 + t2;
        A[start + id + len] = t3 + t4;
        A[start + id + 2 * len] = t1 - t2;
        A[start + id + 3 * len] = t3 - t4;
    }
}

/**
* FFt if number of threads are not sufficient.
*/
__global__ void inplace_fft_outer(Complex_my *__restrict__ A, int len, int n, int threads, bool invert)
{
    int i = (blockIdx.x * threads + threadIdx.x)*len;
    for (int j = 0; j < len / 2; j++)
    {
        inplace_fft_inner(A, i, j, len, n, invert);
    }
}


__global__ void inplace_fft_outer_radix4(Complex_my *__restrict__ A, int len, int n, size_t nthr, bool invert) {
    int blockId = blockIdx.x;
    int id = blockId * nthr + threadIdx.x;
    int start = id * len * 4;

    if (start < n) {
        for (int i = start; i < start + len; i++) {
            if (i < n) {
                // Corrected angle calculation and indexing.
                float angle = 2.0f * M_PI / (4 * len) * (invert ? -1 : 1) * (id % len);
                Complex_my t4;
                // w.x = cos(angle);
                // w.y = sin(angle);

                // Corrected data indices
                Complex_my t1 = A[i] + A[i + len];
                Complex_my t2 = A[i + 2 * len] + A[i + 3 * len];
                Complex_my t3 = A[i] - A[i + len];
                t4.x = 0;
                t4.y = invert ? -1 : 1;

                t4.x = cos((id % len) * angle);
                t4.y = invert ? -sin((id % len) * angle) : sin((id % len) * angle);
                t4 = (A[i + 2 * len] - A[i + 3 * len]) * t4;

                A[i] = t1 + t2;
                A[i + len] = t3 + t4;
                A[i + 2 * len] = t1 - t2;
                A[i + 3 * len] = t3 - t4;
            }
        }
    }
}


template <typename T>
T clamp(const T& val, const T& low, const T& high) {
    return std::min(std::max(val, low), high);
}


/**
* parallel FFT transform and inverse transform
* Arguments vector of complex numbers, invert, balance, number of threads
* Perform inplace transform
*/
void fft(vector<base> &a, bool invert, int balance = 10, int threads = 32)
{
    // Creating array from vector
    int n = (int)a.size();
    int data_size = n * sizeof(Complex_my);
    Complex_my *data_array = (Complex_my *)malloc(data_size);
    for (int i = 0; i < n; i++)
    {
        data_array[i].x = a[i].real();
        data_array[i].y = a[i].imag();
    }
    
    // Copying data to GPU
    Complex_my *A, *dn;
    cudaMalloc((void **)&A, data_size);
    cudaMalloc((void **)&dn, data_size);
    cudaMemcpy(dn, data_array, data_size, cudaMemcpyHostToDevice);
    // Bit reversal reordering
    int s = log2(n);

    bitrev_reorder<<<ceil(float(n) / threads), threads>>>(A, dn, s, threads, n);

    
    // Synchronize
    cudaDeviceSynchronize();
    // Iterative FFT with loop parallelism balancing
    for (int len = 2; len <= n; len <<= 1)
    {
        if (n / len > balance)
        {

            inplace_fft_outer<<<ceil((float)n / threads / len), threads>>>(A, len, n, threads, invert);
        }
        else
        {
            for (int i = 0; i < n; i += len)
            {
                float repeats = len / 2;
                inplace_fft<<<ceil(repeats / threads), threads>>>(A, i, len, n, threads, invert);
            }
        }
    }
    
    if (invert)
        inplace_divide_invert<<<ceil(n * 1.00 / threads), threads>>>(A, n, threads);

    // Copy data from GPU
    Complex_my *result;
    result = (Complex_my *)malloc(data_size);
    cudaMemcpy(result, A, data_size, cudaMemcpyDeviceToHost);
    
    // Saving data to vector<complex> in input.
    for (int i = 0; i < n; i++)
    {
        a[i] = base(result[i].x, result[i].y);
    }
    // Free the memory blocks
    free(data_array);
    cudaFree(A);
    cudaFree(dn);
    return;
}

/* 
* To implement a radix-4 FFT, you need to modify the bit reversal reordering and the FFT computation. 
* The bit reversal reordering needs to be modified to use base-4 instead of base-2. 
* The FFT computation needs to be modified to process four points at a time instead of two. 
*/
void fft_radix4(vector<base> &a, bool invert, int balance = 10, int threads = 32)
{
    // Creating array from vector
    int n = (int)a.size();
    int data_size = n * sizeof(Complex_my);
    Complex_my *data_array = (Complex_my *)malloc(data_size);
    for (int i = 0; i < n; i++)
    {
        data_array[i].x = a[i].real();
        data_array[i].y = a[i].imag();
    }
    
    // Copying data to GPU
    Complex_my *A, *dn;
    cudaMalloc((void **)&A, data_size);
    cudaMalloc((void **)&dn, data_size);
    cudaMemcpy(A, data_array, data_size, cudaMemcpyHostToDevice); // Copy data_array to A
    // Bit reversal reordering
    int s = log2(n);

    // bitrev_reorder_radix4<<<ceil(float(n) / threads), threads>>>(A, dn, s, threads, n);
    bitrev_reorder_radix4<<<ceil(float(n) / threads), threads>>>(A, s, threads, n); 


    
    // Synchronize
    cudaDeviceSynchronize();

    // Bit reversal reordering
    // int s = log2(n);
    // bitrev_reorder_radix4<<<ceil(float(n) / threads), threads>>>(A, dn, s, threads, n);
    int blocks = n / 4;

    // Radix-4 FFT
    for (int len = 1; len < n; len *= 4)
    {
        inplace_fft_radix4<<<ceil(blocks / len), threads>>>(A, 0, len * 4, n, threads, invert);
        // inplace_fft_outer_radix4<<<ceil(blocks / (4 * len)), threads>>>(A, len, n, threads, invert);
        inplace_fft_outer_radix4<<<ceil(float(n) / (4 * len * threads)), threads>>>(A, len, n, threads, invert);
    }
    
    if (invert)
        inplace_divide_invert<<<ceil(n * 1.00 / threads), threads>>>(A, n, threads);

    // Copy data from GPU
    Complex_my *result;
    result = (Complex_my *)malloc(data_size);
    cudaMemcpy(result, A, data_size, cudaMemcpyDeviceToHost);
    
    // Saving data to vector<complex> in input.
    for (int i = 0; i < n; i++)
    {
        a[i] = base(result[i].x, result[i].y);
    }
    // Free the memory blocks
    free(data_array);
    cudaFree(A);
    cudaFree(dn);
    return;
}


/** Applying hanning window*/
void applyHanningWindow(vector<base> &a) {
    float *d_in;
    cudaMalloc(&d_in, a.size() * sizeof(base));
    cudaMemcpy(d_in, a.data(), a.size() * sizeof(base), cudaMemcpyHostToDevice);

    hanningWindow<<<(a.size() + 255) / 256, 256>>>(d_in, a.size(), 0.7);
    cudaDeviceSynchronize();

    cudaMemcpy(a.data(), d_in, a.size() * sizeof(base), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
}

/**
* Performs 2D FFT 
* takes vector of complex vectors, invert and verbose as argument
* performs inplace FFT transform on input vector
*/
void fft2D(vector<vector<base>> &a, bool invert, int balance, int threads, int verbose = 0)
{
    auto matrix = a;
    // Transform the rows
    if (verbose > 0)
        cout << "Transforming Rows" << endl;

    for (auto i = 0; i < matrix.size(); i++)
    {
        //cout<<i<<endl;
        // applyHanningWindow(matrix[i]);
        // fft(matrix[i], invert, balance, threads);
        fft(matrix[i], invert, balance, threads);
    }

    // preparing for transforming columns

    if (verbose > 0)
        cout << "Converting Rows to Columns" << endl;

    a = matrix;
    matrix.resize(a[0].size());
    for (int i = 0; i < matrix.size(); i++)
        matrix[i].resize(a.size());

    // Transposing matrix
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            matrix[j][i] = a[i][j];
        }
    }
    if (verbose > 0)
        cout << "Transforming Columns" << endl;

    // Transform the columns
    for (auto i = 0; i < matrix.size(); i++){
        // applyHanningWindow(matrix[i]);
        // fft(matrix[i], invert, balance, threads);
        fft(matrix[i], invert, balance, threads);
    }

    if (verbose > 0)
        cout << "Storing the result" << endl;

    // Storing the result after transposing
    // [j][i] is getting value of [i][j]
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            a[j][i] = matrix[i][j];
        }
    }
}


// void inplaceTranspose(vector<vector<base>>& matrix, int startRow = 0, int startCol = 0, int size = -1) {
//     if (size == -1) {
//         // If size is not specified, transpose the whole matrix
//         size = matrix.size();
//     }

//     // Base case: 1x1 matrix (no transpose needed)
//     if (size <= 1) {
//         return;
//     }

//     // Recursive step: Divide into four submatrices and transpose each recursively
//     int halfSize = size / 2;
//     for (int i = 0; i < halfSize; i++) {
//         for (int j = 0; j < halfSize; j++) {
//             swap(matrix[startRow + i][startCol + j], matrix[startRow + j][startCol + i]);
//             swap(matrix[startRow + i + halfSize][startCol + j], matrix[startRow + j][startCol + i + halfSize]);
//             swap(matrix[startRow + i][startCol + j + halfSize], matrix[startRow + j + halfSize][startCol + i]);
//             swap(matrix[startRow + i + halfSize][startCol + j + halfSize], matrix[startRow + j + halfSize][startCol + i + halfSize]);
//         }
//     }

//     // Recursively transpose the top-left and bottom-right submatrices
//     inplaceTranspose(matrix, startRow, startCol, halfSize);
//     inplaceTranspose(matrix, startRow + halfSize, startCol + halfSize, halfSize);
// }

void inplaceTranspose(vector<vector<base>>& matrix) {
    int n = matrix.size();
    if (n <= 1) return;

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            swap(matrix[i][j], matrix[j][i]);
        }
    }
}


// This function first applies fft_radix4 to the rows of the matrix, then transposes the matrix, 
// applies fft_radix4 to the rows of the transposed matrix (which are the columns of the original matrix), 
// and finally transposes the result back.
void fft2D_radix4(vector<vector<base>> &a, bool invert, int balance, int threads, int verbose = 0) {
    int rows = a.size();
    int cols = a[0].size();

    // Transform the rows
    if (verbose > 0)
        cout << "Transforming Rows" << endl;

    for (int i = 0; i < rows; i++) {
        fft_radix4(a[i], invert, balance, threads);
    }

    // In-place transpose (assuming rows and cols are powers of 2)
    if (verbose > 0)
        cout << "Transposing Matrix" << endl;
    inplaceTranspose(a);

    // Transform the columns (now rows after transpose)
    if (verbose > 0)
        cout << "Transforming Columns" << endl;
    for (int i = 0; i < cols; i++) {
        fft_radix4(a[i], invert, balance, threads);
    }

    // In-place transpose back
    if (verbose > 0)
        cout << "Transposing Matrix Back" << endl;
    inplaceTranspose(a);
}


/**
* Function to multiply two polynomial
* takes two polynomials represented as vectors as input
* return the product of two vectors
*/
vector<int> mult(vector<int> a, vector<int> b, int balance, int threads)
{
    // Creating complex vector from input vectors
    vector<base> fa(a.begin(), a.end()), fb(b.begin(), b.end());

    // Padding with zero to make their size equal to power of 2
    size_t n = 1;
    while (n < max(a.size(), b.size()))
        n <<= 1;
    n <<= 1;

    fa.resize(n), fb.resize(n);

    // Transforming both a and b
    // Converting to points form
    fft(fa, false, balance, threads), fft(fb, false, balance, threads);

    // performing point wise multipication of points
    for (size_t i = 0; i < n; ++i)
        fa[i] *= fb[i];

    // Performing Inverse transform
    fft(fa, true, balance, threads);

    // Saving the real part as it will be the result
    vector<int> res;
    res.resize(n);
    for (size_t i = 0; i < n; ++i)
        res[i] = int(fa[i].real() + 0.5);

    return res;
}

/**
* Function to perform jpeg compression on image
* takes image, threshold, verbose as input
* image is represented as vector<vector>
* perform inplace compression on the input
*/
void compress_image(vector<vector<uint>> &image, double threshold, int balance, int threads, int verbose = 1)
{
    //Convert image to complex type
    cout << "here";
    vector<vector<base>> complex_image(image.size(), vector<base>(image[0].size()));
    for (auto i = 0; i < image.size(); i++)
    {
        for (auto j = 0; j < image[0].size(); j++)
        {
            complex_image[i][j] = image[i][j];
        }
    }
    if (verbose == 1)
    {
        cout << "input Image" << endl;
        //cout << image;
        cout << endl
            << endl;
    }
    if (verbose > 1)
    {
        cout << "Complex Image" << endl;
        cout << complex_image;
        cout << endl
            << endl;
    }

    
    cout << "before fft2d temp_image[0][0] = " << complex_image[0][2] << endl;

    //Perform 2D fft on image
    cout << "BEFORE here";
    fft2D(complex_image, false, balance, threads, verbose);
    cout << "After fft2d here";
    if (verbose == 1)
    {
        cout << "Performing FFT on Image" << endl;
        ///cout << complex_image;
        cout << endl
            << endl;
    }

    
    cout << "temp_image[0][0] = " << complex_image[0][0] << endl;

    //Threshold the fft

    // for (int i = 0; i < image_M.rows; ++i)
    //     for (int j = 0; j < image_M.cols; ++j)
    //         image_M.at<uint>(i, j) = image[i][j];

    double maximum_value = 0.0;
    for (int i = 0; i < complex_image.size(); i++)
    {
        for (int j = 0; j < complex_image[0].size(); j++)
        {
            maximum_value = max(maximum_value, abs(complex_image[i][j]));
        }
    }
    threshold *= maximum_value;
    // cout << "threshold :" << threshold << endl;
    int count = 0;

    // Setting values less than threshold to zero
    // This step is responsible for compression
    for (int i = 0; i < complex_image.size(); i++)
    {
        for (int j = 0; j < complex_image[0].size(); j++)
        {
            if (abs(complex_image[i][j]) < threshold)
            {
                count++;
                complex_image[i][j] = 0;
            }
        }
    }
    int zeros_count = 0;
    for (int i = 0; i < complex_image.size(); i++)
    {
        for (int j = 0; j < complex_image[0].size(); j++)
        {
            if (abs(complex_image[i][j]) == 0)
            {
                zeros_count++;
            }
        }
    }
    cout << "Components removed(percent): " << ((zeros_count*1.00/(complex_image.size()*complex_image[0].size())))*100 << endl;
    if (verbose > 1)
    {
        cout << "Thresholded Image" << endl;
        //cout << complex_image;
        cout << endl
            << endl;
    }

    // Perform inverse FFT
    fft2D(complex_image, true, balance, threads, verbose);
    if (verbose > 1)
    {
        cout << "Inverted Image" << endl;
        //cout << complex_image;
        cout << endl
            << endl;
    }
    //Convert to uint8 format
    // We will consider only the real part of the image
    for (int i = 0; i < complex_image.size(); i++)
    {
        for (int j = 0; j < complex_image[0].size(); j++)
        {
            image[i][j] = uint(complex_image[i][j].real() + 0.5);
        }
    }
    if (verbose > 0)
    {
        cout << "Compressed Image" << endl;
        //cout << image;
    }
}

void compress_image_radix4_old(vector<vector<uint>> &image, double thresholdPercentage, int balance, int threads, int verbose = 1)
{
    //Convert image to complex type
    // cout << "here";
    vector<vector<base>> complex_image(image.size(), vector<base>(image[0].size()));
    for (auto i = 0; i < image.size(); i++)
    {
        for (auto j = 0; j < image[0].size(); j++)
        {
            complex_image[i][j] = image[i][j];
        }
    }
    vector<vector<base>> temp_image = complex_image;

    if (verbose == 1)
    {
        cout << "input Image" << endl;
        //cout << image;
        cout << endl
            << endl;
    }
    if (verbose > 1)
    {
        cout << "Complex Image" << endl;
        cout << temp_image;
        cout << endl
            << endl;
    }

    
    cout << "before fft2d temp_image[0][0] = " << temp_image[0][2] << endl;

    //Perform 2D fft on image
    // cout << "BEFORE here";
    fft2D_radix4(temp_image, false, balance, threads, verbose);
    // cout << "After fft2d here";
    if (verbose == 1)
    {
        cout << "Performing FFT on Image" << endl;
        ///cout << temp_image;
        cout << endl
            << endl;
    }

    
    // code to check first valye of temp_image vector for debugging
    cout << "temp_image[0][0] = " << temp_image[0][2] << endl;


    // Normalize FFT coefficients
    double normFactor = sqrt(temp_image.size() * temp_image[0].size());
    for (auto& row : temp_image) {
        for (auto& val : row) {
            val /= normFactor;
        }
    }

    // Calculate threshold based on total energy
    double totalEnergy = 0.0;
    for (const auto& row : temp_image) {
        for (const auto& val : row) {
            totalEnergy += norm(val); // norm calculates the magnitude of the complex number
        }
    }
    double threshold = thresholdPercentage * totalEnergy; 



    //Threshold the fft

    // for (int i = 0; i < image_M.rows; ++i)
    //     for (int j = 0; j < image_M.cols; ++j)
    //         image_M.at<uint>(i, j) = image[i][j];

    double maximum_value = 0.0;
    for (int i = 0; i < temp_image.size(); i++)
    {
        for (int j = 0; j < temp_image[0].size(); j++)
        {
            maximum_value = max(maximum_value, abs(temp_image[i][j]));
        }
    }
    threshold *= maximum_value;
    // cout << "threshold :" << threshold << endl;
    int count = 0;

    // Setting values less than threshold to zero
    // This step is responsible for compression
    for (int i = 0; i < temp_image.size(); i++)
    {
        for (int j = 0; j < temp_image[0].size(); j++)
        {
            if (abs(temp_image[i][j]) < threshold)
            {
                count++;
                temp_image[i][j] = 0;
            }
        }
    }
    int zeros_count = 0;
    for (int i = 0; i < temp_image.size(); i++)
    {
        for (int j = 0; j < temp_image[0].size(); j++)
        {
            if (abs(temp_image[i][j]) == 0)
            {
                zeros_count++;
            }
        }
    }
    cout << "Components removed(percent): " << ((zeros_count*1.00/(temp_image.size()*temp_image[0].size())))*100 << endl;
    if (verbose > 1)
    {
        cout << "Thresholded Image" << endl;
        //cout << temp_image;
        cout << endl
            << endl;
    }


    // Renormalize before inverse FFT
    for (auto& row : temp_image) {
        for (auto& val : row) {
            val *= normFactor;
        }
    }

    // Perform inverse FFT
    fft2D_radix4(temp_image, true, balance, threads, verbose);

    // Convert to uint8 format with clamping
    for (int i = 0; i < temp_image.size(); i++) {
        for (int j = 0; j < temp_image[0].size(); j++) {
            double realVal = temp_image[i][j].real();
             image[i][j] = static_cast<uint>(clamp(realVal + 0.5, 0.0, 255.0));
        }
    }


    // Perform inverse FFT
    fft2D_radix4(temp_image, true, balance, threads, verbose);
    if (verbose > 1)
    {
        cout << "Inverted Image" << endl;
        //cout << temp_image;
        cout << endl
            << endl;
    }
    //Convert to uint8 format
    // We will consider only the real part of the image
    for (int i = 0; i < temp_image.size(); i++)
    {
        for (int j = 0; j < temp_image[0].size(); j++)
        {
            image[i][j] = uint(temp_image[i][j].real() + 0.5);
        }
    }
    if (verbose > 0)
    {
        cout << "Compressed Image" << endl;
        //cout << image;
    }
}



void compress_image_radix4(vector<vector<uint>> &image, double thresholdPercentage, int balance, int threads, int verbose) {
    vector<vector<base>> complex_image(image.size(), vector<base>(image[0].size()));
    for (int i = 0; i < image.size(); i++) {
        for (int j = 0; j < image[0].size(); j++) {
            complex_image[i][j] = base(image[i][j], 0.0);
        }
    }

    vector<vector<base>> temp_image = complex_image;

    if (verbose == 1) {
        cout << "input Image" << endl;
    }
    if (verbose > 1) {
        cout << "Complex Image" << endl;
    }

    // Debug: Check initial value before FFT
    cout << "before fft2d temp_image[0][0] = " << temp_image[0][0] << endl;

    cout << "Performing FFT on Image" << endl;
    fft2D_radix4(temp_image, false, balance, threads, verbose);

    // Debug: Check value after FFT
    cout << "complex_image[0][0] = " << temp_image[0][0] << endl;

    // Calculate maximum value in the frequency domain
    double max_val = 0.0;
    for (const auto& row : temp_image) {
        for (const auto& val : row) {
            max_val = max(max_val, abs(val));
        }
    }

    // Calculate threshold based on the maximum value
    double threshold = thresholdPercentage * max_val;

    int count = 0;
    for (int i = 0; i < temp_image.size(); i++) {
        for (int j = 0; j < temp_image[0].size(); j++) {
            if (abs(temp_image[i][j]) < threshold) {
                count++;
                temp_image[i][j] = base(0, 0);
            }
        }
    }

    cout << "Components removed (percent): " << (count * 100.0 / (temp_image.size() * temp_image[0].size())) << "%" << endl;

    // Perform inverse FFT
    fft2D_radix4(temp_image, true, balance, threads, verbose);

    // Debug: Check value after inverse FFT
    cout << "after inverse fft2d temp_image[0][0] = " << temp_image[0][0] << endl;

    for (int i = 0; i < temp_image.size(); i++) {
        for (int j = 0; j < temp_image[0].size(); j++) {
            double realVal = temp_image[i][j].real();
            image[i][j] = static_cast<uint>(clamp(realVal + 0.5, 0.0, 255.0));
        }
    }

    if (verbose > 0) {
        cout << "Compressed Image" << endl;
    }
}


 //Convert to uint8 format
    // We will consider only the real part of the image
    // for (int i = 0; i < complex_image.size(); i++)
    // {
    //     for (int j = 0; j < complex_image[0].size(); j++)
    //     {
    //         image[i][j] = uint(complex_image[i][j].real() + 0.5);
    //     }
    // }






void write2DVectorToFile(const std::vector<std::vector<uint>>& data, const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        for (const auto& row : data) {
            for (const auto& element : row) {
                file << element << ' ';
            }
            file << '\n';
        }
        file.close();
    } else {
        std::cout << "Unable to open file";
    }
}


// Function to calculate the FFT using the Cooley-Tukey algorithm
std::vector<std::complex<double>> fft_cooley_tukey(std::vector<std::complex<double>> x) {
    int N = x.size();
    if (N <= 1) return x;

    // Divide
    std::vector<std::complex<double>> even(N/2);
    std::vector<std::complex<double>> odd(N/2);
    for (int i = 0; i < N/2; i++) {
        even[i] = x[i*2];
        odd[i] = x[i*2+1];
    }

    // Conquer
    even = fft_cooley_tukey(even);
    odd = fft_cooley_tukey(odd);

    // Combine
    std::vector<std::complex<double>> T(N);
    for (int k = 0; k < N/2; k++) {
        std::complex<double> t = std::polar(1.0, -2 * M_PI * k / N) * odd[k];
        T[k] = even[k] + t;
        T[k+N/2] = even[k] - t;
    }

    return T;
}


#define N 100000
#define BALANCE 1024


vector<vector<uint>> read_2d_vector(const std::string& file_path) {
    vector<vector<uint>> data;
    std::ifstream file(file_path);
    std::string line;

    while (std::getline(file, line)) {
        vector<uint> row;
        std::stringstream ss(line);
        uint value;

        while (ss >> value) {
            row.push_back(value);
        }

        data.push_back(row);
    }


    // Now implement zero padding.
    // Determine the next power of 2
    size_t max_dim = std::max(data.size(), data[0].size());
    size_t next_power_of_2 = std::pow(2, std::ceil(std::log2(max_dim)));

    // Resize the outer vector
    data.resize(next_power_of_2);

    // Resize each inner vector
    for (auto& inner : data) {
        inner.resize(next_power_of_2, 0); // 0 is the padding value
    }
    return data;
}


void test_fft2D_radix4() {
    // Known input matrix
    vector<vector<base>> input = {
        {1, 2},
        {3, 4}
    };

    // Expected output (manually computed or using a reliable FFT library)
    vector<vector<base>> expected_output = {
        {base(10, 0), base(-2, 0)},
        {base(-4, 0), base(0, 0)}
    };

    // Apply fft2D_radix4 to the input matrix
    vector<vector<base>> output = input;
    fft2D_radix4(output, false, 10, 32, 1);

    // Compare output with expected output
    bool correct = true;
    for (int i = 0; i < output.size(); i++) {
        for (int j = 0; j < output[0].size(); j++) {
            if (abs(output[i][j] - expected_output[i][j]) > 1e-6) { // Allow small numerical errors
                correct = false;
                break;
            }
        }
        if (!correct) break;
    }

    // Print the result of the test
    if (correct) {
        cout << "fft2D_radix4 test PASSED!" << endl;
    } else {
        cout << "fft2D_radix4 test FAILED!" << endl;
        cout << "Expected output: " << endl;
        for (const auto &row : expected_output) {
            for (const auto &val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
        cout << "Actual output: " << endl;
        for (const auto &row : output) {
            for (const auto &val : row) {
                cout << val << " ";
            }
            cout << endl;
        }
    }
}



int main()
{
    // test_fft2D_radix4();

    std::vector<std::vector<uint>> image = read_2d_vector("image2d.txt");
    
    int count = 1;
    

    for(double thresh = 0.0000000001; thresh < 0.0001; thresh *= 10)
    {
        cout << "For thresh= " << thresh << endl;
        compress_image_radix4(image, thresh, BALANCE, 16, 0);
        
        //call write2DVectorToFile function to write to a file having filename based on threshold
        write2DVectorToFile(image, "./compressedTxt/compressed_image_"+to_string(count)+".txt");
        count++;
    }


    return 0;
}


// int main() {
//     // Initialize the input data
//      base w1,w2,w3,w4;
//     w1.real(1.0);
//     w1.imag(0.0);

//     w2.real(1.0);
//     w2.imag(0.0);

//     w3.real(1.0);
//     w3.imag(0.0);

    
//     w4.real(1.0);
//     w4.imag(0.0);
//     std::vector<base> data_complex = {w1,w2,w3, w4};
//     std::vector<base> data(data_complex.begin(), data_complex.end());

//     // Perform the FFT using fft_radix4
//     fft_radix4(data, false);

//     std::cout << "FFT using radix-4: " << std::endl;
//     // Print the output
//     for (int i = 0; i < data.size(); i++) {
//         std::cout << data[i].real() << " + " << data[i].imag() << "i" << std::endl;
//     }

//     // Perform the FFT using the Cooley-Tukey algorithm
//     std::vector<std::complex<double>> data_cooley_tukey = {1, 1, 1, 1};
//     data_cooley_tukey = fft_cooley_tukey(data_cooley_tukey);

    
//     std::cout << "FFT using cooley-Tukey: " << std::endl;
//     // Print the output
//     for (int i = 0; i < data_cooley_tukey.size(); i++) {
//         std::cout << data_cooley_tukey[i].real() << " + " << data_cooley_tukey[i].imag() << "i" << std::endl;
//     }

//     return 0;
// }