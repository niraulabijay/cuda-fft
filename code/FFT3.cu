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
    Complex_my C;
    C.x = A.x + B.x;
    C.y = A.y + B.y;
    return C;
}

/**
 *  Inverse of Complex_my Number
 */
static __device__ __host__ inline Complex_my Inverse(Complex_my A)
{
    Complex_my C;
    C.x = -A.x;
    C.y = -A.y;
    return C;
}

/**
 *  Multipication of Complex_my Numbers
 */
static __device__ __host__ inline Complex_my Multiply(Complex_my A, Complex_my B)
{
    Complex_my C;
    C.x = A.x * B.x - A.y * B.y;
    C.y = A.y * B.x + A.x * B.y;
    return C;
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
        applyHanningWindow(matrix[i]);
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
        applyHanningWindow(matrix[i]);
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


void fft3(vector<vector<vector<base>>> &a, bool invert, bool balance, int threads) {
    int nx = a.size();
    int ny = a[0].size();
    int nz = a[0][0].size();

    // Perform 2D FFT on each slice along the z dimension
    for (int z = 0; z < nz; z++) {
        vector<vector<base>> slice(nx, vector<base>(ny));
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                slice[x][y] = a[x][y][z];
            }
        }
        fft2D(slice, invert, balance, threads);
        for (int x = 0; x < nx; x++) {
            for (int y = 0; y < ny; y++) {
                a[x][y][z] = slice[x][y];
            }
        }
    }

    // Transpose the data so that the z dimension becomes the y dimension
    vector<vector<vector<base>>> transposed(nz, vector<vector<base>>(nx, vector<base>(ny)));
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                transposed[z][x][y] = a[x][y][z];
            }
        }
    }

    // Perform 2D FFT on each slice along the new z dimension
    for (int z = 0; z < nz; z++) {
        fft2D(transposed[z], invert, balance, threads);
    }

    // Transpose the data back to its original orientation
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                a[x][y][z] = transposed[z][x][y];
            }
        }
    }
}



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

    return data;
}

bool areEqual(const vector<vector<vector<base>>>& vec1, const vector<vector<vector<base>>>& vec2) {
    if (vec1.size() != vec2.size()) {
        return false;
    }

    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec1[i].size() != vec2[i].size()) {
            return false;
        }

        for (size_t j = 0; j < vec1[i].size(); ++j) {
            if (vec1[i][j].size() != vec2[i][j].size()) {
                return false;
            }

            for (size_t k = 0; k < vec1[i][j].size(); ++k) {
                if (abs(vec1[i][j][k] - vec2[i][j][k]) > 1e-6) {
                    return false;
                }
            }
        }
    }

    return true;
}


int main()
{
    // cv::Mat image_M;
    // image_M = imread("squirrel.jpg", IMREAD_GRAYSCALE);
    // if (!image_M.data)
    // {
    //     cout << "Could not open or find the image" << std::endl;
    //     return -1;
    // }
    

    // cv::imwrite("original.jpg", image_M);
    // vector<vector<uint>> image(image_M.rows, vector<uint>(image_M.cols));
    // for (int i = 0; i < image_M.rows; ++i)
    //     for (int j = 0; j < image_M.cols; ++j)
    //         image[i][j] = uint(image_M.at<uint>(i, j));

    // auto temp_image = image;

    // std::vector<std::vector<uint>> image = read_2d_vector("image2d.txt");

    // write2DVectorToFile(image, "originalImage.txt");
    
    // // fft.compress_image(image, 0.00005, 0);
    // freopen("out.txt", "w", stdout);
    // for(double thresh = 0.000001; thresh < 1; thresh*=10)
    // {
    //     cout << "For thresh= " << thresh << endl;
    //     compress_image(image, thresh, BALANCE, 10, 0);
    //     // for (int i = 0; i < image_M.rows; ++i)
    //     //     for (int j = 0; j < image_M.cols; ++j)
    //     //         image_M.at<uint>(i, j) = image[i][j];
    //     // string s = "compressed_";
    //     // s = s+to_string(thresh);
    //     // s += ".jpg";
    //     // cv::imwrite(s, image_M);
        
    //     //call write2DVectorToFile function to write to a file having filename based on threshold
    //     write2DVectorToFile(image, "compressed_image_"+to_string(thresh)+".txt");
    // }
    // Initialize 3D vector with sample data

    int nx = 10, ny = 10, nz = 10;  // replace with your actual dimensions
    vector<vector<vector<base>>> data(nx, vector<vector<base>>(ny, vector<base>(nz)));
    vector<vector<vector<base>>> dataOrig(nx, vector<vector<base>>(ny, vector<base>(nz)));

    // Fill the 3D vector with your sample data
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                // replace with your actual data
                data[x][y][z] = 1.0;
                dataOrig[x][y][z] = 1.0;
            }
        }
    }

    // Invoke the fft3 function
    bool invert = false; 
    bool balance = false;  
    int threads = 4;  
    fft3(data, invert, balance, threads);

    // perform inverse FFT3
    fft3(data, true, balance, threads);

    // Check the result
    bool correct = true;
    double epsilon = 1e-1; // or a value that is acceptable in your context
    for (int x = 0; x < nx; x++) {
        for (int y = 0; y < ny; y++) {
            for (int z = 0; z < nz; z++) {
                cout << data[x][y][z] << " ";
                if (abs(data[x][y][z] - dataOrig[x][y][z]) > epsilon) {
                    correct = false;
                    // break;
                }
            }
            // if (!correct) break;
        }
        // if (!correct) break;
    }

    if (correct) {
        cout << "FFT3 is working correctly." << endl;
    } else {
        cout << "FFT3 is not working correctly." << endl;
    }


    return 0;
}

