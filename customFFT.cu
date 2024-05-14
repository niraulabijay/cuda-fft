#include <vector>
#include <bits/stdc++.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
// #include <iostream>


using namespace std;

#define N 1000
#define BALANCE 2



typedef complex<float> base;
typedef float2 Complex_my;

class FFT
{
public:
    /**
     * parallel FFT transform and inverse transform
     * Arguments vector of complex numbers, invert, balance, number of threads
     * Perform inplace transform
     */
    void fft(vector<base> &a, bool invert)
    {
        // Performing Bit reversal ordering
        int n = (int)a.size();

        for (int i = 1, j = 0; i < n; ++i)
        {
            int bit = n >> 1;
            for (; j >= bit; bit >>= 1)
                j -= bit;
            j += bit;
            if (i < j)
                swap(a[i], a[j]);
        }

        // Iteratinve FFT
        // This part of FFT is parallelizable
        for (int len = 2; len <= n; len <<= 1)
        {
            double ang = 2 * M_PI / len * (invert ? 1 : -1);
            base wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len)
            {
                base w(1);
                for (int j = 0; j < len / 2; ++j)
                {
                    base u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }

        if (invert)
            for (int i = 0; i < n; ++i)
                a[i] /= n;
        return;
    }

    /**
     * Performs 2D FFT 
     * takes vector of complex vectors, invert and verbose as argument
     * performs inplace FFT transform on input vector
     */
    void fft2D(vector<vector<base>> &a, bool invert, int verbose = 0)
    {
        auto matrix = a;
        // Transform the rows
        if (verbose > 0)
            cout << "Transforming Rows" << endl;

        for (auto i = 0; i < matrix.size(); i++)
        {
            //cout<<i<<endl;
            fft(matrix[i], invert);
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
        for (auto i = 0; i < matrix.size(); i++)
            fft(matrix[i], invert);

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
    vector<int> mult(vector<int> a, vector<int> b)
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
        fft(fa, false), fft(fb, false);

        // performing point wise multipication of points
        for (size_t i = 0; i < n; ++i)
            fa[i] *= fb[i];

        // Performing Inverse transform
        fft(fa, true);

        // Saving the real part as it will be the result
        vector<int> res;
        res.resize(n);
        for (size_t i = 0; i < n; ++i)
            res[i] = int(fa[i].real() + 0.5);

        return res;
    }

};


int main()
{
    vector<int> a = {1,1};
    vector<int> b = {1,2,3};
    auto multiplier = FFT();
    cout << "A = "<< a;
    cout << "B = "<< b;
    cout << "A * B = "<< multiplier.mult(a, b)<<endl;
    multiplier.mult(a, b);
    /*
    vector<int> fa(N);
    generate(fa.begin(), fa.end(), rand);
    vector<int> fb(N);
    generate(fb.begin(), fb.end(), rand);
    freopen("out.txt", "w", stdout);
    for(int threads = 1; threads <= 1024; threads*=2){
	cerr << "For threads= " << threads << endl;
        /// For Parallel
        auto start = high_resolution_clock::now(); 

        auto result_parallel = mult(fa, fb, BALANCE, threads);

        auto stop = high_resolution_clock::now(); 
        auto duration = duration_cast<microseconds>(stop - start); 
      
        cout << threads << " " << duration.count();


        /// For Sequential
        auto multiplier = FFT();
        
        start = high_resolution_clock::now(); 
        auto result_sequential = multiplier.mult(fa, fb);

        stop = high_resolution_clock::now(); 
        duration = duration_cast<microseconds>(stop - start); 

        cout << " " << duration.count();

        cout << " " << (result_parallel == result_sequential) << endl;
        cout << endl;
    }
    */
    return 0;
}
