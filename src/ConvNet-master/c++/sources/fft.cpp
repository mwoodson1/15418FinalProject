#include <iostream>
#include <complex>
#include <stdlib.h>
#include <valarray>
#include <vector>
/*-------------------------------------------------------------------------
   Perform a 2D FFT inplace given a complex 2D array
   The direction dir, 1 for forward, -1 for reverse
   The size of the array (nx,ny)
   Return false if there are memory problems or
      the dimensions are not powers of 2
*/

//TODO
//- Implement in IFFT
//- Work on parallelization
//- Get interface to starter code 
//- All of the functions do in-place changes, we might need to change that???
//- Run performance tests for different sized kernels on 32x32
//- Run performance tests for different sized kernels on real sized images.
//- Figure out how to write a program using the Xeon Phi's on latedays
//- Parallelize different parts of the starter neural net code
//   - We could parallelize batch learning
//   - A lot of his functions use loops I think can be vectorized
//   - 
// ... misc. stuff may be added later

const double PI = 3.141592653589793238460;

typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;
typedef std::vector<CArray> C2D;

using namespace std;
void FFT(CArray& x);
int FFT2D(C2D& x,int nx,int ny,int dir);
void makeSize(C2D& kern,int imgW, int imgH, int kernW, int kernH);
int Powerof2(int n,int *m,int *twopm);

void makeSize(C2D& kern,int imgW, int imgH, int kernW, int kernH){
  //We will assume the kernel and image are square
  int dif = imgW - kernW;
  if(dif == 0) return;

  std::cout << dif << std::endl;
  kern.resize(imgH);
  CArray tmp(imgW,0.0);
  for(int row=0; row<imgH; row++){
    kern[row].resize(imgW,0.0);
  }
  return;
}

void conv(C2D& img, C2D& kernel,C2D& out, int imgW, int imgH, int kernW, int kernH){
  //Need to pad the kernel to make it the same dimension as the image
  makeSize(kernel,imgW,imgH,kernW,kernH);
  //Take FFT2D of img and kernel
  FFT2D(img,imgW,imgH,1);
  FFT2D(kernel,imgW,imgH,1);
  //Point-wise multiplications
  for(int i=0; i<imgH; i++){
    for(int j=0; j<imgW; j++){
      img[i][j] = img[i][j] * kernel[i][j];
    }
  }
  //Take IFFT to get result
  FFT2D(img,imgW,imgH,-1);
}

//Need to implement the inverse FFT and FFT2D
int FFT2D(C2D& x,int nx,int ny,int dir)
{
  int i,j;

  CArray tmp(nx,0.0);
  for(j=0;j<ny;j++){
    for(i=0;i<nx;i++){
      tmp[i] = x[i][j];
    }
    FFT(tmp);
    for(i=0;i<nx;i++){
      x[i][j] = tmp[i];
    }
  }
  
  CArray tmp2(ny,0.0);
  for(i=0;i<nx;i++){
    for(j=0;j<ny;j++){
      tmp2[j] = x[i][j];
    }
    FFT(tmp2);
    for(j=0;j<ny;j++){
      x[i][j] = tmp2[j];
    }
  }
  return(1);
}

/*-------------------------------------------------------------------------
   This computes an in-place complex-to-complex FFT
   x and y are the real and imaginary arrays of 2^m points.
   dir =  1 gives forward transform
   dir = -1 gives reverse transform

     Formula: forward
                  N-1
                  ---
              1   \          - j k 2 pi n / N
      X(n) = ---   >   x(k) e                    = forward transform
              N   /                                n=0..N-1
                  ---
                  k=0

      Formula: reverse
                  N-1
                  ---
                  \          j k 2 pi n / N
      X(n) =       >   x(k) e                    = forward transform
                  /                                n=0..N-1
                  ---
                  k=0
*/


void FFT(CArray& x){
  const size_t N = x.size();
  if(N <= 1) return;
  
  int t1, t2;
  t1 = 0;
  t2 = 0;

  CArray even(N/2,0);//x[std::slice(0,N/2,2)];
  CArray odd(N/2,0); //x[std::slice(1,N/2,2)];
  for(int i=0; i<N; i++){
    if(i%2 == 0){
      even[t1] = x[i];
      t1++;
    }
    else{
      odd[t2] = x[i];
      t2++;
    }
  }

  FFT(even);
  FFT(odd);

  for(size_t k=0; k < N/2; ++k){
    Complex t = std::polar(1.0, -2*PI*k/N) * odd[k];
    x[k] = even[k] + t;
    x[k+N/2] = even[k] - t;
  }
}

/*-------------------------------------------------------------------------
   Calculate the closest but lower power of two of a number
   twopm = 2**m <= n
   Return TRUE if 2**m == n
*/
int Powerof2(int n,int *m,int *twopm)
{
   if (n <= 1) {
      *m = 0;
      *twopm = 1;
      return(0);
   }

   *m = 1;
   *twopm = 2;
   do {
      (*m)++;
      (*twopm) *= 2;
   } while (2*(*twopm) <= n);

   if (*twopm != n)
      return(0);
   else
      return(1);
}

int main(){
  
  //Testing FFT
  const Complex test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
  CArray data(test,test + sizeof(test) / sizeof(Complex));
  FFT(data);
  
  std::cout << "fft" << std::endl;
  for(int i=0; i<8; i++){
    std::cout << data[i] << std::endl;
  }

  //Testing FFT2D
  const Complex test1[] = {1.0, 1.0};
  CArray row(test1,test1 + sizeof(test1) / sizeof(Complex));
  C2D arr;
  arr.push_back(row);
  arr.push_back(row);
  std::cout << arr.size() << std::endl;
  std::cout << arr[0].size() << std::endl;
  std::cout << "2dfft" << std::endl;
  FFT2D(arr,2,2,1);
  
  std::cout << arr.size() << std::endl;
  std::cout << arr[0].size() << std::endl;
  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      std::cout << arr[i][j] << std::endl;;
    }
  }

  //Testing resize
   makeSize(arr,8,8,2,2);

  for(int i=0; i<8; i++){
    for(int j=0; j<8; j++){
      std::cout << arr[i][j];
    }
    std::cout << std::endl;
  }
  
   return 0;
}
