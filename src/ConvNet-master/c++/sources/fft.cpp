#include <iostream>
#include <complex>
#include <random>
#include <vector>
#include <valarray>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
//#include <cilk>
//#include <cilk.h>
/*-------------------------------------------------------------------------
   Perform a 2D FFT inplace given a complex 2D array
   The direction dir, 1 for forward, -1 for reverse
   The size of the array (nx,ny)
   Return false if there are memory problems or
      the dimensions are not powers of 2
*/

//TODO
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

//#define DEBUG 1
#define USE_FFT 1
//#define NO_RECURSIVE 1

const double PI = 3.141592653589793238460;

typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;
typedef std::vector<CArray> C2D;

using namespace std;
void FFT(CArray& x, int m, int dir);
int FFT2D(C2D& x,int nx,int ny,int dir);
void makeSize(C2D& kern,int imgW, int imgH, int kernW, int kernH);
int Powerof2(int n,int *m,int *twopm);

bool convolve2DSlow(C2D& in, C2D& out, int dataSizeX, int dataSizeY,
                    C2D& kernel, int kernelSizeX, int kernelSizeY)
{
  int i, j, m, n, mm, nn;
  int kCenterX, kCenterY;                         // center index of kernel
  Complex sum;                                      // temp accumulation buffer
  int rowIndex, colIndex;
  
  // check validity of params
  
  // find center position of kernel (half of kernel size)
  kCenterX = kernelSizeX / 2;
  kCenterY = kernelSizeY / 2;
  
  for(i=0; i < dataSizeY; ++i){                // rows
    for(j=0; j < dataSizeX; ++j){            // columns
      sum = 0.0;                            // init to 0 before sum
      for(m=0; m < kernelSizeY; ++m){      // kernel rows
	mm = kernelSizeY - 1 - m;       // row index of flipped kernel
	
	for(n=0; n < kernelSizeX; ++n){  // kernel columns
	  nn = kernelSizeX - 1 - n;   // column index of flipped kernel
	  
	  // index of input signal, used for checking boundary
	  rowIndex = i + m - kCenterY;
	  colIndex = j + n - kCenterX;
	  
	  // ignore input samples which are out of bound
	  if(rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
	    sum += in[rowIndex][colIndex] * kernel[mm][nn];
	    //sum += in[dataSizeX * rowIndex + colIndex] * kernel[kernelSizeX * mm + nn];
	}
      }
      out[i][j] = fabs(sum);
    }
  }
  
  return true;
}

void makeSize(C2D& kern,int imgW, int imgH, int kernW, int kernH){
  //We will assume the kernel and image are square
  int dif = imgW - kernW;
  if(dif == 0) return;

  kern.resize(imgH);
  CArray tmp(imgW,0.0);
  for(int row=0; row<imgH; row++){
    kern[row].resize(imgW,0.0);
  }
  return;
}

void conv(C2D& img, C2D& kernel,int imgW, int imgH, int kernW, int kernH){
  int newDim = imgW+kernW-1;
  int nextPow2 = 1;
  while(nextPow2 < newDim) nextPow2 <<= 1;
  makeSize(img,nextPow2,nextPow2,imgW, imgH);

  //Need to pad the kernel to make it the same dimension as the image
  makeSize(kernel,nextPow2,nextPow2,kernW,kernH);
  //Take FFT2D of img and kernel
  cilk_spawn FFT2D(img,nextPow2,nextPow2,1);
  FFT2D(kernel,nextPow2,nextPow2,1);
  //Point-wise multiplications
  for(int i=0; i<nextPow2; i++){
    for(int j=0; j<nextPow2; j++){
      img[i][j] = img[i][j] * kernel[i][j];
    }
  }
  //Take IFFT to get result
  FFT2D(img,nextPow2,nextPow2,-1);

  int dif = newDim - imgW;

  //Erase the first dif rows
  for(int i=0; i<dif; i++){
    img.erase(img.begin());
  }

  //Erase the first dif cols
  for(int i=0; i<imgH; i++){
    for(int j=0; j<dif; j++){
      img[i].erase(img[i].begin());
    }
  }

  //Resize to original image size
  img.resize(imgH);
  cilk_for(int i=0; i<imgH; i++){
    img[i].resize(imgW);
  }
}

//2D FFT Function
int FFT2D(C2D& x,int nx,int ny,int dir)
{
  CArray tmp(nx,0.0);
  cilk_for(int j=0;j<ny;j++){
    cilk_for(int i=0;i<nx;i++){
      tmp[i] = x[i][j];
    }
    int len2 = (int)log2((double)nx); //log(size)
    FFT(tmp,len2,dir);
    cilk_for(int i=0;i<nx;i++){
      x[i][j] = tmp[i];
    }
  }
  
  CArray tmp2(ny,0.0);
  cilk_for(int i=0;i<nx;i++){
    cilk_for(int j=0;j<ny;j++){
      tmp2[j] = x[i][j];
    }
    int len2 = (int)log2((double)ny);
    FFT(tmp2,len2,dir);
    cilk_for(int j=0;j<ny;j++){
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


void FFT(CArray& x, int m, int dir){
  #ifdef NO_RECURSIVE
  long i, i1, i2,j, k, l, l1, l2, n;
  complex <double> tx, t1, u, c;

   /*Calculate the number of points */
   n = 1;
   for(i = 0; i < m; i++) 
      n <<= 1;   

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;

   for (i = 0; i < n-1 ; i++){
      if (i < j)
         swap(x[i], x[j]);

      k = i2;

      while (k <= j) {
         j -= k;
         k >>= 1;
      }

      j += k;
   }

   /* Compute the FFT */
   c.real(-1.0);
   c.imag(0.0);
   l2 = 1;
   for (l = 0; l < m; l++) {
      l1 = l2;
      l2 <<= 1;
      u.real(1.0);
      u.imag(0.0);

      for (j = 0; j < l1; j++) {
         for (i = j; i < n; i += l2) {
            i1 = i + l1;
            t1 = u * x[i1];
            x[i1] = x[i] - t1; 
            x[i] += t1;
         }

         u = u * c;
      }

      c.imag(sqrt((1.0 - c.real()) / 2.0));
      if (dir == 1)
         c.imag(-c.imag());
      c.real(sqrt((1.0 + c.real()) / 2.0));
   }

   // Scaling for inverse transform  
   if (dir == -1) {
      for (i = 0; i < n; i++)
         x[i] /= n;      
   }   
   return;
   
#else
   /* Below is an implementation that uses Recursion */
   int N = (int)x.size();
   //If Inverse then conjugate the data 
   if(dir == -1){
     for(int i=0; i<N; i++){
       x[i] = std::conj(x[i]);
     }
   }

   if(N <= 1) return;
   
   CArray odd(N/2,0.0);
   CArray even(N/2,0.0);
   
   int t1, t2;
   int size = (int)log2((double)(N/2));
   t1 = 0;
   t2 = 0;
   
   //#pragma omp parallel for
   cilk_for(int i=0; i<N; i++){
     if(i%2 == 0){
       even[t1] = x[i];
       t1++;
     }
     else{
       odd[t2] = x[i];
       t2++;
     }
   }

   //Do each FFT in parallel
   cilk_spawn FFT(even,size,dir);
   FFT(odd,size,dir);
   cilk_sync;
   
   //#pragma omp parallel for
   cilk_for(size_t k=0; k < N/2; ++k){
     Complex t = std::polar(1.0, -2*PI*k/N) * odd[k];
     x[k] = even[k] + t;
     x[k+N/2] = even[k] - t;
   }

   
   if(dir == -1){
     //#pragma omp parallel for
     cilk_for(int i=0; i<N; i++){
       x[i] = std::conj(x[i]);
       x[i] = x[i]/((double)N);
     }
   }

  return;
#endif
}

unsigned seed1 = 1;
typedef std::minstd_rand0 G;
typedef std::uniform_int_distribution<> D;
G g;
D d(0, 255);

//Generte random x * y "image"
C2D randImg(int x, int y){
  //make tmp ( empty vector )
  C2D retVec;
  for(int j=0; j<y; j++){
    CArray tmp;
    for(int i=0; i<x; i++){
      int randInt  = d(g);
      //construct the row vector of length x
      complex<double> tmpComplex((double)randInt ,0.0);
      tmp.push_back(tmpComplex);
    }
    //push_back the constructed vector
    retVec.push_back(tmp);
  }
  
  //return the 2d vector constructed
  return retVec;
}

double testFFT(int imageSize, int kernelSize){
  C2D testImg = randImg(imageSize,imageSize);
  C2D testImg2 = randImg(imageSize,imageSize);
  C2D testKern = randImg(kernelSize,kernelSize);
  g.seed(seed1);
  seed1++;

  #ifdef DEBUG
  std::cout << "The below is a random image" << std::endl;
  for(int i=0; i<imageSize; i++){
    for(int j=0; j<imageSize; j++){
      std::cout << testImg[i][j] << " ";
    }
    std::cout << std::endl;
  }
  
  std::cout << "The below is a random kernel" << std::endl;
  for(int i=0; i<kernelSize; i++){
    for(int j=0; j<kernelSize; j++){
      std::cout << testKern[i][j] << " ";
    }
    std::cout << std::endl;
  }
  #endif

  double start = omp_get_wtime();
  //FFT Conv
#ifdef USE_FFT
  conv(testImg, testKern,imageSize,imageSize,kernelSize,kernelSize);
#else
  convolve2DSlow(testImg, testImg2, imageSize, imageSize, testKern, kernelSize, kernelSize);
#endif
  double end = omp_get_wtime();

  #ifdef DEBUG
  std::cout << "The below is convolved" << std::endl;
  for(int i=0; i<imageSize; i++){
    for(int j=0; j<imageSize; j++){
      std::cout << testImg[i][j].real() << " ";
    }
    std::cout << std::endl;
  }
  #endif

  #ifdef DEBUG
  std::cout << "The below is convolved" << std::endl;
  for(int i=0; i<imageSize; i++){
    for(int j=0; j<imageSize; j++){
      std::cout << testImg2[i][j].real() << " ";
    }
    std::cout << std::endl;
  }
  #endif
  return end-start;
}

int main(){
  //Testing benchmark for FFT convolution and normal convolution
  //Run each test case 3 times and average the time spent
  //Might measure memory usage later
  int imageSize = 32; //32x32
  int kernelSize = 3; //3x3
  double t1,t2,t3,avgT;

  /* -------------------------- Small Image Results ------------------------ */
  std::cout << "Small (32x32) FFTConv Results" << std::endl;
  std::cout <<  "_____________________" << std::endl;
  std::cout << "3x3   |   "; 
  //32x32x3x3
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "5x5   |   ";
  //32x32x5x5
  kernelSize = 5;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "7x7   |   ";
  //32x32x7x7
  kernelSize = 7;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "9x9   |   ";
  //32x32x9x9
  kernelSize = 9;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "15x15 |   ";
  //32x32x15x15
  kernelSize = 15;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT << std::endl;
  std::cout << "_________________________" << std::endl;


  /* -------------------------- Large Image Results ------------------------ */
  kernelSize = 9;
  imageSize = 400;
  std::cout << "Large (32x32) FFTConv Results" << std::endl;
  std::cout <<  "_____________________" << std::endl;
  std::cout << "9x9   |   "; 
  //32x32x3x3
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "15x15   |   ";
  //32x32x5x5
  kernelSize = 15;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "35x35   |   ";
  //32x32x7x7
  kernelSize = 35;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "65x65   |   ";
  //32x32x9x9
  kernelSize = 65;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "95x95 |   ";
  //32x32x15x15
  kernelSize = 95;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT << std::endl;

  std::cout <<  std::endl;
  std::cout << "125x125 |   ";
  //32x32x15x15
  kernelSize = 125;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT << std::endl;

  std::cout <<  std::endl;
  std::cout << "155x155 |   ";
  //32x32x15x15
  kernelSize = 155;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT << std::endl;

  std::cout <<  std::endl;
  std::cout << "250x250 |   ";
  //32x32x15x15
  kernelSize = 250;
  t1 = testFFT(imageSize,kernelSize);
  t2 = testFFT(imageSize,kernelSize);
  t3 = testFFT(imageSize,kernelSize);
  avgT = (t1+t2+t3)/3;
  std::cout << avgT << std::endl;
  std::cout << "_________________________" << std::endl;
  
  return 0;
}
