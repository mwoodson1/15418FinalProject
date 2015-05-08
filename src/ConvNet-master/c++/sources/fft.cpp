#include <iostream>
#include <complex>
#include <random>
#include <vector>
#include <valarray>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cilk/cilk.h>
#include <cilk/reducer_opadd.h>
/*-------------------------------------------------------------------------
   Perform a 2D FFT inplace given a complex 2D array
   The direction dir, 1 for forward, -1 for reverse
   The size of the array (nx,ny)
   Return false if there are memory problems or
      the dimensions are not powers of 2
*/

//TODO
//- Get interface to starter code 
//- All of the functions do in-place changes, we might need to change that???
//- Figure out how to write a program using the Xeon Phi's on latedays
//- Parallelize different parts of the starter neural net code
//   - We could parallelize batch learning

//#define DEBUG 1
#define USE_FFT 1
#define NO_RECURSIVE 1

const double PI = 3.141592653589793238460;

typedef std::complex<double> Complex;
typedef std::vector<Complex> CArray;
typedef std::vector<CArray> C2D;

using namespace std;
void FFT(CArray& x, int m, int dir);
int FFT2D(C2D& x,int nx,int ny,int dir);
void makeSize(C2D& kern,int imgW, int imgH, int kernW, int kernH);
int Powerof2(int n,int *m,int *twopm);

bool convolve2DSlow(C2D& input, C2D& out, int dataSizeX, int dataSizeY,
                    C2D& kernel, int kernelSizeX, int kernelSizeY)
{
  int mm, nn;
  int kCenterX, kCenterY;                         // center index of kernel
  //Complex sum;                                      // temp accumulation buffer
  int rowIndex, colIndex;
  int i,j,m,n;

  //cilk::reducer< cilk::op_add<double> > sum(0);
  int sum = 0;
  // check validity of params
  
  // find center position of kernel (half of kernel size)
  kCenterX = kernelSizeX / 2;
  kCenterY = kernelSizeY / 2;
  
  #pragma omp parallel for
  //#pragma vector
  for(i=0; i < dataSizeY; ++i){                // rows
    #pragma omp parallel for private(sum)
    for(j=0; j < dataSizeX; ++j){            // columns
      //cilk::reducer< cilk::op_add<double> > sum(0);                           
      sum = 0;
      //#pragma omp parallel for
      #pragma vector
      for(m=0; m < kernelSizeY; ++m){      // kernel rows
	mm = kernelSizeY - 1 - m;       // row index of flipped kernel
        #pragma vector
	for(n=0; n < kernelSizeX; ++n){  // kernel columns
	  nn = kernelSizeX - 1 - n;   // column index of flipped kernel
	  
	  // index of input signal, used for checking boundary
	  rowIndex = i + m - kCenterY;
	  colIndex = j + n - kCenterX;
	  
	  // ignore input samples which are out of bound
	  if(rowIndex >= 0 && rowIndex < dataSizeY && colIndex >= 0 && colIndex < dataSizeX)
	    //*sum += in[rowIndex][colIndex].real() * kernel[mm][nn].real();
	    //#pragma omp atomic capture
	    sum += input[rowIndex][colIndex].real() * kernel[mm][nn].real();
	}
      }
      //out[i][j] = fabs(sum.get_value());
      out[i][j] = fabs(sum);
    }
  }
  
  return true;
}

void makeSize(C2D& kern,int imgW, int imgH, int kernW, int kernH){
  //We will assume the kernel and image are square
  int row;
  int dif = imgW - kernW;
  if(dif == 0) return;

  kern.resize(imgH);
  CArray tmp(imgW,0.0);
  //#pragma omp parallel for if(imgH > 32)
  #pragma loop_count min(32), max(2048), avg(1024)
  for(row=0; row<imgH; row++){
    kern[row].resize(imgW,0.0);
  }
  return;
}

void conv(C2D& img, C2D& kernel,int imgW, int imgH, int kernW, int kernH){
  int i, j;
  int newDim = imgW+kernW-1;
  int nextPow2 = 1;
  while(nextPow2 < newDim) nextPow2 <<= 1;
  makeSize(kernel,nextPow2,nextPow2,kernW,kernH);
  makeSize(img,nextPow2,nextPow2,imgW, imgH);
  //Need to pad the kernel to make it the same dimension as the image
  //makeSize(kernel,nextPow2,nextPow2,kernW,kernH);
  //cilk_sync;
  //Take FFT2D of img and kernel
  FFT2D(kernel,nextPow2,nextPow2,1);
  FFT2D(img,nextPow2,nextPow2,1);
  //Point-wise multiplications
  //This can be vectorized
  //#pragma omp parallel for if(nextPow2 > 32)
#pragma loop_count min(50),max(2048),avg(1024)
  for(i=0; i<nextPow2; i++){
    //#pragma simd
    for(j=0; j<nextPow2; j++){
      img[i][j] = img[i][j] * kernel[i][j];
    }
  }
  //Take IFFT to get result
  FFT2D(img,nextPow2,nextPow2,-1);

  int dif = newDim - imgW;

  //Erase the first dif rows
#pragma loop_count min(0), max(1024), avg(500)
  for(i=0; i<dif; i++){
    img.erase(img.begin());
  }

  //Erase the first dif cols
  //#pragma omp parallel for if(imgH > 32)
#pragma loop_count min(32), max(1024)
  for(i=0; i<imgH; i++){
    for(j=0; j<dif; j++){
      img[i].erase(img[i].begin());
    }
  }

  //Resize to original image size
  img.resize(imgH);
  #pragma omp parallel for if(imgH > 32)
  //#pragma loop_count min(32), max(1050)
  for(i=0; i<imgH; i++){
    img[i].resize(imgW);
  }
}

//2D FFT Function
int FFT2D(C2D& x,int nx,int ny,int dir)
{
  int i, j;
  CArray tmp(nx,0.0);
  int len2 = (int)log2((double)ny);
  for(j=0;j<ny;j++){
    #pragma simd
    for(i=0;i<nx;i++){
      tmp[i] = x[i][j];
    }
    FFT(tmp,len2,dir);
    #pragma simd
    for(i=0;i<nx;i++){
      x[i][j] = tmp[i];
    }
  }

  len2 = (int)log2((double)nx);
  #pragma omp parallel for if(ny > 32)
  for(i=0;i<ny;i++){
    FFT(x[i],len2,dir);
  }
  return(1);
}

void FFT(CArray& x, int m, int dir){
  #ifdef NO_RECURSIVE
  long i,j,l, i1, i2, k, l1, l2, n;
  complex <double> tx, t1, u, c;

   /*Calculate the number of points */
   n = 1;
   //for(i = 0; i < m; i++) 
   n <<= m;   

   /* Do the bit reversal */
   i2 = n >> 1;
   j = 0;

   //#pragma omp parallel for shared(x,k,j)
   for(i = 0; i < n-1 ; i++){
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
   #pragma omp parallel for if(m > 100)
   for(l = 0; l < m; l++) {
     l1 = 1 << l;
     l2 = 1 << (l+1);
     u.real(1.0);
     u.imag(0.0);
     //#pragma loop_count min(0),max(15)
     for(j = 0; j < l1; j++) {
       //#pragma omp parallel for shared(x)
       //#pragma simd
       for(i = j; i < n; i += l2) {
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
     #pragma simd
     for(i = 0; i < n; i++)
         x[i] /= n;      
   }   
   return;
   
#else
   /* Below is an implementation that uses Recursion */
   int N = (int)x.size();
   int half = N/2;
   Complex t;
   //If Inverse then conjugate the data 
   if(dir == -1){
     #pragma simd
     for(int i=0; i<N; i++){
       x[i] = std::conj(x[i]);
     }
   }

   if(N <= 1) return;
   
   CArray odd(half,0.0);
   CArray even(half,0.0);
   
   int t1, t2;
   int size = (int)log2((double)(N/2));
   t1 = 0;
   t2 = 0;
   
   //the below will break code
   //#pragma omp parallel for if(N >100) firstprivate(t1,t2)
   #pragma loop_count min(2), max(500)
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

   //Do each FFT in parallel
   cilk_spawn FFT(even,size,dir);
   FFT(odd,size,dir);
   cilk_sync;
   
   //REPLACED N/2 WITH HALF
   //#pragma omp parallel for if(N>50)
   for(size_t k=0; k < half; ++k){
     t = std::polar(1.0, -2*PI*k/N) * odd[k];
     x[k] = even[k] + t;
     x[k+half] = even[k] - t;
   }

   double N2 = (double)N;
   if(dir == -1){
     for(int i=0; i<N; i++){
       x[i] = std::conj(x[i]);
       x[i] = x[i]/N2;
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
  #ifdef DEBUG
  std::cout << "The below is convolved" << std::endl;
  for(int i=0; i<imageSize; i++){
    for(int j=0; j<imageSize; j++){
      std::cout << testImg[i][j].real() << " ";
    }
    std::cout << std::endl;
  }
  #endif
#else
  convolve2DSlow(testImg, testImg2, imageSize, imageSize, testKern, kernelSize, kernelSize);
  #ifdef DEBUG
  std::cout << "The below is convolved" << std::endl;
  for(int i=0; i<imageSize; i++){
    for(int j=0; j<imageSize; j++){
      std::cout << testImg2[i][j].real() << " ";
    }
    std::cout << std::endl;
  }
  #endif
#endif
  double end = omp_get_wtime();

  return end-start;
}

int main(){
  //Testing benchmark for FFT convolution and normal convolution
  //Run each test case 3 times and average the time spent
  //Might measure memory usage later
  int imageSize = 32; //32x32
  int kernelSize = 3; //3x3
  int i;
  double t1,t2,t3,t4,t5,tTotal,avgT;
  tTotal = 0;

#ifdef USE_FFT
  std::cout << "Using FFT: ";
#ifdef NO_RECURSIVE
  std::cout << "Non-recursive" << std::endl;
#else
  std::cout << "Recursive" << std::endl;
#endif
#else
  std::cout << "Normal Conv" << std::endl;
#endif

  /* -------------------------- Small Image Results ------------------------ */
  std::cout << "Small (32x32)" << std::endl;
  std::cout <<  "_____________________" << std::endl;
  std::cout << "3x3   |   "; 
  //32x32x3x3
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "5x5   |   ";
  //32x32x5x5
  kernelSize = 5;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "7x7   |   ";
  //32x32x7x7
  kernelSize = 7;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "9x9   |   ";
  //32x32x9x9
  kernelSize = 9;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "15x15 |   ";
  //32x32x15x15
  kernelSize = 15;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT << std::endl;
  std::cout << "_________________________" << std::endl;


  /* -------------------------- Large Image Results ------------------------ */
  kernelSize = 3;
  imageSize = 400;
  std::cout << "Large (400x400) " << std::endl;
  std::cout <<  "_____________________" << std::endl;
  std::cout << "3x3   |   "; 
  //32x32x3x3
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout << std::endl;
  kernelSize = 5;
  imageSize = 400;
  std::cout << "5x5   |   "; 
  //32x32x3x3
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout << std::endl;
  kernelSize = 7;
  imageSize = 400;
  std::cout << "7x7   |   "; 
  //32x32x3x3
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;


  std::cout << std::endl;
  kernelSize = 9;
  imageSize = 400;
  std::cout << "9x9   |   "; 
  //32x32x3x3
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "15x15   |   ";
  //32x32x5x5
  kernelSize = 15;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "35x35   |   ";
  //32x32x7x7
  kernelSize = 35;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "65x65   |   ";
  //32x32x9x9
  kernelSize = 65;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT;

  std::cout <<  std::endl;
  std::cout << "95x95 |   ";
  //32x32x15x15
  kernelSize = 95;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT << std::endl;

  std::cout <<  std::endl;
  std::cout << "125x125 |   ";
  //32x32x15x15
  kernelSize = 125;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT << std::endl;

  std::cout <<  std::endl;
  std::cout << "155x155 |   ";
  //32x32x15x15
  kernelSize = 155;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT << std::endl;

  std::cout <<  std::endl;
  std::cout << "250x250 |   ";
  //32x32x15x15
  kernelSize = 250;
  for(i=0; i<10; i++){
    t1 = testFFT(imageSize,kernelSize);
    tTotal += t1;
  }
  avgT = tTotal/10;
  tTotal = 0;
  std::cout << avgT << std::endl;
  std::cout << "_________________________" << std::endl;
  
  return 0;
}
