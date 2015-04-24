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

const double PI = 3.141592653589793238460;

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;
typedef std::vector<CArray> C2D;

using namespace std;
void FFT(CArray& x);
//void FFT(int dir, long m, std::complex<double> *x);
int Powerof2(int n,int *m,int *twopm);

void conv(C2D& img, C2D& kernel,C2D& out, int imgW, int imgH, int kernW, int kernH){
  //Take FFT2D of img and kernel
  //Point-wise multiplications
  //Take IFFT to get result
}

int FFT2D(C2D& x,int nx,int ny,int dir)
{
  int i,j;

  CArray tmp(0.0,nx);
  for(j=0;j<ny;j++){
    for(i=0;i<nx;i++){
      tmp[i] = x[i][j];
    }
    FFT(tmp);
    for(i=0;i<nx;i++){
      x[i][j] = tmp[i];
    }
  }

  CArray tmp2(0.0,ny);
  for(i=0;i<nx;i++){
    for(j=0;j<ny;j++){
      tmp2[j] = x[i][j];
    }
    FFT(tmp2);
    for(j=0;j<ny;j++){
      x[i][j] = tmp2[j];
    }
  }
   
  /*
   int i,j;
   int m,twopm;
   double *real,*imag;
   std::complex<double> *tmp = new std::complex<double>[nx];

   //* Transform the rows 
   //real = (double *)malloc(nx * sizeof(double));
   //imag = (double *)malloc(nx * sizeof(double));
   //tmp  = (std::complex<double> *)malloc(nx * sizeof( std::complex<double> ));
   //if (real == NULL || imag == NULL)
   // return(0);
   //if (!Powerof2(nx,&m,&twopm) || twopm != nx)
   //return(0);
   for (j=0;j<ny;j++) {
      for (i=0;i<nx;i++) {
         tmp[i] = c[i][j];
         //real[i] = c[i][j].real;
         //imag[i] = c[i][j].imag;
      }
      FFT(dir,m,tmp);
      for (i=0;i<nx;i++) {
         c[i][j] = tmp[i];
         //c[i][j].real = real[i];
         //c[i][j].imag = imag[i];
      }
   }
   free(real);
   free(imag);
   //free(tmp);
   delete [] tmp;

   //* Transform the columns 
   //real = (double *)malloc(ny * sizeof(double));
   //imag = (double *)malloc(ny * sizeof(double));
   tmp = new std::complex<double>[ny];
   //tmp  = (std::complex<double> *)malloc(ny * sizeof( std::complex<double> ));
   //if (real == NULL || imag == NULL)
   // return(0);
   //if (!Powerof2(ny,&m,&twopm) || twopm != ny)
   // return(0);
   for (i=0;i<nx;i++) {
      for (j=0;j<ny;j++) {
         tmp[i] = c[i][j];
         //real[j] = c[i][j].real;
         //imag[j] = c[i][j].imag;
      }
      FFT(dir,m,tmp);
      for (j=0;j<ny;j++) {
         c[i][j] = tmp[j];
         //c[i][j].real = real[j];
         //c[i][j].imag = imag[j];
      }
   }
   //free(real);
   //free(imag);
   //free(tmp);
   delete [] tmp;
  */
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

  CArray even = x[std::slice(0,N/2,2)];
  CArray odd = x[std::slice(1,N/2,2)];

  FFT(even);
  FFT(odd);

  for(size_t k=0; k < N/2; ++k){
    Complex t = std::polar(1.0, -2*PI*k/N) * odd[k];
    x[k] = even[k] + t;
    x[k+N/2] = even[k] - t;
  }
}

/*
void FFT(int dir, long m, std::complex<double> *x)
{
   long i, i1, i2,j, k, l, l1, l2, n;
   std::complex<double> tx, t1, u, c;

   //*Calculate the number of points 
   n = 1;
   for(i = 0; i < m; i++) 
      n <<= 1;   

   //* Do the bit reversal 
   i2 = n >> 1;
   j = 0;

   for (i = 0; i < n-1 ; i++)
   {
      if (i < j)
         swap(x[i], x[j]);

      k = i2;

      while (k <= j) 
     {
         j -= k;
         k >>= 1;
      }

      j += k;
   }

   //* Compute the FFT 
   c.real(-1.0);
   c.imag(0.0);
   l2 = 1;
   for (l = 0; l < m; l++) 
   {
      l1 = l2;
      l2 <<= 1;
      u.real(1.0);
      u.imag(0.0);

      for (j = 0; j < l1; j++) 
     {
         for (i = j; i < n; i += l2) 
       {
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

   //* Scaling for forward transform 
   if (dir == 1) 
   {
      for (i = 0; i < n; i++)
         x[i] /= n;      
   }   
   return;
}
*/

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
  
  const Complex test[] = {1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0};
  CArray data(test,8);
  FFT(data);
  std::cout << "fft" << std::endl;
  for(int i=0; i<8; i++){
    std::cout << data[i] << std::endl;
  }

  const Complex test1[] = {1.0, 1.0};
  CArray test11(test1,2);
  std::vector<CArray> test2;
  test2.push_back(test11);
  test2.push_back(test11);
  std::cout << "2dfft" << std::endl;
  FFT2D(test2,2,2,1);

  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      std::cout << test2[i][j] << std::endl;;
    }
  }
  /*
   int t;
   int size = sizeof(std::complex<double>);
   std::complex<double>** test = new std::complex<double>*[2];
   for(int i=0; i<2; i++){
     test[i] = new std::complex<double>[2];
   }
   test[0][0] = 1.0;
   test[0][1] = 1.0;
   test[1][0] = 1.0;
   test[1][1] = 1.0;
     //std::complex<double> test[2][2] = {{{1, 1}, {1,1}}, {{1, 1}, {1,1}}};
   t = FFT2D(test,2,2,1);
   // output each array element's value                      
   for ( int i = 0; i < 2; i++ ){
      for ( int j = 0; j < 2; j++ )
      {
         cout << "a[" << i << "][" << j << "]: ";
         cout << test[i][j]<< endl;
      }
      }*/
   return 0;
}
