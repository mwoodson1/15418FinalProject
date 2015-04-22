#include <iostream>
#include <complex>
/*-------------------------------------------------------------------------
   Perform a 2D FFT inplace given a complex 2D array
   The direction dir, 1 for forward, -1 for reverse
   The size of the array (nx,ny)
   Return false if there are memory problems or
      the dimensions are not powers of 2
*/
int FFT2D(complex <double> c[][],int nx,int ny,int dir)
{
   int i,j;
   int m,twopm;
   double *real,*imag;
   complex <double> *tmp;

   /* Transform the rows */
   //real = (double *)malloc(nx * sizeof(double));
   //imag = (double *)malloc(nx * sizeof(double));
   tmp  = (complex <double> *)malloc(nx * sizeof(complex <double>));
   if (real == NULL || imag == NULL)
      return(FALSE);
   if (!Powerof2(nx,&m,&twopm) || twopm != nx)
      return(FALSE);
   for (j=0;j<ny;j++) {
      for (i=0;i<nx;i++) {
         tmp[i] = c[i][j];
         //real[i] = c[i][j].real;
         //imag[i] = c[i][j].imag;
      }
      FFT(dir,m,tmp);
      for (i=0;i<nx;i++) {
         c[i][j] = tmp[i][j];
         //c[i][j].real = real[i];
         //c[i][j].imag = imag[i];
      }
   }
   free(real);
   free(imag);
   free(tmp);

   /* Transform the columns */
   //real = (double *)malloc(ny * sizeof(double));
   //imag = (double *)malloc(ny * sizeof(double));
   tmp  = (complex <double> *)malloc(ny * sizeof(complex <double>));
   if (real == NULL || imag == NULL)
      return(FALSE);
   if (!Powerof2(ny,&m,&twopm) || twopm != ny)
      return(FALSE);
   for (i=0;i<nx;i++) {
      for (j=0;j<ny;j++) {
         tmp[i] = c[i][j];
         //real[j] = c[i][j].real;
         //imag[j] = c[i][j].imag;
      }
      FFT(dir,m,tmp);
      for (j=0;j<ny;j++) {
         c[i][j] = tmp[i][j];
         //c[i][j].real = real[j];
         //c[i][j].imag = imag[j];
      }
   }
   //free(real);
   //free(imag);
   free(tmp);

   return(TRUE);
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
void FFT(int dir, long m, complex <double> x[])
{
   long i, i1, i2,j, k, l, l1, l2, n;
   complex <double> tx, t1, u, c;

   /*Calculate the number of points */
   n = 1;
   for(i = 0; i < m; i++) 
      n <<= 1;   

   /* Do the bit reversal */
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

   /* Compute the FFT */
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

   /* Scaling for forward transform */
   if (dir == 1) 
   {
      for (i = 0; i < n; i++)
         x[i] /= n;      
   }   
   return;
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
      return(FALSE);
   }

   *m = 1;
   *twopm = 2;
   do {
      (*m)++;
      (*twopm) *= 2;
   } while (2*(*twopm) <= n);

   if (*twopm != n)
      return(FALSE);
   else
      return(TRUE);
}

int main(){
   int t;
   std::complex<double> test[2][2] = {{{1, 1}, {1,1}}, {{1, 1}, {1,1}}};
   t = FFT2D(test,2,2,1);
   // output each array element's value                      
   for ( int i = 0; i < 2; i++ ){
      for ( int j = 0; j < 2; j++ )
      {
         cout << "a[" << i << "][" << j << "]: ";
         cout << test[i][j]<< endl;
      }
   }
   return 0;
}