/* 
 * -- High Performance Computing Linpack Benchmark (HPL)                
 *    HPL - 2.2 - February 24, 2016                          
 *    Antoine P. Petitet                                                
 *    University of Tennessee, Knoxville                                
 *    Innovative Computing Laboratory                                 
 *    (C) Copyright 2000-2008 All Rights Reserved                       
 *                                                                      
 * -- Copyright notice and Licensing terms:                             
 *                                                                      
 * Redistribution  and  use in  source and binary forms, with or without
 * modification, are  permitted provided  that the following  conditions
 * are met:                                                             
 *                                                                      
 * 1. Redistributions  of  source  code  must retain the above copyright
 * notice, this list of conditions and the following disclaimer.        
 *                                                                      
 * 2. Redistributions in binary form must reproduce  the above copyright
 * notice, this list of conditions,  and the following disclaimer in the
 * documentation and/or other materials provided with the distribution. 
 *                                                                      
 * 3. All  advertising  materials  mentioning  features  or  use of this
 * software must display the following acknowledgement:                 
 * This  product  includes  software  developed  at  the  University  of
 * Tennessee, Knoxville, Innovative Computing Laboratory.             
 *                                                                      
 * 4. The name of the  University,  the name of the  Laboratory,  or the
 * names  of  its  contributors  may  not  be used to endorse or promote
 * products  derived   from   this  software  without  specific  written
 * permission.                                                          
 *                                                                      
 * -- Disclaimer:                                                       
 *                                                                      
 * THIS  SOFTWARE  IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES,  INCLUDING,  BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
 * OR  CONTRIBUTORS  BE  LIABLE FOR ANY  DIRECT,  INDIRECT,  INCIDENTAL,
 * SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL DAMAGES  (INCLUDING,  BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA OR PROFITS; OR BUSINESS INTERRUPTION)  HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT,  STRICT LIABILITY,  OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. 
 * ---------------------------------------------------------------------
 */ 
/*
 * Include files
 */
#include "hpl.h"
#include "unistd.h"
#include <math.h>
#include <assert.h>
#if _POSIX_TIMERS
#include <time.h>
#define HAVE_CLOCKGETTIME 1
#else
#include <sys/time.h>
#define HAVE_GETTIMEOFDAY 1
#endif

FILE *get_measure_file() {
#ifdef SMPI_MEASURE
    static FILE *measure_file=NULL;
    if(!measure_file) {
        int my_rank;
        char filename[50];
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        sprintf (filename, "blas_%d.trace", my_rank);
        measure_file=fopen(filename, "w");
        if(!measure_file) {
            fprintf(stderr, "Error opening file %s\n", filename);
            exit(1);
        }
    }
    return measure_file;
#endif
}

#ifdef HAVE_CLOCKGETTIME
#define PRECISION 1000000000.0
#elif HAVE_GETTIMEOFDAY
#define PRECISION 1000000.0
#else
#define PRECISION 1
#endif

timestamp_t get_time(){
#ifdef HAVE_CLOCKGETTIME
    struct timespec tp;
    clock_gettime (CLOCK_REALTIME, &tp);
    return (tp.tv_sec * 1000000000 + tp.tv_nsec);
#elif HAVE_GETTIMEOFDAY
    struct timeval tv;
    gettimeofday (&tv, NULL);
    return (tv.tv_sec * 1000000 + tv.tv_usec)*1000;
#endif
}

timestamp_t get_timestamp(void) {
    static timestamp_t start = 0;
    if(start == 0) {
        start = get_time();
        return 0;
    }
    return get_time() - start;
}

void record_measure(const char *file, int line, const char *function, timestamp_t start, timestamp_t duration, int n_args, int *args) {
#ifdef SMPI_MEASURE
    static int my_rank = -1;
    if(my_rank < 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    }
    FILE *measure_file = get_measure_file();
    if(!measure_file) {fprintf(stderr, "error with measure_file\n"); exit(1);}
    fprintf(measure_file, "%s, %d, %s, %d, %e, %e", file, line, function, my_rank, start/PRECISION, duration/PRECISION);
    for(int i = 0; i < n_args; i++) {
        fprintf(measure_file, ", %d", args[i]);
    }
    fprintf(measure_file, "\n");
#endif
}

/*
 * This function supposes that the hosts are named with the convention:
 *      <prefix><radical><suffix>
 * For instance, dahu-8.grid5000.fr
 * It returns the value of the radical (8 in this example).
 */
int get_nodeid(void) {
    static int my_rank = -1;
    static int my_node = -1;
    if(my_rank < 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        char hostname[100];
        int max_size;
        MPI_Get_processor_name(hostname, &max_size);
        int i=0;
        while((hostname[i] < '0' || hostname[i] > '9') && i < max_size) {
            i++;
        }
        int j=i;
        while(hostname[j] >= '0' && hostname[j] <= '9' && j < max_size) {
            j++;
        }
        assert(j < max_size);
        hostname[j] = '\0';
        my_node = atoi(&hostname[i]);
    }
    return my_node;
}

/*
 * In Dahu@G5K, there are two CPUs per node.
 * This function computes the ID of the CPU, supposing that the ranks are split in even/odd.
 * See the output of lstopo to verify this hypothesis.
 * Note: this function does *not* suppose that all the nodes of Dahu are used or that the mapping
 * is done in order. It uses the hostname to get the right ID. If we made these assumptions,
 * returning 2*(rank/32) + rank%2 would be equivalent.
 */
int get_cpuid(void) {
    static int my_rank = -1;
    static int my_cpu = -1;
    static int my_node = -1;
    if(my_rank < 0) {
        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
        my_node = get_nodeid();
        // 2 CPUs per node, the ranks are split in even/odd, see the output of lstopo
        my_cpu = my_node*2 + my_rank%2;
//      printf("rank %d, node=%d, cpu=%d\n", my_rank, my_node, my_cpu);
    }
    return my_cpu;
}

double random_normal(void) {
    // From https://rosettacode.org/wiki/Statistics/Normal_distribution#C
    double x, y, rsq, f;
    do {
        x = 2.0 * rand() / (double)RAND_MAX - 1.0;
        y = 2.0 * rand() / (double)RAND_MAX - 1.0;
        rsq = x * x + y * y;
    }while( rsq >= 1. || rsq == 0. );
    f = sqrt( -2.0 * log(rsq) / rsq );
    return (x * f); // y*f would also be good
}

double random_halfnormal(void) {
    double x = random_normal();
    if(x < 0) {
        x = -x;
    }
    return x;
}

double random_halfnormal_shifted(double exp, double std) {
    // Here, exp and std are the desired expectation and standard deviation.
    // We compute the corresponding mu and sigma parameters for the normal distribution.
    double mu, sigma;
    sigma = std/sqrt(1-2/M_PI);
    mu = exp - sigma*sqrt(2/M_PI);
    double x = random_halfnormal();
    return x*sigma + mu;
}

void smpi_execute_normal(double mu, double sigma) {
    double coefficient = random_halfnormal_shifted(mu, sigma);
    if(coefficient > 0) {
        smpi_execute_benched(coefficient);
    }
}

void smpi_execute_normal_size(double mu, double sigma, double size) {
    double coefficient = random_halfnormal_shifted(mu, sigma);
    if(coefficient > 0 && size > 0) {
        smpi_execute_benched(size * coefficient);
    }
}

double dgemm_time(double M, double N, double K) {
    double mnk = M*N*K;
    double mn = M*N;
    double mk = M*K;
    double nk = N*K;
    double mu, sigma;
    switch(get_cpuid()) {
        case 2: // node 1
            mu    = 5.403580e-07 + 6.977255e-11*mnk;
            sigma = 6.666780e-07 + 1.230458e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 3: // node 1
            mu    = 5.220591e-07 + 6.659212e-11*mnk;
            sigma = 6.982747e-07 + 4.178078e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 4: // node 2
            mu    = 5.276913e-07 + 6.846163e-11*mnk;
            sigma = 6.044271e-07 + 1.192095e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 5: // node 2
            mu    = 5.192003e-07 + 6.698403e-11*mnk;
            sigma = 5.981250e-07 + 6.207988e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 6: // node 3
            mu    = 5.228191e-07 + 6.819123e-11*mnk;
            sigma = 5.833644e-07 + 1.109231e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 7: // node 3
            mu    = 5.071483e-07 + 6.640306e-11*mnk;
            sigma = 5.754146e-07 + 3.291517e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 8: // node 4
            mu    = 5.182449e-07 + 6.653141e-11*mnk;
            sigma = 5.959539e-07 + 4.337544e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 9: // node 4
            mu    = 5.176756e-07 + 6.650606e-11*mnk;
            sigma = 6.224987e-07 + 3.560624e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 10: // node 5
            mu    = 5.127542e-07 + 6.812662e-11*mnk;
            sigma = 5.664282e-07 + 1.143984e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 11: // node 5
            mu    = 5.362046e-07 + 6.619122e-11*mnk;
            sigma = 6.972006e-07 + 3.476378e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 12: // node 6
            mu    = 5.409690e-07 + 6.987692e-11*mnk;
            sigma = 6.312426e-07 + 1.136751e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 13: // node 6
            mu    = 5.032638e-07 + 6.609574e-11*mnk;
            sigma = 5.519080e-07 + 3.132958e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 14: // node 7
            mu    = 5.298281e-07 + 6.695839e-11*mnk;
            sigma = 6.521218e-07 + 5.393026e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 15: // node 7
            mu    = 4.937630e-07 + 6.585236e-11*mnk;
            sigma = 5.272074e-07 + 2.409341e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 16: // node 8
            mu    = 5.441216e-07 + 6.862712e-11*mnk;
            sigma = 6.673456e-07 + 1.363293e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 17: // node 8
            mu    = 5.000922e-07 + 6.659126e-11*mnk;
            sigma = 5.642580e-07 + 3.499616e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 18: // node 9
            mu    = 5.145131e-07 + 6.640818e-11*mnk;
            sigma = 5.899199e-07 + 3.755341e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 19: // node 9
            mu    = 5.049606e-07 + 6.651080e-11*mnk;
            sigma = 6.100888e-07 + 3.696837e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 20: // node 10
            mu    = 5.222622e-07 + 6.646985e-11*mnk;
            sigma = 6.697024e-07 + 3.154188e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 21: // node 10
            mu    = 5.240088e-07 + 6.652910e-11*mnk;
            sigma = 6.504793e-07 + 4.844305e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 22: // node 11
            mu    = 5.027986e-07 + 6.629976e-11*mnk;
            sigma = 5.948043e-07 + 2.949056e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 23: // node 11
            mu    = 5.418981e-07 + 6.843232e-11*mnk;
            sigma = 6.994459e-07 + 9.715857e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 24: // node 12
            mu    = 5.221636e-07 + 6.650839e-11*mnk;
            sigma = 6.667711e-07 + 3.785191e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 25: // node 12
            mu    = 5.256128e-07 + 6.618461e-11*mnk;
            sigma = 6.816091e-07 + 3.698915e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 34: // node 17
            mu    = 5.290902e-07 + 6.935939e-11*mnk;
            sigma = 6.040287e-07 + 1.113003e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 35: // node 17
            mu    = 5.222500e-07 + 6.651865e-11*mnk;
            sigma = 6.385465e-07 + 3.051929e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 36: // node 18
            mu    = 5.446243e-07 + 6.904369e-11*mnk;
            sigma = 7.242937e-07 + 9.096512e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 37: // node 18
            mu    = 4.973578e-07 + 6.639897e-11*mnk;
            sigma = 5.467784e-07 + 4.304195e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 38: // node 19
            mu    = 5.089191e-07 + 6.665264e-11*mnk;
            sigma = 5.694593e-07 + 4.120047e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 39: // node 19
            mu    = 5.322644e-07 + 6.671212e-11*mnk;
            sigma = 6.375594e-07 + 4.037225e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 40: // node 20
            mu    = 5.020687e-07 + 6.669615e-11*mnk;
            sigma = 5.396520e-07 + 3.786277e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 41: // node 20
            mu    = 5.361672e-07 + 6.669128e-11*mnk;
            sigma = 6.692976e-07 + 4.534718e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 42: // node 21
            mu    = 5.300218e-07 + 6.666058e-11*mnk;
            sigma = 6.927166e-07 + 5.976817e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 43: // node 21
            mu    = 5.249067e-07 + 6.649016e-11*mnk;
            sigma = 6.676958e-07 + 4.122023e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 44: // node 22
            mu    = 5.165410e-07 + 6.655833e-11*mnk;
            sigma = 5.997449e-07 + 4.313472e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 45: // node 22
            mu    = 5.150627e-07 + 6.707265e-11*mnk;
            sigma = 5.943844e-07 + 4.158497e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 46: // node 23
            mu    = 5.039400e-07 + 6.663073e-11*mnk;
            sigma = 5.998227e-07 + 2.945152e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 47: // node 23
            mu    = 5.168232e-07 + 6.628401e-11*mnk;
            sigma = 6.163759e-07 + 3.767985e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 48: // node 24
            mu    = 5.151544e-07 + 6.677963e-11*mnk;
            sigma = 5.993471e-07 + 3.876634e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 49: // node 24
            mu    = 5.078540e-07 + 6.626634e-11*mnk;
            sigma = 5.766558e-07 + 3.456476e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 50: // node 25
            mu    = 5.451601e-07 + 7.176179e-11*mnk;
            sigma = 6.490962e-07 + 9.213432e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 51: // node 25
            mu    = 4.997062e-07 + 6.613779e-11*mnk;
            sigma = 5.554807e-07 + 3.191015e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 52: // node 26
            mu    = 5.143945e-07 + 6.676052e-11*mnk;
            sigma = 5.757350e-07 + 3.737394e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 53: // node 26
            mu    = 5.274588e-07 + 6.765515e-11*mnk;
            sigma = 6.413720e-07 + 9.666468e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 54: // node 27
            mu    = 5.258334e-07 + 6.739887e-11*mnk;
            sigma = 6.297259e-07 + 5.794684e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 55: // node 27
            mu    = 5.332343e-07 + 6.789129e-11*mnk;
            sigma = 6.911383e-07 + 9.887092e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 56: // node 28
            mu    = 5.429231e-07 + 6.940035e-11*mnk;
            sigma = 6.893141e-07 + 1.059565e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 57: // node 28
            mu    = 5.134762e-07 + 6.633054e-11*mnk;
            sigma = 6.211102e-07 + 3.486190e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 58: // node 29
            mu    = 5.167773e-07 + 6.636452e-11*mnk;
            sigma = 6.069087e-07 + 3.766431e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 59: // node 29
            mu    = 5.163550e-07 + 6.644597e-11*mnk;
            sigma = 6.359725e-07 + 4.009133e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 60: // node 30
            mu    = 5.174845e-07 + 6.641888e-11*mnk;
            sigma = 6.391262e-07 + 5.524286e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 61: // node 30
            mu    = 5.115994e-07 + 6.632651e-11*mnk;
            sigma = 6.088435e-07 + 3.305459e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 62: // node 31
            mu    = 5.186783e-07 + 6.638550e-11*mnk;
            sigma = 5.830566e-07 + 4.824844e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 63: // node 31
            mu    = 5.122672e-07 + 6.649479e-11*mnk;
            sigma = 5.925127e-07 + 4.038332e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 64: // node 32
            mu    = 5.136935e-07 + 6.611778e-11*mnk;
            sigma = 5.761126e-07 + 4.115257e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 65: // node 32
            mu    = 5.062397e-07 + 6.628275e-11*mnk;
            sigma = 5.601277e-07 + 3.177925e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        default:
            mu    = 5.198703e-07 + 6.707309e-11*mnk;
            sigma = 6.195940e-07 + 5.568865e-13*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
    }
    return 0;
}

void smpi_execute_dgemm(int M, int N, int K) {
    double time = dgemm_time(M, N, K);
    if(time > 0) {
        smpi_execute_benched(time);
    }
}

#ifndef HPL_dgemm

#ifdef HPL_CALL_VSIPL

#ifdef STDC_HEADERS
static void HPL_dgemmNN
(
   const int                  M,
   const int                  N,
   const int                  K,
   const double               ALPHA,
   const double               * A,
   const int                  LDA,
   const double               * B,
   const int                  LDB,
   const double               BETA,
   double                     * C,
   const int                  LDC
)
#else
static void HPL_dgemmNN( M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
   const int                  K, LDA, LDB, LDC, M, N;
   const double               ALPHA, BETA;
   const double               * A, * B;
   double                     * C;
#endif
{
   register double            t0;
   int                        i, iail, iblj, icij, j, jal, jbj, jcj, l;

   for( j = 0, jbj = 0, jcj  = 0; j < N; j++, jbj += LDB, jcj += LDC )
   {
      HPL_dscal( M, BETA, C+jcj, 1 );
      for( l = 0, jal = 0, iblj = jbj; l < K; l++, jal += LDA, iblj += 1 )
      {
         t0 = ALPHA * B[iblj];
         for( i = 0, iail = jal, icij = jcj; i < M; i++, iail += 1, icij += 1 )
         { C[icij] += A[iail] * t0; }
      }
   }
}

#ifdef STDC_HEADERS
static void HPL_dgemmNT
(
   const int                  M,
   const int                  N,
   const int                  K,
   const double               ALPHA,
   const double               * A,
   const int                  LDA,
   const double               * B,
   const int                  LDB,
   const double               BETA,
   double                     * C,
   const int                  LDC
)
#else
static void HPL_dgemmNT( M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
   const int                  K, LDA, LDB, LDC, M, N;
   const double               ALPHA, BETA;
   const double               * A, * B;
   double                     * C;
#endif
{
   register double            t0;
   int                        i, iail, ibj, ibjl, icij, j, jal, jcj, l;

   for( j = 0, ibj  = 0, jcj  = 0; j < N; j++, ibj += 1, jcj += LDC )
   {
      HPL_dscal( M, BETA, C+jcj, 1 );
      for( l = 0, jal = 0, ibjl = ibj; l < K; l++, jal += LDA, ibjl += LDB )
      {
         t0 = ALPHA * B[ibjl];
         for( i = 0, iail = jal, icij = jcj; i < M; i++, iail += 1, icij += 1 )
         { C[icij] += A[iail] * t0; }
      }
   }
}

#ifdef STDC_HEADERS
static void HPL_dgemmTN
(
   const int                  M,
   const int                  N,
   const int                  K,
   const double               ALPHA,
   const double               * A,
   const int                  LDA,
   const double               * B,
   const int                  LDB,
   const double               BETA,
   double                     * C,
   const int                  LDC
)
#else
static void HPL_dgemmTN( M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
   const int                  K, LDA, LDB, LDC, M, N;
   const double               ALPHA, BETA;
   const double               * A, * B;
   double                     * C;
#endif
{
   register double            t0;
   int                        i, iai, iail, iblj, icij, j, jbj, jcj, l;

   for( j = 0, jbj = 0, jcj = 0; j < N; j++, jbj += LDB, jcj += LDC )
   {
      for( i = 0, icij = jcj, iai = 0; i < M; i++, icij += 1, iai += LDA )
      {
         t0 = HPL_rzero;
         for( l = 0, iail = iai, iblj = jbj; l < K; l++, iail += 1, iblj += 1 )
         { t0 += A[iail] * B[iblj]; }
         if( BETA == HPL_rzero ) C[icij]  = HPL_rzero;
         else                    C[icij] *= BETA;
         C[icij] += ALPHA * t0;
      }
   }
}

#ifdef STDC_HEADERS
static void HPL_dgemmTT
(
   const int                  M,
   const int                  N,
   const int                  K,
   const double               ALPHA,
   const double               * A,
   const int                  LDA,
   const double               * B,
   const int                  LDB,
   const double               BETA,
   double                     * C,
   const int                  LDC
)
#else
static void HPL_dgemmTT( M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
   const int                  K, LDA, LDB, LDC, M, N;
   const double               ALPHA, BETA;
   const double               * A, * B;
   double                     * C;
#endif
{
   register double            t0;
   int                        i, iali, ibj, ibjl, icij, j, jai, jcj, l;

   for( j = 0, ibj = 0, jcj  = 0; j < N; j++, ibj += 1, jcj += LDC )
   {
      for( i = 0, icij = jcj, jai = 0; i < M; i++, icij += 1, jai += LDA )
      {
         t0 = HPL_rzero;
         for( l = 0,      iali  = jai, ibjl  = ibj;
              l < K; l++, iali += 1,   ibjl += LDB ) t0 += A[iali] * B[ibjl];
         if( BETA == HPL_rzero ) C[icij]  = HPL_rzero;
         else                    C[icij] *= BETA;
         C[icij] += ALPHA * t0;
      }
   }
}

#ifdef STDC_HEADERS
static void HPL_dgemm0
(
   const enum HPL_TRANS       TRANSA,
   const enum HPL_TRANS       TRANSB,
   const int                  M,
   const int                  N,
   const int                  K,
   const double               ALPHA,
   const double               * A,
   const int                  LDA,
   const double               * B,
   const int                  LDB,
   const double               BETA,
   double                     * C,
   const int                  LDC
)
#else
static void HPL_dgemm0( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB,
                        BETA, C, LDC )
   const enum HPL_TRANS       TRANSA, TRANSB;
   const int                  K, LDA, LDB, LDC, M, N;
   const double               ALPHA, BETA;
   const double               * A, * B;
   double                     * C;
#endif
{
   int                        i, j;

   if( ( M == 0 ) || ( N == 0 ) ||
       ( ( ( ALPHA == HPL_rzero ) || ( K == 0 ) ) &&
         ( BETA == HPL_rone ) ) ) return;

   if( ALPHA == HPL_rzero )
   {
      for( j = 0; j < N; j++ )
      {  for( i = 0; i < M; i++ ) *(C+i+j*LDC) = HPL_rzero; }
      return;
   }

   if( TRANSB == HplNoTrans )
   {
      if( TRANSA == HplNoTrans )
      { HPL_dgemmNN( M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ); }
      else
      { HPL_dgemmTN( M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ); }
   }
   else
   {
      if( TRANSA == HplNoTrans )
      { HPL_dgemmNT( M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ); }
      else
      { HPL_dgemmTT( M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC ); }
   }
}

#endif

#ifdef STDC_HEADERS
void HPL_dgemm
(
   const enum HPL_ORDER             ORDER,
   const enum HPL_TRANS             TRANSA,
   const enum HPL_TRANS             TRANSB,
   const int                        M,
   const int                        N,
   const int                        K,
   const double                     ALPHA,
   const double *                   A,
   const int                        LDA,
   const double *                   B,
   const int                        LDB,
   const double                     BETA,
   double *                         C,
   const int                        LDC
)
#else
void HPL_dgemm
( ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC )
   const enum HPL_ORDER             ORDER;
   const enum HPL_TRANS             TRANSA;
   const enum HPL_TRANS             TRANSB;
   const int                        M;
   const int                        N;
   const int                        K;
   const double                     ALPHA;
   const double *                   A;
   const int                        LDA;
   const double *                   B;
   const int                        LDB;
   const double                     BETA;
   double *                         C;
   const int                        LDC;
#endif
{
/* 
 * Purpose
 * =======
 *
 * HPL_dgemm performs one of the matrix-matrix operations
 *  
 *     C := alpha * op( A ) * op( B ) + beta * C
 *  
 *  where op( X ) is one of
 *  
 *     op( X ) = X   or   op( X ) = X^T.
 *  
 * Alpha and beta are scalars,  and A,  B and C are matrices, with op(A)
 * an m by k matrix, op(B) a k by n matrix and  C an m by n matrix.
 *
 * Arguments
 * =========
 *
 * ORDER   (local input)                 const enum HPL_ORDER
 *         On entry, ORDER  specifies the storage format of the operands
 *         as follows:                                                  
 *            ORDER = HplRowMajor,                                      
 *            ORDER = HplColumnMajor.                                   
 *
 * TRANSA  (local input)                 const enum HPL_TRANS
 *         On entry, TRANSA  specifies the form of  op(A)  to be used in
 *         the matrix-matrix operation follows:                         
 *            TRANSA==HplNoTrans    : op( A ) = A,                     
 *            TRANSA==HplTrans      : op( A ) = A^T,                   
 *            TRANSA==HplConjTrans  : op( A ) = A^T.                   
 *
 * TRANSB  (local input)                 const enum HPL_TRANS
 *         On entry, TRANSB  specifies the form of  op(B)  to be used in
 *         the matrix-matrix operation follows:                         
 *            TRANSB==HplNoTrans    : op( B ) = B,                     
 *            TRANSB==HplTrans      : op( B ) = B^T,                   
 *            TRANSB==HplConjTrans  : op( B ) = B^T.                   
 *
 * M       (local input)                 const int
 *         On entry,  M  specifies  the  number  of rows  of the  matrix
 *         op(A)  and  of  the  matrix  C.  M  must  be  at least  zero.
 *
 * N       (local input)                 const int
 *         On entry,  N  specifies  the number  of columns of the matrix
 *         op(B)  and  the number of columns of the matrix  C. N must be
 *         at least zero.
 *
 * K       (local input)                 const int
 *         On entry,  K  specifies  the  number of columns of the matrix
 *         op(A) and the number of rows of the matrix op(B).  K  must be
 *         be at least  zero.
 *
 * ALPHA   (local input)                 const double
 *         On entry, ALPHA specifies the scalar alpha.   When  ALPHA  is
 *         supplied  as  zero  then the elements of the matrices A and B
 *         need not be set on input.
 *
 * A       (local input)                 const double *
 *         On entry,  A  is an array of dimension (LDA,ka),  where ka is
 *         k  when   TRANSA==HplNoTrans,  and  is  m  otherwise.  Before
 *         entry  with  TRANSA==HplNoTrans, the  leading  m by k part of
 *         the array  A must contain the matrix A, otherwise the leading
 *         k  by  m  part of the array  A  must  contain the  matrix  A.
 *
 * LDA     (local input)                 const int
 *         On entry, LDA  specifies the first dimension of A as declared
 *         in the  calling (sub) program. When  TRANSA==HplNoTrans  then
 *         LDA must be at least max(1,m), otherwise LDA must be at least
 *         max(1,k).
 *
 * B       (local input)                 const double *
 *         On entry, B is an array of dimension (LDB,kb),  where  kb  is
 *         n   when  TRANSB==HplNoTrans, and  is  k  otherwise.   Before
 *         entry with TRANSB==HplNoTrans,  the  leading  k by n  part of
 *         the array  B must contain the matrix B, otherwise the leading
 *         n  by  k  part of the array  B  must  contain  the matrix  B.
 *
 * LDB     (local input)                 const int
 *         On entry, LDB  specifies the first dimension of B as declared
 *         in the  calling (sub) program. When  TRANSB==HplNoTrans  then
 *         LDB must be at least max(1,k), otherwise LDB must be at least
 *         max(1,n).
 *
 * BETA    (local input)                 const double
 *         On entry,  BETA  specifies the scalar  beta.   When  BETA  is
 *         supplied  as  zero  then  the  elements of the matrix C  need
 *         not be set on input.
 *
 * C       (local input/output)          double *
 *         On entry,  C  is an array of dimension (LDC,n). Before entry,
 *         the  leading m by n part  of  the  array  C  must contain the
 *         matrix C,  except when beta is zero, in which case C need not
 *         be set on entry. On exit, the array  C  is overwritten by the
 *         m by n  matrix ( alpha*op( A )*op( B ) + beta*C ).
 *
 * LDC     (local input)                 const int
 *         On entry, LDC  specifies the first dimension of C as declared
 *         in  the   calling  (sub)  program.   LDC  must  be  at  least
 *         max(1,m).
 *
 * ---------------------------------------------------------------------
 */ 
#ifdef HPL_CALL_CBLAS
   cblas_dgemm( ORDER, TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB,
                BETA, C, LDC );
#endif
#ifdef HPL_CALL_VSIPL
   if( ORDER == HplColumnMajor )
   {
      HPL_dgemm0( TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA,
                  C, LDC );
   }
   else
   {
      HPL_dgemm0( TRANSB, TRANSA, N, M, K, ALPHA, B, LDB, A, LDA, BETA,
                  C, LDC );
   }
#endif
#ifdef HPL_CALL_FBLAS
   double                    alpha = ALPHA, beta = BETA;
#ifdef StringSunStyle
#ifdef HPL_USE_F77_INTEGER_DEF
   F77_INTEGER               IONE = 1;
#else
   int                       IONE = 1;
#endif
#endif
#ifdef StringStructVal
   F77_CHAR                  ftransa;
   F77_CHAR                  ftransb;
#endif
#ifdef StringStructPtr
   F77_CHAR                  ftransa;
   F77_CHAR                  ftransb;
#endif
#ifdef StringCrayStyle
   F77_CHAR                  ftransa;
   F77_CHAR                  ftransb;
#endif
#ifdef HPL_USE_F77_INTEGER_DEF
   const F77_INTEGER         F77M   = M,   F77N   = N,   F77K = K,
                             F77lda = LDA, F77ldb = LDB, F77ldc = LDC;
#else
#define F77M                 M
#define F77N                 N
#define F77K                 K
#define F77lda               LDA
#define F77ldb               LDB
#define F77ldc               LDC
#endif
   char                      ctransa, ctransb;

   if(      TRANSA == HplNoTrans ) ctransa = 'N';
   else if( TRANSA == HplTrans   ) ctransa = 'T';
   else                            ctransa = 'C';
 
   if(      TRANSB == HplNoTrans ) ctransb = 'N';
   else if( TRANSB == HplTrans   ) ctransb = 'T';
   else                            ctransb = 'C';

   if( ORDER == HplColumnMajor )
   {
#ifdef StringSunStyle
      F77dgemm( &ctransa, &ctransb, &F77M, &F77N, &F77K, &alpha, A, &F77lda,
                B, &F77ldb, &beta, C, &F77ldc, IONE, IONE );
#endif
#ifdef StringCrayStyle
      ftransa = HPL_C2F_CHAR( ctransa ); ftransb = HPL_C2F_CHAR( ctransb );
      F77dgemm( ftransa,  ftransb,  &F77M, &F77N, &F77K, &alpha, A, &F77lda,
                B, &F77ldb, &beta, C, &F77ldc );
#endif
#ifdef StringStructVal
      ftransa.len = 1; ftransa.cp = &ctransa;
      ftransb.len = 1; ftransb.cp = &ctransb;
      F77dgemm( ftransa,  ftransb,  &F77M, &F77N, &F77K, &alpha, A, &F77lda,
                B, &F77ldb, &beta, C, &F77ldc );
#endif
#ifdef StringStructPtr
      ftransa.len = 1; ftransa.cp = &ctransa;
      ftransb.len = 1; ftransb.cp = &ctransb;
      F77dgemm( &ftransa, &ftransb, &F77M, &F77N, &F77K, &alpha, A, &F77lda,
                B, &F77ldb, &beta, C, &F77ldc );
#endif
   }
   else
   {
#ifdef StringSunStyle
      F77dgemm( &ctransb, &ctransa, &F77N, &F77M, &F77K, &alpha, B, &F77ldb,
                A, &F77lda, &beta, C, &F77ldc, IONE, IONE );
#endif
#ifdef StringCrayStyle
      ftransa = HPL_C2F_CHAR( ctransa ); ftransb = HPL_C2F_CHAR( ctransb );
      F77dgemm( ftransb,  ftransa,  &F77N, &F77M, &F77K, &alpha, B, &F77ldb,
                A, &F77lda, &beta, C, &F77ldc );
#endif
#ifdef StringStructVal
      ftransa.len = 1; ftransa.cp = &ctransa;
      ftransb.len = 1; ftransb.cp = &ctransb;
      F77dgemm( ftransb,  ftransa,  &F77N, &F77M, &F77K, &alpha, B, &F77ldb,
                A, &F77lda, &beta, C, &F77ldc );
#endif
#ifdef StringStructPtr
      ftransa.len = 1; ftransa.cp = &ctransa;
      ftransb.len = 1; ftransb.cp = &ctransb;
      F77dgemm( &ftransb, &ftransa, &F77N, &F77M, &F77K, &alpha, B, &F77ldb,
                A, &F77lda, &beta, C, &F77ldc );
#endif
   }
#endif
/*
 * End of HPL_dgemm
 */
}

#endif
