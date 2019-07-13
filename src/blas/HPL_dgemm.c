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
            mu    = 2.031875e-06 + 7.089503e-11*mnk;
            sigma = 3.568280e-07 + 2.146765e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 3: // node 1
            mu    = 1.823625e-06 + 6.665738e-11*mnk;
            sigma = 2.867134e-07 + 2.109864e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 4: // node 2
            mu    = 1.793563e-06 + 6.943151e-11*mnk;
            sigma = 2.294533e-07 + 2.192932e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 5: // node 2
            mu    = 2.039063e-06 + 6.656724e-11*mnk;
            sigma = 4.459159e-07 + 2.114460e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 6: // node 3
            mu    = 1.959375e-06 + 6.840299e-11*mnk;
            sigma = 2.402006e-07 + 2.192171e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 7: // node 3
            mu    = 1.847375e-06 + 6.642321e-11*mnk;
            sigma = 3.594620e-07 + 2.059016e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 8: // node 4
            mu    = 1.998187e-06 + 6.655238e-11*mnk;
            sigma = 3.049520e-07 + 2.071881e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 9: // node 4
            mu    = 1.817000e-06 + 6.660695e-11*mnk;
            sigma = 3.134492e-07 + 2.065467e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 10: // node 5
            mu    = 1.820687e-06 + 6.884917e-11*mnk;
            sigma = 2.678372e-07 + 2.163212e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 11: // node 5
            mu    = 1.947375e-06 + 6.618913e-11*mnk;
            sigma = 3.068313e-07 + 2.005688e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 12: // node 6
            mu    = 1.713125e-06 + 6.882313e-11*mnk;
            sigma = 3.035692e-07 + 2.159327e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 13: // node 6
            mu    = 1.699313e-06 + 6.595870e-11*mnk;
            sigma = 3.698865e-07 + 2.014535e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 14: // node 7
            mu    = 1.886312e-06 + 6.654638e-11*mnk;
            sigma = 2.518608e-07 + 2.090538e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 15: // node 7
            mu    = 1.796812e-06 + 6.604634e-11*mnk;
            sigma = 3.626939e-07 + 1.993063e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 16: // node 8
            mu    = 1.804438e-06 + 6.750281e-11*mnk;
            sigma = 2.663016e-07 + 2.111583e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 17: // node 8
            mu    = 1.838500e-06 + 6.629657e-11*mnk;
            sigma = 2.998226e-07 + 2.046241e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 18: // node 9
            mu    = 1.917312e-06 + 6.645334e-11*mnk;
            sigma = 3.490054e-07 + 2.026941e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 19: // node 9
            mu    = 1.847000e-06 + 6.630077e-11*mnk;
            sigma = 3.180082e-07 + 2.022967e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 20: // node 10
            mu    = 1.761937e-06 + 6.662196e-11*mnk;
            sigma = 2.238800e-07 + 2.087194e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 21: // node 10
            mu    = 1.922875e-06 + 6.661536e-11*mnk;
            sigma = 3.366890e-07 + 2.079113e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 22: // node 11
            mu    = 1.861937e-06 + 6.622357e-11*mnk;
            sigma = 4.185254e-07 + 2.031334e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 23: // node 11
            mu    = 2.156563e-06 + 6.751946e-11*mnk;
            sigma = 4.030012e-07 + 2.158426e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 24: // node 12
            mu    = 2.022250e-06 + 6.649450e-11*mnk;
            sigma = 2.922302e-07 + 2.064654e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 25: // node 12
            mu    = 1.895062e-06 + 6.622475e-11*mnk;
            sigma = 3.338113e-07 + 2.043099e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 26: // node 13
            mu    = 1.946938e-06 + 6.712666e-11*mnk;
            sigma = 2.636078e-07 + 2.126319e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 27: // node 13
            mu    = 1.915750e-06 + 6.640446e-11*mnk;
            sigma = 4.160324e-07 + 2.035143e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 28: // node 14
            mu    = 1.888000e-06 + 6.721964e-11*mnk;
            sigma = 3.060832e-07 + 2.146180e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 29: // node 14
            mu    = 1.876250e-06 + 6.640571e-11*mnk;
            sigma = 4.447404e-07 + 2.061288e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 30: // node 15
            mu    = 2.027313e-06 + 7.105527e-11*mnk;
            sigma = 3.439558e-07 + 2.162025e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 31: // node 15
            mu    = 1.897313e-06 + 6.629775e-11*mnk;
            sigma = 3.715223e-07 + 2.034065e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 32: // node 16
            mu    = 1.805000e-06 + 6.843961e-11*mnk;
            sigma = 1.983260e-07 + 2.140443e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 33: // node 16
            mu    = 2.061687e-06 + 6.650259e-11*mnk;
            sigma = 3.943524e-07 + 2.053045e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 34: // node 17
            mu    = 1.927125e-06 + 7.088907e-11*mnk;
            sigma = 2.357691e-07 + 2.149820e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 35: // node 17
            mu    = 1.840500e-06 + 6.662498e-11*mnk;
            sigma = 2.561754e-07 + 2.104414e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 36: // node 18
            mu    = 2.027813e-06 + 7.018252e-11*mnk;
            sigma = 3.002064e-07 + 2.134772e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 37: // node 18
            mu    = 1.824375e-06 + 6.630521e-11*mnk;
            sigma = 3.967755e-07 + 2.012848e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 38: // node 19
            mu    = 1.970375e-06 + 6.682882e-11*mnk;
            sigma = 2.644972e-07 + 2.120927e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 39: // node 19
            mu    = 2.141750e-06 + 6.639155e-11*mnk;
            sigma = 5.203626e-07 + 2.052522e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 40: // node 20
            mu    = 1.865688e-06 + 6.725642e-11*mnk;
            sigma = 2.772743e-07 + 2.117133e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 41: // node 20
            mu    = 1.844062e-06 + 6.635935e-11*mnk;
            sigma = 3.830493e-07 + 2.053453e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 42: // node 21
            mu    = 1.899687e-06 + 6.811646e-11*mnk;
            sigma = 2.948335e-07 + 2.168911e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 43: // node 21
            mu    = 1.940625e-06 + 6.656142e-11*mnk;
            sigma = 1.735386e-07 + 2.077141e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 44: // node 22
            mu    = 1.919000e-06 + 6.667892e-11*mnk;
            sigma = 2.798954e-07 + 2.096335e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 45: // node 22
            mu    = 1.875250e-06 + 6.687615e-11*mnk;
            sigma = 2.939988e-07 + 2.073643e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 46: // node 23
            mu    = 1.888875e-06 + 6.666754e-11*mnk;
            sigma = 3.385195e-07 + 2.090285e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 47: // node 23
            mu    = 1.901437e-06 + 6.605016e-11*mnk;
            sigma = 3.148678e-07 + 2.004616e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 48: // node 24
            mu    = 1.881375e-06 + 6.668186e-11*mnk;
            sigma = 2.714981e-07 + 2.086284e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 49: // node 24
            mu    = 2.094625e-06 + 6.627112e-11*mnk;
            sigma = 6.198059e-07 + 2.027162e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 50: // node 25
            mu    = 2.032188e-06 + 7.135967e-11*mnk;
            sigma = 3.254480e-07 + 2.154060e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 51: // node 25
            mu    = 1.833188e-06 + 6.599957e-11*mnk;
            sigma = 3.375893e-07 + 2.021207e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 52: // node 26
            mu    = 1.880250e-06 + 6.647706e-11*mnk;
            sigma = 2.736377e-07 + 2.051089e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 53: // node 26
            mu    = 1.970313e-06 + 6.691876e-11*mnk;
            sigma = 4.829437e-07 + 2.113398e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 54: // node 27
            mu    = 1.980687e-06 + 6.674318e-11*mnk;
            sigma = 2.983720e-07 + 2.099196e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 55: // node 27
            mu    = 1.974188e-06 + 6.686199e-11*mnk;
            sigma = 2.836010e-07 + 2.106967e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 56: // node 28
            mu    = 1.977875e-06 + 6.869802e-11*mnk;
            sigma = 2.558424e-07 + 2.147067e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 57: // node 28
            mu    = 1.857938e-06 + 6.630950e-11*mnk;
            sigma = 3.404247e-07 + 2.033643e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 58: // node 29
            mu    = 1.896375e-06 + 6.652144e-11*mnk;
            sigma = 2.431139e-07 + 2.075896e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 59: // node 29
            mu    = 1.762000e-06 + 6.649413e-11*mnk;
            sigma = 2.861419e-07 + 2.044431e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 60: // node 30
            mu    = 1.794250e-06 + 6.628271e-11*mnk;
            sigma = 3.752261e-07 + 2.032734e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 61: // node 30
            mu    = 1.848187e-06 + 6.613762e-11*mnk;
            sigma = 4.537737e-07 + 2.029401e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 62: // node 31
            mu    = 1.764000e-06 + 6.657323e-11*mnk;
            sigma = 2.465163e-07 + 2.050375e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 63: // node 31
            mu    = 1.943188e-06 + 6.653508e-11*mnk;
            sigma = 3.981035e-07 + 2.078539e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 64: // node 32
            mu    = 1.809313e-06 + 6.626337e-11*mnk;
            sigma = 3.340864e-07 + 2.032057e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 65: // node 32
            mu    = 1.880875e-06 + 6.621148e-11*mnk;
            sigma = 3.672168e-07 + 2.008040e-12*mnk;
            return mu + random_halfnormal_shifted(0, sigma);
        default:
            mu    = 1.901050e-06 + 6.710598e-11*mnk;
            sigma = 3.266040e-07 + 2.082146e-12*mnk;
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
