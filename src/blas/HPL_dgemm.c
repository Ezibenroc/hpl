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
            mu    = 1.920706e-06 + 2.725859e-09*mk + 6.135157e-10*mn + 5.670235e-11*mnk + 4.420071e-09*nk;
            sigma = 2.307468e-07 + 3.819801e-11*mk + 1.952596e-11*mn + 2.932509e-14*mnk + 3.407333e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 3: // node 1
            mu    = 1.887630e-06 + 2.705612e-09*mk + 6.058066e-10*mn + 5.670967e-11*mnk + 4.394546e-09*nk;
            sigma = 2.279908e-07 + 6.287517e-11*mk + 2.758403e-11*mn + 3.292636e-14*mnk + 7.955969e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 4: // node 2
            mu    = 1.950269e-06 + 2.706446e-09*mk + 6.070946e-10*mn + 5.669854e-11*mnk + 4.392050e-09*nk;
            sigma = 2.414690e-07 + 6.984441e-11*mk + 2.846902e-11*mn + 3.357951e-14*mnk + 8.578821e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 5: // node 2
            mu    = 1.869491e-06 + 2.698235e-09*mk + 6.052865e-10*mn + 5.668332e-11*mnk + 4.378333e-09*nk;
            sigma = 2.222736e-07 + 9.254787e-11*mk + 4.264450e-11*mn + 4.267720e-14*mnk + 1.321826e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 6: // node 3
            mu    = 1.948711e-06 + 2.727051e-09*mk + 6.153261e-10*mn + 5.669153e-11*mnk + 4.421226e-09*nk;
            sigma = 2.317713e-07 + 2.753851e-11*mk + 1.619755e-11*mn + 2.065736e-14*mnk + 1.965521e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 7: // node 3
            mu    = 1.855512e-06 + 2.712284e-09*mk + 6.089147e-10*mn + 5.670728e-11*mnk + 4.403437e-09*nk;
            sigma = 2.012789e-07 + 4.844793e-11*mk + 2.468433e-11*mn + 2.801854e-14*mnk + 5.364372e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 8: // node 4
            mu    = 1.934426e-06 + 2.715745e-09*mk + 6.128094e-10*mn + 5.670998e-11*mnk + 4.407961e-09*nk;
            sigma = 2.242818e-07 + 5.427509e-11*mk + 2.431847e-11*mn + 2.865003e-14*mnk + 5.928664e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 9: // node 4
            mu    = 1.885227e-06 + 2.714240e-09*mk + 6.114392e-10*mn + 5.669571e-11*mnk + 4.408306e-09*nk;
            sigma = 2.324742e-07 + 4.777760e-11*mk + 2.651458e-11*mn + 2.750099e-14*mnk + 5.256685e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 10: // node 5
            mu    = 1.932120e-06 + 2.718976e-09*mk + 6.164016e-10*mn + 5.670671e-11*mnk + 4.412873e-09*nk;
            sigma = 2.374500e-07 + 4.638788e-11*mk + 2.138970e-11*mn + 3.261918e-14*mnk + 5.023944e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 11: // node 5
            mu    = 1.895676e-06 + 2.715419e-09*mk + 6.081014e-10*mn + 5.671280e-11*mnk + 4.406977e-09*nk;
            sigma = 2.442500e-07 + 4.706232e-11*mk + 2.197807e-11*mn + 2.952676e-14*mnk + 5.205191e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 12: // node 6
            mu    = 1.862565e-06 + 2.715319e-09*mk + 6.075288e-10*mn + 5.669460e-11*mnk + 4.408910e-09*nk;
            sigma = 2.291848e-07 + 4.717645e-11*mk + 1.857311e-11*mn + 3.289645e-14*mnk + 5.638879e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 13: // node 6
            mu    = 1.913481e-06 + 2.705180e-09*mk + 6.030422e-10*mn + 5.670778e-11*mnk + 4.398764e-09*nk;
            sigma = 2.278779e-07 + 5.759927e-11*mk + 2.457283e-11*mn + 3.954850e-14*mnk + 7.239593e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 14: // node 7
            mu    = 1.902472e-06 + 2.706026e-09*mk + 6.097172e-10*mn + 5.668807e-11*mnk + 4.390366e-09*nk;
            sigma = 2.403094e-07 + 8.000297e-11*mk + 3.628487e-11*mn + 3.985617e-14*mnk + 1.059955e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 15: // node 7
            mu    = 1.885164e-06 + 2.701479e-09*mk + 6.036106e-10*mn + 5.670548e-11*mnk + 4.385622e-09*nk;
            sigma = 2.243023e-07 + 7.991319e-11*mk + 3.541987e-11*mn + 4.459534e-14*mnk + 1.062360e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 16: // node 8
            mu    = 1.871708e-06 + 2.707646e-09*mk + 6.068004e-10*mn + 5.672451e-11*mnk + 4.392900e-09*nk;
            sigma = 2.329226e-07 + 6.614114e-11*mk + 2.321142e-11*mn + 3.382509e-14*mnk + 8.643419e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 17: // node 8
            mu    = 1.927280e-06 + 2.720178e-09*mk + 6.119683e-10*mn + 5.671810e-11*mnk + 4.416377e-09*nk;
            sigma = 2.394787e-07 + 3.800090e-11*mk + 1.989680e-11*mn + 2.294893e-14*mnk + 3.688772e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 18: // node 9
            mu    = 1.876484e-06 + 2.714920e-09*mk + 6.113282e-10*mn + 5.670705e-11*mnk + 4.403605e-09*nk;
            sigma = 2.297661e-07 + 6.167638e-11*mk + 2.583711e-11*mn + 2.470829e-14*mnk + 7.749450e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 19: // node 9
            mu    = 1.886650e-06 + 2.712276e-09*mk + 6.143693e-10*mn + 5.669198e-11*mnk + 4.402077e-09*nk;
            sigma = 2.250603e-07 + 6.082408e-11*mk + 2.930420e-11*mn + 2.322688e-14*mnk + 7.598976e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 20: // node 10
            mu    = 1.924947e-06 + 2.723635e-09*mk + 6.176081e-10*mn + 5.670487e-11*mnk + 4.421637e-09*nk;
            sigma = 2.328078e-07 + 3.505035e-11*mk + 1.875443e-11*mn + 2.968472e-14*mnk + 3.396515e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 21: // node 10
            mu    = 1.897461e-06 + 2.704222e-09*mk + 6.053472e-10*mn + 5.671046e-11*mnk + 4.396249e-09*nk;
            sigma = 2.392551e-07 + 7.104717e-11*mk + 3.024342e-11*mn + 3.562971e-14*mnk + 9.009846e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 22: // node 11
            mu    = 1.895319e-06 + 2.719426e-09*mk + 6.152623e-10*mn + 5.669265e-11*mnk + 4.415522e-09*nk;
            sigma = 2.260287e-07 + 4.493692e-11*mk + 2.330992e-11*mn + 3.378032e-14*mnk + 5.020119e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 23: // node 11
            mu    = 1.865354e-06 + 2.697095e-09*mk + 6.041174e-10*mn + 5.669703e-11*mnk + 4.382192e-09*nk;
            sigma = 2.215312e-07 + 9.170886e-11*mk + 3.729793e-11*mn + 3.851437e-14*mnk + 1.292127e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 24: // node 12
            mu    = 1.907576e-06 + 2.719012e-09*mk + 6.120470e-10*mn + 5.669864e-11*mnk + 4.412049e-09*nk;
            sigma = 2.445912e-07 + 4.735191e-11*mk + 2.306978e-11*mn + 2.685373e-14*mnk + 5.366969e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 25: // node 12
            mu    = 1.883954e-06 + 2.706609e-09*mk + 6.078908e-10*mn + 5.669386e-11*mnk + 4.393487e-09*nk;
            sigma = 2.464341e-07 + 7.212046e-11*mk + 3.217446e-11*mn + 2.640438e-14*mnk + 9.497405e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 26: // node 13
            mu    = 1.898725e-06 + 2.724537e-09*mk + 6.149651e-10*mn + 5.669619e-11*mnk + 4.417265e-09*nk;
            sigma = 2.525347e-07 + 3.893779e-11*mk + 2.036006e-11*mn + 2.183424e-14*mnk + 3.593404e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 27: // node 13
            mu    = 1.927664e-06 + 2.719273e-09*mk + 6.119252e-10*mn + 5.671673e-11*mnk + 4.412973e-09*nk;
            sigma = 2.336985e-07 + 4.005228e-11*mk + 2.173311e-11*mn + 1.644343e-14*mnk + 4.142709e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 28: // node 14
            mu    = 1.939646e-06 + 2.720850e-09*mk + 6.130993e-10*mn + 5.669571e-11*mnk + 4.416221e-09*nk;
            sigma = 2.416626e-07 + 4.540853e-11*mk + 2.247863e-11*mn + 3.424767e-14*mnk + 4.989955e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 29: // node 14
            mu    = 1.893868e-06 + 2.723296e-09*mk + 6.151683e-10*mn + 5.669865e-11*mnk + 4.415781e-09*nk;
            sigma = 2.352773e-07 + 3.788123e-11*mk + 1.951612e-11*mn + 2.341041e-14*mnk + 3.630120e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 30: // node 15
            mu    = 1.878576e-06 + 2.722005e-09*mk + 6.138975e-10*mn + 5.669555e-11*mnk + 4.414045e-09*nk;
            sigma = 2.392584e-07 + 4.477390e-11*mk + 2.485462e-11*mn + 3.378944e-14*mnk + 4.946738e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 31: // node 15
            mu    = 1.918641e-06 + 2.707463e-09*mk + 6.063423e-10*mn + 5.672054e-11*mnk + 4.399950e-09*nk;
            sigma = 2.413315e-07 + 5.620384e-11*mk + 2.735850e-11*mn + 1.628087e-14*mnk + 6.805498e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 32: // node 16
            mu    = 1.881806e-06 + 2.723875e-09*mk + 6.178082e-10*mn + 5.671116e-11*mnk + 4.420583e-09*nk;
            sigma = 2.374260e-07 + 3.643894e-11*mk + 1.935298e-11*mn + 2.864269e-14*mnk + 3.552435e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 33: // node 16
            mu    = 1.894875e-06 + 2.708836e-09*mk + 6.087423e-10*mn + 5.669359e-11*mnk + 4.396711e-09*nk;
            sigma = 2.328593e-07 + 6.921612e-11*mk + 3.191381e-11*mn + 4.105096e-14*mnk + 8.652871e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 34: // node 17
            mu    = 1.897231e-06 + 2.731949e-09*mk + 6.182256e-10*mn + 5.669974e-11*mnk + 4.424907e-09*nk;
            sigma = 2.151565e-07 + 2.690814e-11*mk + 1.639295e-11*mn + 2.136001e-14*mnk + 1.703962e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 35: // node 17
            mu    = 1.882014e-06 + 2.705613e-09*mk + 6.044872e-10*mn + 5.670653e-11*mnk + 4.392877e-09*nk;
            sigma = 2.196792e-07 + 7.438905e-11*mk + 3.062409e-11*mn + 2.476248e-14*mnk + 9.703095e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 36: // node 18
            mu    = 1.870854e-06 + 2.727205e-09*mk + 6.163998e-10*mn + 5.669247e-11*mnk + 4.422025e-09*nk;
            sigma = 2.288106e-07 + 3.709047e-11*mk + 1.844526e-11*mn + 2.493683e-14*mnk + 3.694847e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 37: // node 18
            mu    = 1.937782e-06 + 2.716580e-09*mk + 6.124036e-10*mn + 5.670583e-11*mnk + 4.408750e-09*nk;
            sigma = 2.435029e-07 + 5.075269e-11*mk + 2.219622e-11*mn + 2.246202e-14*mnk + 5.689918e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 38: // node 19
            mu    = 1.958897e-06 + 2.718765e-09*mk + 6.095301e-10*mn + 5.682796e-11*mnk + 4.396082e-09*nk;
            sigma = 2.338462e-07 + 3.781502e-11*mk + 2.056522e-11*mn + 3.698859e-14*mnk + 3.646371e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 39: // node 19
            mu    = 1.903088e-06 + 2.707518e-09*mk + 5.999742e-10*mn + 5.676448e-11*mnk + 4.385741e-09*nk;
            sigma = 2.197380e-07 + 6.020179e-11*mk + 2.853776e-11*mn + 3.383798e-14*mnk + 7.632710e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 40: // node 20
            mu    = 1.905081e-06 + 2.703730e-09*mk + 6.049936e-10*mn + 5.669376e-11*mnk + 4.385142e-09*nk;
            sigma = 2.405159e-07 + 8.945173e-11*mk + 3.903439e-11*mn + 4.518541e-14*mnk + 1.241346e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 41: // node 20
            mu    = 1.900877e-06 + 2.721216e-09*mk + 6.150117e-10*mn + 5.670207e-11*mnk + 4.414276e-09*nk;
            sigma = 2.362510e-07 + 3.927819e-11*mk + 2.246089e-11*mn + 2.068631e-14*mnk + 3.695693e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 42: // node 21
            mu    = 2.036896e-06 + 2.708216e-09*mk + 6.126964e-10*mn + 5.709169e-11*mnk + 4.386165e-09*nk;
            sigma = 2.382824e-07 + 6.550238e-11*mk + 2.862876e-11*mn + 7.278683e-14*mnk + 8.310319e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 43: // node 21
            mu    = 1.874995e-06 + 2.697369e-09*mk + 6.006641e-10*mn + 5.676125e-11*mnk + 4.383113e-09*nk;
            sigma = 2.481845e-07 + 8.891891e-11*mk + 3.640857e-11*mn + 3.731432e-14*mnk + 1.227079e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 44: // node 22
            mu    = 1.924771e-06 + 2.731042e-09*mk + 6.178840e-10*mn + 5.670739e-11*mnk + 4.425973e-09*nk;
            sigma = 2.443419e-07 + 2.730660e-11*mk + 1.446454e-11*mn + 2.178005e-14*mnk + 1.810482e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 45: // node 22
            mu    = 1.919322e-06 + 2.722726e-09*mk + 6.144909e-10*mn + 5.670418e-11*mnk + 4.416999e-09*nk;
            sigma = 2.322999e-07 + 2.732499e-11*mk + 1.670010e-11*mn + 1.969793e-14*mnk + 1.794393e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 46: // node 23
            mu    = 1.911187e-06 + 2.711560e-09*mk + 6.131393e-10*mn + 5.668691e-11*mnk + 4.401086e-09*nk;
            sigma = 2.360715e-07 + 5.937826e-11*mk + 2.689745e-11*mn + 2.893081e-14*mnk + 7.327680e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 47: // node 23
            mu    = 1.865384e-06 + 2.724238e-09*mk + 6.152294e-10*mn + 5.671321e-11*mnk + 4.424378e-09*nk;
            sigma = 2.220855e-07 + 2.754952e-11*mk + 1.706113e-11*mn + 2.022079e-14*mnk + 1.876936e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 48: // node 24
            mu    = 1.904234e-06 + 2.711768e-09*mk + 6.101231e-10*mn + 5.669669e-11*mnk + 4.398196e-09*nk;
            sigma = 2.510465e-07 + 6.574323e-11*mk + 3.314970e-11*mn + 5.088368e-14*mnk + 8.987711e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 49: // node 24
            mu    = 1.885525e-06 + 2.714160e-09*mk + 6.092265e-10*mn + 5.670450e-11*mnk + 4.411087e-09*nk;
            sigma = 2.285639e-07 + 5.318812e-11*mk + 2.482290e-11*mn + 2.896387e-14*mnk + 5.869807e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 50: // node 25
            mu    = 1.909718e-06 + 2.716573e-09*mk + 6.072186e-10*mn + 5.670564e-11*mnk + 4.392506e-09*nk;
            sigma = 2.466478e-07 + 5.811632e-11*mk + 2.911246e-11*mn + 4.447138e-14*mnk + 7.472041e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 51: // node 25
            mu    = 1.892946e-06 + 2.698435e-09*mk + 5.973497e-10*mn + 5.673092e-11*mnk + 4.368839e-09*nk;
            sigma = 2.371654e-07 + 8.157661e-11*mk + 3.225617e-11*mn + 3.015695e-14*mnk + 1.115735e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 52: // node 26
            mu    = 1.850664e-06 + 2.724825e-09*mk + 6.143635e-10*mn + 5.668903e-11*mnk + 4.418161e-09*nk;
            sigma = 2.267901e-07 + 3.527565e-11*mk + 2.014481e-11*mn + 2.844389e-14*mnk + 3.321849e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 53: // node 26
            mu    = 1.945968e-06 + 2.718538e-09*mk + 6.118655e-10*mn + 5.672092e-11*mnk + 4.408394e-09*nk;
            sigma = 2.415751e-07 + 4.998107e-11*mk + 2.460588e-11*mn + 2.371669e-14*mnk + 5.526109e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 54: // node 27
            mu    = 1.939458e-06 + 2.716514e-09*mk + 6.133014e-10*mn + 5.670191e-11*mnk + 4.406144e-09*nk;
            sigma = 2.246002e-07 + 5.825189e-11*mk + 3.014528e-11*mn + 3.255645e-14*mnk + 7.194435e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 55: // node 27
            mu    = 1.861833e-06 + 2.709786e-09*mk + 6.077501e-10*mn + 5.667645e-11*mnk + 4.399580e-09*nk;
            sigma = 2.429368e-07 + 5.782015e-11*mk + 2.717669e-11*mn + 4.176077e-14*mnk + 6.959823e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 56: // node 28
            mu    = 1.939167e-06 + 2.729897e-09*mk + 6.178734e-10*mn + 5.671991e-11*mnk + 4.424076e-09*nk;
            sigma = 2.540062e-07 + 2.668533e-11*mk + 1.528206e-11*mn + 2.456749e-14*mnk + 1.720506e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 57: // node 28
            mu    = 1.897465e-06 + 2.721007e-09*mk + 6.148260e-10*mn + 5.670854e-11*mnk + 4.414193e-09*nk;
            sigma = 2.299895e-07 + 3.865860e-11*mk + 2.162831e-11*mn + 2.259478e-14*mnk + 3.690942e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 58: // node 29
            mu    = 1.946632e-06 + 2.704774e-09*mk + 6.094217e-10*mn + 5.670019e-11*mnk + 4.393668e-09*nk;
            sigma = 2.469101e-07 + 7.331337e-11*mk + 3.293681e-11*mn + 3.580201e-14*mnk + 9.824060e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 59: // node 29
            mu    = 1.897905e-06 + 2.711404e-09*mk + 6.098139e-10*mn + 5.670481e-11*mnk + 4.403201e-09*nk;
            sigma = 2.372141e-07 + 5.759077e-11*mk + 2.883223e-11*mn + 3.506004e-14*mnk + 7.017518e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 60: // node 30
            mu    = 1.858229e-06 + 2.700690e-09*mk + 6.076182e-10*mn + 5.668788e-11*mnk + 4.388052e-09*nk;
            sigma = 2.202803e-07 + 8.098992e-11*mk + 3.332682e-11*mn + 5.005898e-14*mnk + 1.087010e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 61: // node 30
            mu    = 1.920544e-06 + 2.691553e-09*mk + 6.023993e-10*mn + 5.670158e-11*mnk + 4.371786e-09*nk;
            sigma = 2.243067e-07 + 9.440233e-11*mk + 3.982972e-11*mn + 5.610920e-14*mnk + 1.301376e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 62: // node 31
            mu    = 1.938060e-06 + 2.717827e-09*mk + 6.118964e-10*mn + 5.669892e-11*mnk + 4.405276e-09*nk;
            sigma = 2.526594e-07 + 6.072126e-11*mk + 2.914607e-11*mn + 2.995915e-14*mnk + 7.374222e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 63: // node 31
            mu    = 1.906109e-06 + 2.720967e-09*mk + 6.119801e-10*mn + 5.670432e-11*mnk + 4.420093e-09*nk;
            sigma = 2.393165e-07 + 3.268973e-11*mk + 1.653357e-11*mn + 2.599903e-14*mnk + 2.204525e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 64: // node 32
            mu    = 2.010109e-06 + 2.717128e-09*mk + 6.178669e-10*mn + 5.694224e-11*mnk + 4.399556e-09*nk;
            sigma = 2.634176e-07 + 3.692376e-11*mk + 2.144768e-11*mn + 4.355908e-14*mnk + 3.723030e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 65: // node 32
            mu    = 1.916970e-06 + 2.717555e-09*mk + 6.108009e-10*mn + 5.676000e-11*mnk + 4.413285e-09*nk;
            sigma = 2.275490e-07 + 4.610031e-11*mk + 2.452559e-11*mn + 3.449074e-14*mnk + 5.078184e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 66: // node 33
            mu    = 1.909947e-06 + 2.720488e-09*mk + 6.153097e-10*mn + 5.670123e-11*mnk + 4.411440e-09*nk;
            sigma = 2.327736e-07 + 4.198604e-11*mk + 1.913370e-11*mn + 2.729109e-14*mnk + 4.279724e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 67: // node 33
            mu    = 1.914009e-06 + 2.711521e-09*mk + 6.113062e-10*mn + 5.669747e-11*mnk + 4.403921e-09*nk;
            sigma = 2.577761e-07 + 5.901531e-11*mk + 2.614888e-11*mn + 2.808144e-14*mnk + 7.154107e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 68: // node 34
            mu    = 1.958162e-06 + 2.703713e-09*mk + 6.090492e-10*mn + 5.670194e-11*mnk + 4.387397e-09*nk;
            sigma = 2.544535e-07 + 9.052012e-11*mk + 4.146121e-11*mn + 4.102742e-14*mnk + 1.268659e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 69: // node 34
            mu    = 1.886104e-06 + 2.713880e-09*mk + 6.134058e-10*mn + 5.669885e-11*mnk + 4.402579e-09*nk;
            sigma = 2.325427e-07 + 6.150689e-11*mk + 2.812709e-11*mn + 3.428103e-14*mnk + 7.802808e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 70: // node 35
            mu    = 1.916610e-06 + 2.688959e-09*mk + 6.074085e-10*mn + 5.670561e-11*mnk + 4.367637e-09*nk;
            sigma = 2.244576e-07 + 9.374051e-11*mk + 4.093957e-11*mn + 4.411181e-14*mnk + 1.329797e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 71: // node 35
            mu    = 1.899417e-06 + 2.703349e-09*mk + 6.128400e-10*mn + 5.669694e-11*mnk + 4.393911e-09*nk;
            sigma = 2.281443e-07 + 7.274220e-11*mk + 3.015513e-11*mn + 3.314320e-14*mnk + 9.433706e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 72: // node 36
            mu    = 1.935539e-06 + 2.723390e-09*mk + 6.122637e-10*mn + 5.669844e-11*mnk + 4.416154e-09*nk;
            sigma = 2.651846e-07 + 4.092522e-11*mk + 1.939605e-11*mn + 1.840959e-14*mnk + 4.071634e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 73: // node 36
            mu    = 1.838729e-06 + 2.713514e-09*mk + 6.090739e-10*mn + 5.670258e-11*mnk + 4.405291e-09*nk;
            sigma = 2.080884e-07 + 4.269608e-11*mk + 1.980651e-11*mn + 2.497354e-14*mnk + 4.485522e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 74: // node 37
            mu    = 1.878898e-06 + 2.725232e-09*mk + 6.170351e-10*mn + 5.668991e-11*mnk + 4.418323e-09*nk;
            sigma = 2.347488e-07 + 3.740339e-11*mk + 1.843363e-11*mn + 2.219675e-14*mnk + 3.841937e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 75: // node 37
            mu    = 1.941660e-06 + 2.710036e-09*mk + 6.112637e-10*mn + 5.668935e-11*mnk + 4.396529e-09*nk;
            sigma = 2.451271e-07 + 7.105412e-11*mk + 3.266245e-11*mn + 3.534216e-14*mnk + 8.946008e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 76: // node 38
            mu    = 1.909206e-06 + 2.701758e-09*mk + 6.084279e-10*mn + 5.668934e-11*mnk + 4.383983e-09*nk;
            sigma = 2.222435e-07 + 8.371111e-11*mk + 3.755269e-11*mn + 3.955329e-14*mnk + 1.134456e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 77: // node 38
            mu    = 1.932873e-06 + 2.712864e-09*mk + 6.100259e-10*mn + 5.670251e-11*mnk + 4.401993e-09*nk;
            sigma = 2.472921e-07 + 5.895454e-11*mk + 2.416910e-11*mn + 3.198506e-14*mnk + 7.245126e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 78: // node 39
            mu    = 1.881190e-06 + 2.726202e-09*mk + 6.166486e-10*mn + 5.669073e-11*mnk + 4.420027e-09*nk;
            sigma = 2.280898e-07 + 3.756449e-11*mk + 1.992922e-11*mn + 2.978040e-14*mnk + 3.538007e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 79: // node 39
            mu    = 1.929032e-06 + 2.705691e-09*mk + 6.080815e-10*mn + 5.672167e-11*mnk + 4.390337e-09*nk;
            sigma = 2.316736e-07 + 7.064503e-11*mk + 2.966645e-11*mn + 3.874913e-14*mnk + 9.151989e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 80: // node 40
            mu    = 1.957352e-06 + 2.720212e-09*mk + 6.124939e-10*mn + 5.670488e-11*mnk + 4.412234e-09*nk;
            sigma = 2.471365e-07 + 3.931848e-11*mk + 2.095184e-11*mn + 3.006186e-14*mnk + 4.201551e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 81: // node 40
            mu    = 1.850058e-06 + 2.716517e-09*mk + 6.100103e-10*mn + 5.669520e-11*mnk + 4.409207e-09*nk;
            sigma = 2.176599e-07 + 4.808481e-11*mk + 2.436412e-11*mn + 2.950799e-14*mnk + 5.180545e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 82: // node 41
            mu    = 1.940076e-06 + 2.734403e-09*mk + 6.149549e-10*mn + 5.669110e-11*mnk + 4.408932e-09*nk;
            sigma = 2.546291e-07 + 3.733910e-11*mk + 2.045416e-11*mn + 2.912733e-14*mnk + 3.547441e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 83: // node 41
            mu    = 1.910089e-06 + 2.719440e-09*mk + 6.079553e-10*mn + 5.670900e-11*mnk + 4.396100e-09*nk;
            sigma = 2.289839e-07 + 6.141056e-11*mk + 2.867150e-11*mn + 3.655837e-14*mnk + 7.798562e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 84: // node 42
            mu    = 1.930757e-06 + 2.728899e-09*mk + 6.179919e-10*mn + 5.667520e-11*mnk + 4.414111e-09*nk;
            sigma = 2.491303e-07 + 5.247392e-11*mk + 2.397246e-11*mn + 2.040085e-14*mnk + 6.305615e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 85: // node 42
            mu    = 1.961012e-06 + 2.721076e-09*mk + 6.169960e-10*mn + 5.667348e-11*mnk + 4.407490e-09*nk;
            sigma = 2.431261e-07 + 5.208138e-11*mk + 2.659938e-11*mn + 1.998535e-14*mnk + 6.061687e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 86: // node 43
            mu    = 1.884819e-06 + 2.727720e-09*mk + 6.198700e-10*mn + 5.669140e-11*mnk + 4.427013e-09*nk;
            sigma = 2.277172e-07 + 2.883373e-11*mk + 1.711964e-11*mn + 2.512395e-14*mnk + 2.017483e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 87: // node 43
            mu    = 1.936588e-06 + 2.701675e-09*mk + 6.070144e-10*mn + 5.671760e-11*mnk + 4.381342e-09*nk;
            sigma = 2.483451e-07 + 8.783852e-11*mk + 3.627385e-11*mn + 3.633400e-14*mnk + 1.159768e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 88: // node 44
            mu    = 1.944813e-06 + 2.720489e-09*mk + 6.157698e-10*mn + 5.669742e-11*mnk + 4.412804e-09*nk;
            sigma = 2.487983e-07 + 4.849856e-11*mk + 2.412076e-11*mn + 3.141909e-14*mnk + 5.207961e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 89: // node 44
            mu    = 1.835093e-06 + 2.711027e-09*mk + 6.084052e-10*mn + 5.670828e-11*mnk + 4.398082e-09*nk;
            sigma = 2.182157e-07 + 5.977851e-11*mk + 2.769890e-11*mn + 2.895959e-14*mnk + 7.464531e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 90: // node 45
            mu    = 1.872194e-06 + 2.724320e-09*mk + 6.140230e-10*mn + 5.669662e-11*mnk + 4.420542e-09*nk;
            sigma = 2.186232e-07 + 3.619977e-11*mk + 1.922438e-11*mn + 2.911143e-14*mnk + 3.536974e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 91: // node 45
            mu    = 1.928484e-06 + 2.702886e-09*mk + 6.032524e-10*mn + 5.670813e-11*mnk + 4.389887e-09*nk;
            sigma = 2.429586e-07 + 8.238648e-11*mk + 3.536980e-11*mn + 3.657427e-14*mnk + 1.107123e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 92: // node 46
            mu    = 1.877032e-06 + 2.721893e-09*mk + 6.131291e-10*mn + 5.670149e-11*mnk + 4.413182e-09*nk;
            sigma = 2.558242e-07 + 3.784282e-11*mk + 1.762168e-11*mn + 2.650360e-14*mnk + 3.641821e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 93: // node 46
            mu    = 1.913736e-06 + 2.691475e-09*mk + 6.002893e-10*mn + 5.670450e-11*mnk + 4.367146e-09*nk;
            sigma = 2.434046e-07 + 1.050942e-10*mk + 4.560105e-11*mn + 4.388403e-14*mnk + 1.532165e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 94: // node 47
            mu    = 1.941785e-06 + 2.722041e-09*mk + 6.153748e-10*mn + 5.670602e-11*mnk + 4.412736e-09*nk;
            sigma = 2.426432e-07 + 4.624921e-11*mk + 2.445853e-11*mn + 3.446390e-14*mnk + 4.865539e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 95: // node 47
            mu    = 1.879947e-06 + 2.700997e-09*mk + 6.052866e-10*mn + 5.670455e-11*mnk + 4.385106e-09*nk;
            sigma = 2.282054e-07 + 8.318427e-11*mk + 3.669758e-11*mn + 3.505591e-14*mnk + 1.097604e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 96: // node 48
            mu    = 1.897067e-06 + 2.725046e-09*mk + 6.192626e-10*mn + 5.669587e-11*mnk + 4.416329e-09*nk;
            sigma = 2.405780e-07 + 3.603298e-11*mk + 1.881126e-11*mn + 3.161687e-14*mnk + 3.101562e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 97: // node 48
            mu    = 1.943583e-06 + 2.717373e-09*mk + 6.099342e-10*mn + 5.670931e-11*mnk + 4.408828e-09*nk;
            sigma = 2.532946e-07 + 4.661381e-11*mk + 2.376601e-11*mn + 3.123715e-14*mnk + 5.243805e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 98: // node 49
            mu    = 1.926565e-06 + 2.726646e-09*mk + 6.155567e-10*mn + 5.669550e-11*mnk + 4.420002e-09*nk;
            sigma = 2.399266e-07 + 3.525871e-11*mk + 1.886121e-11*mn + 3.056969e-14*mnk + 3.455762e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 99: // node 49
            mu    = 1.904456e-06 + 2.712009e-09*mk + 6.094891e-10*mn + 5.670357e-11*mnk + 4.401626e-09*nk;
            sigma = 2.341259e-07 + 4.851798e-11*mk + 2.287357e-11*mn + 2.683522e-14*mnk + 5.478514e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 100: // node 50
            mu    = 2.041608e-06 + 2.714230e-09*mk + 6.266637e-10*mn + 5.687153e-11*mnk + 4.399335e-09*nk;
            sigma = 2.628429e-07 + 3.818141e-11*mk + 2.288787e-11*mn + 4.258976e-14*mnk + 3.726142e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 101: // node 50
            mu    = 1.832684e-06 + 2.696517e-09*mk + 6.102356e-10*mn + 5.673744e-11*mnk + 4.385759e-09*nk;
            sigma = 2.193310e-07 + 6.841029e-11*mk + 3.354199e-11*mn + 5.383326e-14*mnk + 8.827420e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 102: // node 51
            mu    = 1.888792e-06 + 2.717827e-09*mk + 6.111255e-10*mn + 5.669108e-11*mnk + 4.410529e-09*nk;
            sigma = 2.451056e-07 + 4.709239e-11*mk + 2.091167e-11*mn + 3.358293e-14*mnk + 5.226122e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 103: // node 51
            mu    = 1.931521e-06 + 2.699625e-09*mk + 6.048411e-10*mn + 5.669054e-11*mnk + 4.377660e-09*nk;
            sigma = 2.241790e-07 + 9.934759e-11*mk + 4.376964e-11*mn + 4.817029e-14*mnk + 1.433068e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 104: // node 52
            mu    = 1.870278e-06 + 2.719347e-09*mk + 6.120703e-10*mn + 5.670289e-11*mnk + 4.412310e-09*nk;
            sigma = 2.308383e-07 + 4.917291e-11*mk + 2.227219e-11*mn + 2.977320e-14*mnk + 5.546712e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 105: // node 52
            mu    = 1.941183e-06 + 2.703038e-09*mk + 6.038220e-10*mn + 5.671226e-11*mnk + 4.394646e-09*nk;
            sigma = 2.392716e-07 + 7.168909e-11*mk + 2.702850e-11*mn + 3.431878e-14*mnk + 9.501115e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 106: // node 53
            mu    = 1.946965e-06 + 2.706049e-09*mk + 6.067010e-10*mn + 5.670011e-11*mnk + 4.387330e-09*nk;
            sigma = 2.596267e-07 + 8.895484e-11*mk + 3.533295e-11*mn + 4.036235e-14*mnk + 1.292962e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 107: // node 53
            mu    = 1.893400e-06 + 2.702388e-09*mk + 6.037705e-10*mn + 5.669872e-11*mnk + 4.391780e-09*nk;
            sigma = 2.449333e-07 + 7.800110e-11*mk + 3.572260e-11*mn + 4.828034e-14*mnk + 1.035908e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 108: // node 54
            mu    = 1.994507e-06 + 2.724781e-09*mk + 6.261027e-10*mn + 5.690597e-11*mnk + 4.406548e-09*nk;
            sigma = 2.513929e-07 + 2.632399e-11*mk + 1.559929e-11*mn + 4.266861e-14*mnk + 1.619744e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 109: // node 54
            mu    = 1.931488e-06 + 2.713668e-09*mk + 6.112101e-10*mn + 5.676226e-11*mnk + 4.402745e-09*nk;
            sigma = 2.313605e-07 + 5.896344e-11*mk + 2.816476e-11*mn + 3.341294e-14*mnk + 7.165340e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 110: // node 55
            mu    = 1.919757e-06 + 2.726633e-09*mk + 6.173929e-10*mn + 5.670347e-11*mnk + 4.419192e-09*nk;
            sigma = 2.384193e-07 + 3.944974e-11*mk + 2.048901e-11*mn + 2.088533e-14*mnk + 3.902069e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 111: // node 55
            mu    = 1.894125e-06 + 2.702719e-09*mk + 6.044747e-10*mn + 5.670861e-11*mnk + 4.387428e-09*nk;
            sigma = 2.217238e-07 + 7.476768e-11*mk + 3.331092e-11*mn + 3.737109e-14*mnk + 1.005317e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 112: // node 56
            mu    = 1.912484e-06 + 2.710427e-09*mk + 6.111896e-10*mn + 5.670005e-11*mnk + 4.393307e-09*nk;
            sigma = 2.273095e-07 + 8.203514e-11*mk + 3.239435e-11*mn + 4.003547e-14*mnk + 1.064243e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 113: // node 56
            mu    = 1.923456e-06 + 2.692534e-09*mk + 6.028907e-10*mn + 5.670703e-11*mnk + 4.377302e-09*nk;
            sigma = 2.391456e-07 + 9.431419e-11*mk + 3.708635e-11*mn + 4.412309e-14*mnk + 1.332023e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 114: // node 57
            mu    = 1.905933e-06 + 2.722709e-09*mk + 6.151534e-10*mn + 5.670239e-11*mnk + 4.413910e-09*nk;
            sigma = 2.564043e-07 + 4.722577e-11*mk + 2.147937e-11*mn + 3.249631e-14*mnk + 5.267661e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 115: // node 57
            mu    = 1.877292e-06 + 2.706037e-09*mk + 6.051430e-10*mn + 5.668867e-11*mnk + 4.389724e-09*nk;
            sigma = 2.204650e-07 + 7.869631e-11*mk + 3.799421e-11*mn + 3.922813e-14*mnk + 1.086141e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 116: // node 58
            mu    = 1.913646e-06 + 2.736646e-09*mk + 6.174583e-10*mn + 5.667555e-11*mnk + 4.415259e-09*nk;
            sigma = 2.363474e-07 + 5.668449e-11*mk + 2.462538e-11*mn + 2.183978e-14*mnk + 6.424781e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 117: // node 58
            mu    = 1.947440e-06 + 2.738775e-09*mk + 6.183055e-10*mn + 5.666891e-11*mnk + 4.418465e-09*nk;
            sigma = 2.317654e-07 + 4.478798e-11*mk + 2.433869e-11*mn + 2.416902e-14*mnk + 4.367734e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 118: // node 59
            mu    = 1.886975e-06 + 2.727283e-09*mk + 6.149294e-10*mn + 5.670267e-11*mnk + 4.420544e-09*nk;
            sigma = 2.142487e-07 + 3.746945e-11*mk + 1.937631e-11*mn + 2.821163e-14*mnk + 3.372333e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 119: // node 59
            mu    = 1.919757e-06 + 2.716352e-09*mk + 6.119505e-10*mn + 5.670693e-11*mnk + 4.409044e-09*nk;
            sigma = 2.383390e-07 + 4.918510e-11*mk + 2.552691e-11*mn + 2.420923e-14*mnk + 5.594589e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 120: // node 60
            mu    = 1.934613e-06 + 2.697053e-09*mk + 6.066026e-10*mn + 5.673217e-11*mnk + 4.399064e-09*nk;
            sigma = 2.583064e-07 + 5.128431e-11*mk + 2.407079e-11*mn + 2.551764e-14*mnk + 5.691146e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 121: // node 60
            mu    = 1.844809e-06 + 2.671066e-09*mk + 5.935149e-10*mn + 5.673428e-11*mnk + 4.366688e-09*nk;
            sigma = 2.191625e-07 + 7.951782e-11*mk + 3.269684e-11*mn + 3.924966e-14*mnk + 1.091659e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 122: // node 61
            mu    = 1.913109e-06 + 2.717291e-09*mk + 6.132714e-10*mn + 5.669266e-11*mnk + 4.404847e-09*nk;
            sigma = 2.633726e-07 + 5.926221e-11*mk + 2.895731e-11*mn + 3.147868e-14*mnk + 7.140834e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 123: // node 61
            mu    = 1.926231e-06 + 2.706669e-09*mk + 6.045874e-10*mn + 5.671430e-11*mnk + 4.394204e-09*nk;
            sigma = 2.431750e-07 + 7.127436e-11*mk + 3.284493e-11*mn + 4.183023e-14*mnk + 9.249361e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 124: // node 62
            mu    = 1.939755e-06 + 2.706025e-09*mk + 6.136227e-10*mn + 5.668926e-11*mnk + 4.390118e-09*nk;
            sigma = 2.418386e-07 + 7.245486e-11*mk + 3.471327e-11*mn + 3.833742e-14*mnk + 9.812457e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 125: // node 62
            mu    = 1.870873e-06 + 2.705453e-09*mk + 6.082530e-10*mn + 5.670042e-11*mnk + 4.388179e-09*nk;
            sigma = 2.213333e-07 + 6.764084e-11*mk + 2.891295e-11*mn + 2.967090e-14*mnk + 8.224516e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 126: // node 63
            mu    = 1.963514e-06 + 2.727578e-09*mk + 6.171396e-10*mn + 5.671507e-11*mnk + 4.425163e-09*nk;
            sigma = 2.296497e-07 + 2.763853e-11*mk + 1.710843e-11*mn + 1.910027e-14*mnk + 2.026500e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 127: // node 63
            mu    = 1.839815e-06 + 2.706244e-09*mk + 6.084514e-10*mn + 5.669871e-11*mnk + 4.392244e-09*nk;
            sigma = 2.091283e-07 + 7.105481e-11*mk + 3.144709e-11*mn + 3.017026e-14*mnk + 9.313877e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 128: // node 64
            mu    = 1.933208e-06 + 2.735679e-09*mk + 6.253504e-10*mn + 5.667169e-11*mnk + 4.423579e-09*nk;
            sigma = 2.429682e-07 + 3.819928e-11*mk + 2.134754e-11*mn + 2.555529e-14*mnk + 3.752306e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 129: // node 64
            mu    = 1.920051e-06 + 2.724117e-09*mk + 6.167576e-10*mn + 5.667263e-11*mnk + 4.406170e-09*nk;
            sigma = 2.223868e-07 + 6.075459e-11*mk + 3.001284e-11*mn + 3.219550e-14*mnk + 7.549895e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 130: // node 65
            mu    = 2.034924e-06 + 2.715373e-09*mk + 6.224000e-10*mn + 5.698277e-11*mnk + 4.394186e-09*nk;
            sigma = 2.733254e-07 + 4.056611e-11*mk + 1.861610e-11*mn + 4.912166e-14*mnk + 4.163102e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 131: // node 65
            mu    = 1.918567e-06 + 2.710319e-09*mk + 6.086194e-10*mn + 5.676183e-11*mnk + 4.401457e-09*nk;
            sigma = 2.348022e-07 + 6.986353e-11*mk + 3.273843e-11*mn + 3.566796e-14*mnk + 9.090988e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 132: // node 66
            mu    = 2.035338e-06 + 2.716366e-09*mk + 6.172112e-10*mn + 5.709857e-11*mnk + 4.397954e-09*nk;
            sigma = 2.721835e-07 + 4.607797e-11*mk + 2.207722e-11*mn + 6.444564e-14*mnk + 5.088881e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 133: // node 66
            mu    = 1.876792e-06 + 2.721728e-09*mk + 6.160836e-10*mn + 5.675530e-11*mnk + 4.416476e-09*nk;
            sigma = 2.333508e-07 + 6.068540e-11*mk + 3.061081e-11*mn + 2.691855e-14*mnk + 7.827369e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 134: // node 67
            mu    = 1.885021e-06 + 2.719016e-09*mk + 6.113547e-10*mn + 5.670866e-11*mnk + 4.408223e-09*nk;
            sigma = 2.501635e-07 + 5.712499e-11*mk + 2.713992e-11*mn + 3.919184e-14*mnk + 6.905416e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 135: // node 67
            mu    = 1.917058e-06 + 2.711553e-09*mk + 6.072857e-10*mn + 5.671110e-11*mnk + 4.403217e-09*nk;
            sigma = 2.286335e-07 + 5.362419e-11*mk + 2.709774e-11*mn + 4.253450e-14*mnk + 6.065990e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 136: // node 68
            mu    = 1.872664e-06 + 2.720288e-09*mk + 6.152273e-10*mn + 5.668499e-11*mnk + 4.412718e-09*nk;
            sigma = 2.287999e-07 + 4.681244e-11*mk + 2.524295e-11*mn + 3.454303e-14*mnk + 4.909286e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 137: // node 68
            mu    = 1.969208e-06 + 2.722755e-09*mk + 6.143413e-10*mn + 5.670636e-11*mnk + 4.415397e-09*nk;
            sigma = 2.341090e-07 + 3.928092e-11*mk + 1.981116e-11*mn + 2.139433e-14*mnk + 3.730112e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 138: // node 69
            mu    = 1.920850e-06 + 2.725967e-09*mk + 6.192393e-10*mn + 5.669505e-11*mnk + 4.421386e-09*nk;
            sigma = 2.337872e-07 + 3.567782e-11*mk + 2.105466e-11*mn + 2.968849e-14*mnk + 3.267684e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 139: // node 69
            mu    = 1.849315e-06 + 2.713805e-09*mk + 6.102476e-10*mn + 5.669712e-11*mnk + 4.404072e-09*nk;
            sigma = 2.194782e-07 + 4.912665e-11*mk + 2.249971e-11*mn + 2.568063e-14*mnk + 5.684984e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 140: // node 70
            mu    = 1.903229e-06 + 2.729853e-09*mk + 6.192252e-10*mn + 5.669355e-11*mnk + 4.423721e-09*nk;
            sigma = 2.414613e-07 + 2.754123e-11*mk + 1.577198e-11*mn + 2.391188e-14*mnk + 1.733762e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 141: // node 70
            mu    = 1.943685e-06 + 2.719815e-09*mk + 6.116576e-10*mn + 5.672119e-11*mnk + 4.415193e-09*nk;
            sigma = 2.443572e-07 + 3.890410e-11*mk + 2.112270e-11*mn + 2.468903e-14*mnk + 3.747977e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 142: // node 71
            mu    = 1.890713e-06 + 2.704608e-09*mk + 6.070470e-10*mn + 5.669199e-11*mnk + 4.388175e-09*nk;
            sigma = 2.419681e-07 + 7.488721e-11*mk + 3.007742e-11*mn + 3.595389e-14*mnk + 9.846805e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 143: // node 71
            mu    = 1.890391e-06 + 2.714255e-09*mk + 6.074044e-10*mn + 5.670840e-11*mnk + 4.398280e-09*nk;
            sigma = 2.202015e-07 + 5.989363e-11*mk + 2.940208e-11*mn + 3.972950e-14*mnk + 7.607670e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 144: // node 72
            mu    = 1.944840e-06 + 2.715352e-09*mk + 6.122965e-10*mn + 5.669293e-11*mnk + 4.404998e-09*nk;
            sigma = 2.233898e-07 + 6.231457e-11*mk + 2.798290e-11*mn + 2.491685e-14*mnk + 7.698071e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 145: // node 72
            mu    = 1.868002e-06 + 2.706061e-09*mk + 6.076617e-10*mn + 5.670136e-11*mnk + 4.391002e-09*nk;
            sigma = 2.070661e-07 + 7.123741e-11*mk + 3.448813e-11*mn + 4.133863e-14*mnk + 9.312375e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        default:
            mu    = 1.910907e-06 + 2.714325e-09*mk + 6.113027e-10*mn + 5.671624e-11*mnk + 4.403225e-09*nk;
            sigma = 2.360326e-07 + 5.608520e-11*mk + 2.633984e-11*mn + 3.262784e-14*mnk + 6.725982e-11*nk;
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
