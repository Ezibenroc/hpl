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
            mu    = 5.393244e-07 + 7.160371e-11*mnk + -5.740287e-10*mn + 1.361118e-09*mk + 2.524071e-09*nk;
            sigma = 6.433163e-07 + 1.824113e-12*mnk + 1.009451e-10*mn + 1.347495e-10*mk + 9.457558e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 3: // node 1
            mu    = 5.228051e-07 + 6.812406e-11*mnk + -1.334303e-10*mn + 1.660894e-09*mk + 2.965415e-09*nk;
            sigma = 6.225043e-07 + 5.113089e-13*mnk + 1.976500e-11*mn + 5.622314e-11*mk + 5.870116e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 4: // node 2
            mu    = 5.329919e-07 + 7.018230e-11*mnk + -4.171307e-10*mn + 1.428695e-09*mk + 2.710319e-09*nk;
            sigma = 5.876532e-07 + 1.871904e-12*mnk + 1.152215e-11*mn + 4.975200e-11*mk + 1.887183e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 5: // node 2
            mu    = 5.262917e-07 + 6.852644e-11*mnk + -1.616727e-10*mn + 1.650467e-09*mk + 3.011181e-09*nk;
            sigma = 6.142762e-07 + 8.672489e-13*mnk + 1.792774e-11*mn + 5.265701e-11*mk + 6.082794e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 6: // node 3
            mu    = 5.305164e-07 + 6.987903e-11*mnk + -3.740277e-10*mn + 1.458767e-09*mk + 2.779049e-09*nk;
            sigma = 5.923702e-07 + 1.731721e-12*mnk + 1.189282e-11*mn + 4.741327e-11*mk + 2.357065e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 7: // node 3
            mu    = 5.136505e-07 + 6.791413e-11*mnk + -9.740071e-11*mn + 1.678005e-09*mk + 3.002629e-09*nk;
            sigma = 5.874046e-07 + 3.605968e-13*mnk + 3.339540e-11*mn + 5.059355e-11*mk + 5.858528e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 8: // node 4
            mu    = 5.230183e-07 + 6.805912e-11*mnk + -1.267585e-10*mn + 1.641870e-09*mk + 2.988689e-09*nk;
            sigma = 5.810166e-07 + 5.526344e-13*mnk + 1.344342e-11*mn + 4.505368e-11*mk + 5.126023e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 9: // node 4
            mu    = 5.268878e-07 + 6.801869e-11*mnk + -9.948909e-11*mn + 1.677894e-09*mk + 3.011612e-09*nk;
            sigma = 6.407215e-07 + 4.079723e-13*mnk + 1.953734e-11*mn + 4.503107e-11*mk + 6.347004e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 10: // node 5
            mu    = 5.190615e-07 + 6.977845e-11*mnk + -3.173703e-10*mn + 1.548433e-09*mk + 2.821357e-09*nk;
            sigma = 5.844281e-07 + 1.778073e-12*mnk + 2.361859e-11*mn + 7.380883e-11*mk + 2.018451e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 11: // node 5
            mu    = 5.322948e-07 + 6.771328e-11*mnk + -8.666260e-11*mn + 1.632169e-09*mk + 2.940922e-09*nk;
            sigma = 6.321226e-07 + 3.918068e-13*mnk + 2.645732e-11*mn + 5.564334e-11*mk + 6.106570e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 12: // node 6
            mu    = 5.491498e-07 + 7.161547e-11*mnk + -3.753763e-10*mn + 1.526969e-09*mk + 2.762434e-09*nk;
            sigma = 6.274122e-07 + 1.653940e-12*mnk + 1.094427e-10*mn + 1.206953e-10*mk + 1.335707e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 13: // node 6
            mu    = 5.087136e-07 + 6.761309e-11*mnk + -8.990279e-11*mn + 1.624337e-09*mk + 2.954883e-09*nk;
            sigma = 5.362213e-07 + 3.353323e-13*mnk + 2.516824e-11*mn + 5.463920e-11*mk + 5.663861e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 14: // node 7
            mu    = 5.385376e-07 + 6.848558e-11*mnk + -1.085675e-10*mn + 1.675846e-09*mk + 3.031570e-09*nk;
            sigma = 6.471845e-07 + 7.280079e-13*mnk + 2.056132e-11*mn + 5.775322e-11*mk + 5.753924e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 15: // node 7
            mu    = 5.023913e-07 + 6.735477e-11*mnk + -7.262560e-11*mn + 1.620455e-09*mk + 2.983781e-09*nk;
            sigma = 5.343881e-07 + 2.188962e-13*mnk + 1.798127e-11*mn + 4.872682e-11*mk + 5.032753e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 16: // node 8
            mu    = 5.492241e-07 + 7.031480e-11*mnk + -3.770104e-10*mn + 1.502750e-09*mk + 2.812904e-09*nk;
            sigma = 6.733427e-07 + 2.115356e-12*mnk + 4.653663e-11*mn + 8.707011e-11*mk + 7.255564e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 17: // node 8
            mu    = 5.076557e-07 + 6.814376e-11*mnk + -1.366242e-10*mn + 1.606765e-09*mk + 2.910988e-09*nk;
            sigma = 5.655804e-07 + 3.881762e-13*mnk + 3.147357e-11*mn + 6.516342e-11*mk + 6.524439e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 18: // node 9
            mu    = 5.231685e-07 + 6.794667e-11*mnk + -1.199330e-10*mn + 1.610505e-09*mk + 2.934597e-09*nk;
            sigma = 5.964673e-07 + 4.459490e-13*mnk + 1.937225e-11*mn + 5.018231e-11*mk + 5.910065e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 19: // node 9
            mu    = 5.153587e-07 + 6.805513e-11*mnk + -1.204156e-10*mn + 1.616533e-09*mk + 2.919605e-09*nk;
            sigma = 5.959051e-07 + 4.494021e-13*mnk + 1.580925e-11*mn + 4.237050e-11*mk + 4.294049e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 20: // node 10
            mu    = 5.261699e-07 + 6.796471e-11*mnk + -7.340727e-11*mn + 1.700672e-09*mk + 3.057443e-09*nk;
            sigma = 6.469882e-07 + 3.283416e-13*mnk + 2.882459e-11*mn + 5.589685e-11*mk + 6.332054e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 21: // node 10
            mu    = 5.288127e-07 + 6.805150e-11*mnk + -1.028025e-10*mn + 1.654537e-09*mk + 2.993711e-09*nk;
            sigma = 6.169246e-07 + 6.500448e-13*mnk + 9.911284e-12*mn + 3.664809e-11*mk + 4.506676e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 22: // node 11
            mu    = 5.092075e-07 + 6.781534e-11*mnk + -1.004866e-10*mn + 1.643766e-09*mk + 2.990651e-09*nk;
            sigma = 5.726342e-07 + 2.967345e-13*mnk + 2.996774e-11*mn + 5.202909e-11*mk + 6.606632e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 23: // node 11
            mu    = 5.414760e-07 + 7.014003e-11*mnk + -4.190672e-10*mn + 1.443803e-09*mk + 2.748622e-09*nk;
            sigma = 6.561465e-07 + 1.454804e-12*mnk + 3.240033e-11*mn + 8.273599e-11*mk + 4.048271e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 24: // node 12
            mu    = 5.233135e-07 + 6.804495e-11*mnk + -1.191088e-10*mn + 1.609964e-09*mk + 2.962635e-09*nk;
            sigma = 6.099668e-07 + 4.606837e-13*mnk + 1.498512e-11*mn + 3.918575e-11*mk + 5.127312e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 25: // node 12
            mu    = 5.265666e-07 + 6.768453e-11*mnk + -7.996460e-11*mn + 1.666515e-09*mk + 3.016302e-09*nk;
            sigma = 6.399890e-07 + 4.308984e-13*mnk + 2.240832e-11*mn + 5.539865e-11*mk + 6.012762e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 26: // node 13
            mu    = 5.854250e-07 + 7.920080e-11*mnk + -4.312371e-10*mn + 1.570852e-09*mk + 2.897047e-09*nk;
            sigma = 6.898829e-07 + 9.474129e-12*mnk + 1.542634e-10*mn + 8.782591e-11*mk + 7.385075e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 27: // node 13
            mu    = 5.451164e-07 + 7.640014e-11*mnk + -6.124134e-10*mn + 1.424546e-09*mk + 2.626436e-09*nk;
            sigma = 6.136947e-07 + 1.383671e-12*mnk + 2.030638e-10*mn + 2.081739e-10*mk + 2.144467e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 28: // node 14
            mu    = 5.872356e-07 + 7.825684e-11*mnk + -4.333443e-10*mn + 1.679397e-09*mk + 3.005435e-09*nk;
            sigma = 7.236303e-07 + 2.564661e-12*mnk + 2.515660e-10*mn + 2.039354e-10*mk + 2.183537e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 29: // node 14
            mu    = 5.806772e-07 + 7.879295e-11*mnk + -7.616645e-10*mn + 1.487116e-09*mk + 2.601874e-09*nk;
            sigma = 7.199230e-07 + 1.651169e-12*mnk + 3.515796e-10*mn + 2.983138e-10*mk + 3.562135e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 30: // node 15
            mu    = 6.008916e-07 + 8.450100e-11*mnk + -3.407193e-10*mn + 1.920816e-09*mk + 3.206279e-09*nk;
            sigma = 6.852550e-07 + 4.222922e-12*mnk + 1.716549e-10*mn + 1.196224e-10*mk + 1.923672e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 31: // node 15
            mu    = 6.049290e-07 + 8.384803e-11*mnk + -4.844474e-10*mn + 1.822356e-09*mk + 3.099135e-09*nk;
            sigma = 7.446548e-07 + 1.080379e-11*mnk + -3.917419e-13*mn + -5.299022e-11*mk + 7.561419e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 32: // node 16
            mu    = 5.847983e-07 + 7.985392e-11*mnk + -5.136920e-10*mn + 1.604278e-09*mk + 2.872621e-09*nk;
            sigma = 7.258348e-07 + 2.105082e-12*mnk + 2.925803e-10*mn + 2.668062e-10*mk + 3.169975e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 33: // node 16
            mu    = 5.692598e-07 + 7.703683e-11*mnk + -9.022247e-11*mn + 2.001776e-09*mk + 3.381385e-09*nk;
            sigma = 7.167982e-07 + 4.668696e-12*mnk + 2.669372e-10*mn + 3.232302e-10*mk + 2.501175e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 34: // node 17
            mu    = 5.344454e-07 + 7.110007e-11*mnk + -4.274161e-10*mn + 1.470139e-09*mk + 2.724523e-09*nk;
            sigma = 6.004602e-07 + 1.634435e-12*mnk + 9.119124e-11*mn + 1.183724e-10*mk + 9.917033e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 35: // node 17
            mu    = 5.299293e-07 + 6.803053e-11*mnk + -1.259036e-10*mn + 1.688626e-09*mk + 3.017974e-09*nk;
            sigma = 6.198819e-07 + 3.002163e-13*mnk + 3.248806e-11*mn + 7.237567e-11*mk + 6.760120e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 36: // node 18
            mu    = 5.468983e-07 + 7.076063e-11*mnk + -3.807323e-10*mn + 1.516048e-09*mk + 2.725249e-09*nk;
            sigma = 6.586774e-07 + 1.287209e-12*mnk + 8.682976e-11*mn + 1.253534e-10*mk + 9.683318e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 37: // node 18
            mu    = 5.067599e-07 + 6.795031e-11*mnk + -1.350813e-10*mn + 1.585363e-09*mk + 2.898263e-09*nk;
            sigma = 5.465379e-07 + 5.348396e-13*mnk + 3.449688e-11*mn + 5.544574e-11*mk + 6.214726e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 38: // node 19
            mu    = 5.134214e-07 + 6.817183e-11*mnk + -9.901022e-11*mn + 1.663628e-09*mk + 3.018796e-09*nk;
            sigma = 5.640162e-07 + 5.132867e-13*mnk + 1.290475e-11*mn + 4.780033e-11*mk + 5.015694e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 39: // node 19
            mu    = 5.368837e-07 + 6.826187e-11*mnk + -1.479424e-10*mn + 1.619711e-09*mk + 2.943221e-09*nk;
            sigma = 6.391290e-07 + 5.036981e-13*mnk + 1.138776e-11*mn + 4.142007e-11*mk + 5.003563e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 40: // node 20
            mu    = 5.103642e-07 + 6.822038e-11*mnk + -1.178451e-10*mn + 1.662917e-09*mk + 3.007687e-09*nk;
            sigma = 5.788666e-07 + 4.459034e-13*mnk + 1.878462e-11*mn + 5.025264e-11*mk + 6.374513e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 41: // node 20
            mu    = 5.397179e-07 + 6.822589e-11*mnk + -1.123143e-10*mn + 1.636110e-09*mk + 2.979555e-09*nk;
            sigma = 6.508375e-07 + 5.842850e-13*mnk + 2.220993e-11*mn + 4.467521e-11*mk + 6.062142e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 42: // node 21
            mu    = 5.317256e-07 + 6.815687e-11*mnk + -9.284797e-11*mn + 1.704904e-09*mk + 3.090069e-09*nk;
            sigma = 6.667179e-07 + 8.487308e-13*mnk + 6.779641e-12*mn + 3.445108e-11*mk + 4.021661e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 43: // node 21
            mu    = 5.208321e-07 + 6.802506e-11*mnk + -1.326302e-10*mn + 1.636123e-09*mk + 2.953639e-09*nk;
            sigma = 6.146244e-07 + 5.102391e-13*mnk + 2.977270e-11*mn + 4.805156e-11*mk + 4.482061e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 44: // node 22
            mu    = 5.231029e-07 + 6.809005e-11*mnk + -1.002971e-10*mn + 1.636146e-09*mk + 2.961246e-09*nk;
            sigma = 6.013081e-07 + 5.339565e-13*mnk + 3.333609e-11*mn + 5.910749e-11*mk + 6.083720e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 45: // node 22
            mu    = 5.206402e-07 + 6.861823e-11*mnk + -1.391267e-10*mn + 1.648974e-09*mk + 2.997566e-09*nk;
            sigma = 6.026599e-07 + 5.045647e-13*mnk + 2.262201e-11*mn + 6.262390e-11*mk + 5.576603e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 46: // node 23
            mu    = 5.064498e-07 + 6.815025e-11*mnk + -1.127577e-10*mn + 1.665394e-09*mk + 3.018353e-09*nk;
            sigma = 5.660005e-07 + 2.953845e-13*mnk + 2.742789e-11*mn + 5.041331e-11*mk + 6.274031e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 47: // node 23
            mu    = 5.190310e-07 + 6.782314e-11*mnk + -1.248876e-10*mn + 1.598032e-09*mk + 2.924428e-09*nk;
            sigma = 5.859136e-07 + 4.583653e-13*mnk + 1.862309e-11*mn + 4.806724e-11*mk + 4.257829e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 48: // node 24
            mu    = 5.185796e-07 + 6.831819e-11*mnk + -1.265035e-10*mn + 1.636102e-09*mk + 2.981898e-09*nk;
            sigma = 5.812255e-07 + 4.566174e-13*mnk + 2.544243e-11*mn + 5.397706e-11*mk + 6.632500e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 49: // node 24
            mu    = 5.156076e-07 + 6.778105e-11*mnk + -8.437938e-11*mn + 1.645000e-09*mk + 2.975996e-09*nk;
            sigma = 5.615274e-07 + 3.821451e-13*mnk + 3.573403e-11*mn + 5.190488e-11*mk + 6.109547e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 50: // node 25
            mu    = 5.552318e-07 + 7.355154e-11*mnk + -3.886047e-10*mn + 1.558966e-09*mk + 2.812087e-09*nk;
            sigma = 6.491666e-07 + 1.262363e-12*mnk + 1.124913e-10*mn + 1.229869e-10*mk + 1.540899e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 51: // node 25
            mu    = 5.071344e-07 + 6.768929e-11*mnk + -1.394035e-10*mn + 1.555916e-09*mk + 2.884728e-09*nk;
            sigma = 5.456455e-07 + 3.701218e-13*mnk + 1.235565e-11*mn + 4.115298e-11*mk + 3.147907e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 52: // node 26
            mu    = 5.220811e-07 + 6.830379e-11*mnk + -1.606740e-10*mn + 1.626739e-09*mk + 2.978407e-09*nk;
            sigma = 5.822232e-07 + 4.417249e-13*mnk + 1.807460e-11*mn + 5.077949e-11*mk + 5.385897e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 53: // node 26
            mu    = 5.305365e-07 + 6.930499e-11*mnk + -3.202027e-10*mn + 1.499278e-09*mk + 2.807434e-09*nk;
            sigma = 6.062084e-07 + 1.521465e-12*mnk + -1.894556e-11*mn + 2.097636e-11*mk + -1.678183e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 54: // node 27
            mu    = 5.365141e-07 + 6.899888e-11*mnk + -2.587134e-10*mn + 1.593261e-09*mk + 2.889037e-09*nk;
            sigma = 6.655126e-07 + 7.964453e-13*mnk + 1.103735e-11*mn + 5.334813e-11*mk + 4.773279e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 55: // node 27
            mu    = 5.276706e-07 + 6.956274e-11*mnk + -3.601891e-10*mn + 1.483759e-09*mk + 2.775141e-09*nk;
            sigma = 6.297032e-07 + 1.531971e-12*mnk + -6.907262e-13*mn + 5.045010e-11*mk + 9.056070e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 56: // node 28
            mu    = 5.420053e-07 + 7.117041e-11*mnk + -4.974038e-10*mn + 1.459098e-09*mk + 2.628189e-09*nk;
            sigma = 6.476256e-07 + 1.530540e-12*mnk + 8.780899e-11*mn + 1.246792e-10*mk + 1.119746e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 57: // node 28
            mu    = 5.149328e-07 + 6.784200e-11*mnk + -9.665357e-11*mn + 1.646485e-09*mk + 3.013190e-09*nk;
            sigma = 5.893493e-07 + 4.001245e-13*mnk + 2.461958e-11*mn + 5.148661e-11*mk + 4.889745e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 58: // node 29
            mu    = 5.255900e-07 + 6.786821e-11*mnk + -6.500140e-11*mn + 1.688902e-09*mk + 3.001173e-09*nk;
            sigma = 6.195277e-07 + 4.398077e-13*mnk + 3.061241e-11*mn + 4.846374e-11*mk + 5.632437e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 59: // node 29
            mu    = 5.208931e-07 + 6.796154e-11*mnk + -1.045580e-10*mn + 1.645969e-09*mk + 3.019445e-09*nk;
            sigma = 6.126106e-07 + 4.862725e-13*mnk + 2.656993e-11*mn + 5.149211e-11*mk + 5.389626e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 60: // node 30
            mu    = 5.265286e-07 + 6.793564e-11*mnk + -7.189099e-11*mn + 1.656573e-09*mk + 2.979553e-09*nk;
            sigma = 6.395850e-07 + 7.677435e-13*mnk + 1.316460e-11*mn + 4.184614e-11*mk + 5.221730e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 61: // node 30
            mu    = 5.145376e-07 + 6.784335e-11*mnk + -1.051588e-10*mn + 1.635049e-09*mk + 3.000886e-09*nk;
            sigma = 5.647746e-07 + 3.579749e-13*mnk + 2.836230e-11*mn + 5.204211e-11*mk + 6.824112e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 62: // node 31
            mu    = 5.255507e-07 + 6.792859e-11*mnk + -1.155430e-10*mn + 1.603461e-09*mk + 2.916228e-09*nk;
            sigma = 5.942084e-07 + 6.461464e-13*mnk + 1.780135e-11*mn + 4.575905e-11*mk + 4.315604e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 63: // node 31
            mu    = 5.203696e-07 + 6.802131e-11*mnk + -9.089474e-11*mn + 1.650291e-09*mk + 2.961259e-09*nk;
            sigma = 5.969045e-07 + 5.094830e-13*mnk + 1.293906e-11*mn + 4.133834e-11*mk + 4.200090e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 64: // node 32
            mu    = 5.231576e-07 + 6.761471e-11*mnk + -4.793847e-11*mn + 1.653379e-09*mk + 3.015021e-09*nk;
            sigma = 5.847438e-07 + 5.130015e-13*mnk + 2.488421e-11*mn + 4.040013e-11*mk + 5.981118e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 65: // node 32
            mu    = 5.114425e-07 + 6.780890e-11*mnk + -7.866192e-11*mn + 1.629375e-09*mk + 2.938603e-09*nk;
            sigma = 5.693536e-07 + 3.540383e-13*mnk + 2.490204e-11*mn + 4.602318e-11*mk + 4.618505e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        default:
            mu    = 5.249134e-07 + 1.604311e-09*mk + -1.793648e-10*mn + 6.864500e-11*mnk + 2.922754e-09*nk;
            sigma = 6.060802e-07 + 5.905245e-11*mk + 2.927438e-11*mn + 7.495901e-13*mnk + 5.780894e-11*nk;
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
