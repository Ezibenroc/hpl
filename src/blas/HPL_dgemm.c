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
            mu    = 2.031875e-06 + 6.919787e-11*mnk + 1.894452e-10*mn + 1.957824e-09*mk + 3.059798e-09*nk;
            sigma = 3.568280e-07 + 6.546626e-13*mnk + 3.490725e-12*mn + 1.151553e-11*mk + 5.660122e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 3: // node 1
            mu    = 1.823625e-06 + 6.479946e-11*mnk + 1.793597e-10*mn + 1.830960e-09*mk + 3.183051e-09*nk;
            sigma = 2.867134e-07 + 2.095819e-13*mnk + 5.485074e-12*mn + 3.488718e-11*mk + 3.129052e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 4: // node 2
            mu    = 1.793563e-06 + 6.774704e-11*mnk + 1.774582e-10*mn + 1.938141e-09*mk + 3.001103e-09*nk;
            sigma = 2.294533e-07 + 9.302963e-13*mnk + -1.058776e-12*mn + 1.914962e-11*mk + 4.256655e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 5: // node 2
            mu    = 2.039063e-06 + 6.470399e-11*mnk + 1.835837e-10*mn + 1.821377e-09*mk + 3.190992e-09*nk;
            sigma = 4.459159e-07 + 1.710315e-13*mnk + 1.579534e-11*mn + 2.760985e-11*mk + 3.029784e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 6: // node 3
            mu    = 1.959375e-06 + 6.661109e-11*mnk + 1.764182e-10*mn + 1.916735e-09*mk + 3.100890e-09*nk;
            sigma = 2.402006e-07 + 7.941621e-13*mnk + 1.217797e-12*mn + 1.612409e-11*mk + 1.541685e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 7: // node 3
            mu    = 1.847375e-06 + 6.463682e-11*mnk + 1.764626e-10*mn + 1.797731e-09*mk + 3.131726e-09*nk;
            sigma = 3.594620e-07 + 1.994159e-13*mnk + 1.555055e-12*mn + 2.090426e-11*mk + 2.464432e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 8: // node 4
            mu    = 1.998187e-06 + 6.474189e-11*mnk + 1.824372e-10*mn + 1.811525e-09*mk + 3.143610e-09*nk;
            sigma = 3.049520e-07 + 2.106099e-13*mnk + 8.917642e-12*mn + 2.398301e-11*mk + 2.528166e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 9: // node 4
            mu    = 1.817000e-06 + 6.481249e-11*mnk + 1.785609e-10*mn + 1.806540e-09*mk + 3.135141e-09*nk;
            sigma = 3.134492e-07 + 2.072184e-13*mnk + 8.829644e-13*mn + 2.808735e-11*mk + 2.638956e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 10: // node 5
            mu    = 1.820687e-06 + 6.705632e-11*mnk + 1.894869e-10*mn + 1.946019e-09*mk + 3.075501e-09*nk;
            sigma = 2.678372e-07 + 7.033274e-13*mnk + 4.570862e-12*mn + 3.233506e-11*mk + 3.230984e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 11: // node 5
            mu    = 1.947375e-06 + 6.448573e-11*mnk + 1.862013e-10*mn + 1.756690e-09*mk + 3.062519e-09*nk;
            sigma = 3.068313e-07 + 2.164357e-13*mnk + 4.195734e-12*mn + 2.529380e-11*mk + 2.894407e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 12: // node 6
            mu    = 1.713125e-06 + 6.708489e-11*mnk + 1.828127e-10*mn + 1.940450e-09*mk + 3.026592e-09*nk;
            sigma = 3.035692e-07 + 7.542549e-13*mnk + 3.441705e-12*mn + 1.852717e-11*mk + 3.062344e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 13: // node 6
            mu    = 1.699313e-06 + 6.420634e-11*mnk + 1.880559e-10*mn + 1.742905e-09*mk + 3.115455e-09*nk;
            sigma = 3.698865e-07 + 1.329991e-13*mnk + 1.889689e-12*mn + 2.289241e-11*mk + 2.447070e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 14: // node 7
            mu    = 1.886312e-06 + 6.468654e-11*mnk + 1.800674e-10*mn + 1.832612e-09*mk + 3.179120e-09*nk;
            sigma = 2.518608e-07 + 1.699576e-13*mnk + 3.179885e-12*mn + 2.565463e-11*mk + 3.214284e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 15: // node 7
            mu    = 1.796812e-06 + 6.435848e-11*mnk + 1.856477e-10*mn + 1.720842e-09*mk + 3.073904e-09*nk;
            sigma = 3.626939e-07 + 1.829955e-13*mnk + 2.203707e-12*mn + 1.879176e-11*mk + 2.035166e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 16: // node 8
            mu    = 1.804438e-06 + 6.569211e-11*mnk + 2.018187e-10*mn + 1.868968e-09*mk + 3.090713e-09*nk;
            sigma = 2.663016e-07 + 6.413282e-13*mnk + 6.523818e-12*mn + 2.301578e-11*mk + 3.021082e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 17: // node 8
            mu    = 1.838500e-06 + 6.451100e-11*mnk + 1.848484e-10*mn + 1.782762e-09*mk + 3.130368e-09*nk;
            sigma = 2.998226e-07 + 1.838364e-13*mnk + 4.468914e-12*mn + 2.228399e-11*mk + 2.692329e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 18: // node 9
            mu    = 1.917312e-06 + 6.470338e-11*mnk + 1.846905e-10*mn + 1.788500e-09*mk + 3.091349e-09*nk;
            sigma = 3.490054e-07 + 1.821914e-13*mnk + 2.405213e-12*mn + 2.555409e-11*mk + 3.335459e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 19: // node 9
            mu    = 1.847000e-06 + 6.456668e-11*mnk + 1.829354e-10*mn + 1.775471e-09*mk + 3.084816e-09*nk;
            sigma = 3.180082e-07 + 1.891671e-13*mnk + 2.728627e-12*mn + 1.914810e-11*mk + 2.859961e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 20: // node 10
            mu    = 1.761937e-06 + 6.477195e-11*mnk + 1.855436e-10*mn + 1.821154e-09*mk + 3.179540e-09*nk;
            sigma = 2.238800e-07 + 1.737691e-13*mnk + 8.993402e-12*mn + 2.347787e-11*mk + 2.859278e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 21: // node 10
            mu    = 1.922875e-06 + 6.479865e-11*mnk + 1.770054e-10*mn + 1.813126e-09*mk + 3.159407e-09*nk;
            sigma = 3.366890e-07 + 1.816816e-13*mnk + 4.338806e-12*mn + 2.668150e-11*mk + 3.678854e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 22: // node 11
            mu    = 1.861937e-06 + 6.447836e-11*mnk + 1.815856e-10*mn + 1.771325e-09*mk + 3.098444e-09*nk;
            sigma = 4.185254e-07 + 1.938070e-13*mnk + 1.537229e-12*mn + 2.329886e-11*mk + 2.229895e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 23: // node 11
            mu    = 2.156563e-06 + 6.570887e-11*mnk + 1.694949e-10*mn + 1.898300e-09*mk + 3.101271e-09*nk;
            sigma = 4.030012e-07 + 6.282040e-13*mnk + 5.396304e-13*mn + 3.475840e-11*mk + 8.167141e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 24: // node 12
            mu    = 2.022250e-06 + 6.469280e-11*mnk + 1.857814e-10*mn + 1.800993e-09*mk + 3.138911e-09*nk;
            sigma = 2.922302e-07 + 1.848088e-13*mnk + 8.117540e-12*mn + 2.387615e-11*mk + 2.887370e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 25: // node 12
            mu    = 1.895062e-06 + 6.447220e-11*mnk + 1.785727e-10*mn + 1.761633e-09*mk + 3.116495e-09*nk;
            sigma = 3.338113e-07 + 2.172739e-13*mnk + 4.419396e-12*mn + 2.463158e-11*mk + 2.695258e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 26: // node 13
            mu    = 1.946938e-06 + 6.525883e-11*mnk + 1.877887e-10*mn + 1.868990e-09*mk + 3.163174e-09*nk;
            sigma = 2.636078e-07 + 4.638386e-13*mnk + 6.017135e-12*mn + 3.442548e-11*mk + 6.810535e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 27: // node 13
            mu    = 1.915750e-06 + 6.461585e-11*mnk + 1.910115e-10*mn + 1.802216e-09*mk + 3.116125e-09*nk;
            sigma = 4.160324e-07 + 1.383615e-13*mnk + 6.492119e-12*mn + 2.946518e-11*mk + 3.494893e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 28: // node 14
            mu    = 1.888000e-06 + 6.537768e-11*mnk + 1.864690e-10*mn + 1.875896e-09*mk + 3.131536e-09*nk;
            sigma = 3.060832e-07 + 5.750957e-13*mnk + 3.680653e-12*mn + 2.101870e-11*mk + -1.092789e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 29: // node 14
            mu    = 1.876250e-06 + 6.461061e-11*mnk + 1.820832e-10*mn + 1.808254e-09*mk + 3.121687e-09*nk;
            sigma = 4.447404e-07 + 1.892946e-13*mnk + 3.545912e-12*mn + 3.104860e-11*mk + 2.832664e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 30: // node 15
            mu    = 2.027313e-06 + 6.929758e-11*mnk + 1.922304e-10*mn + 1.948290e-09*mk + 3.141456e-09*nk;
            sigma = 3.439558e-07 + 5.432291e-13*mnk + -5.454007e-13*mn + 2.034084e-11*mk + 5.456848e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 31: // node 15
            mu    = 1.897313e-06 + 6.453426e-11*mnk + 1.843830e-10*mn + 1.780865e-09*mk + 3.107970e-09*nk;
            sigma = 3.715223e-07 + 1.996895e-13*mnk + 2.746758e-12*mn + 3.419336e-11*mk + 2.789535e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 32: // node 16
            mu    = 1.805000e-06 + 6.668373e-11*mnk + 1.963515e-10*mn + 1.902175e-09*mk + 3.055625e-09*nk;
            sigma = 1.983260e-07 + 7.795085e-13*mnk + -1.877178e-13*mn + 1.898083e-11*mk + 2.763580e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 33: // node 16
            mu    = 2.061687e-06 + 6.471312e-11*mnk + 1.845531e-10*mn + 1.828339e-09*mk + 3.097299e-09*nk;
            sigma = 3.943524e-07 + 2.342175e-13*mnk + 1.423471e-12*mn + 3.001454e-11*mk + 2.359264e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 34: // node 17
            mu    = 1.927125e-06 + 6.918716e-11*mnk + 1.856543e-10*mn + 1.951536e-09*mk + 3.074791e-09*nk;
            sigma = 2.357691e-07 + 6.183317e-13*mnk + -1.695110e-12*mn + 9.921222e-12*mk + 5.907562e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 35: // node 17
            mu    = 1.840500e-06 + 6.475463e-11*mnk + 1.927742e-10*mn + 1.831524e-09*mk + 3.180327e-09*nk;
            sigma = 2.561754e-07 + 2.190658e-13*mnk + 2.123358e-11*mn + 3.975663e-11*mk + 3.633289e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 36: // node 18
            mu    = 2.027813e-06 + 6.846164e-11*mnk + 1.897923e-10*mn + 1.946888e-09*mk + 3.061945e-09*nk;
            sigma = 3.002064e-07 + 6.535486e-13*mnk + 1.534159e-12*mn + 2.021704e-11*mk + 4.761329e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 37: // node 18
            mu    = 1.824375e-06 + 6.456765e-11*mnk + 1.900092e-10*mn + 1.774452e-09*mk + 3.080977e-09*nk;
            sigma = 3.967755e-07 + 1.613003e-13*mnk + 7.973807e-12*mn + 3.017221e-11*mk + 3.535747e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 38: // node 19
            mu    = 1.970375e-06 + 6.494850e-11*mnk + 1.920121e-10*mn + 1.865684e-09*mk + 3.167318e-09*nk;
            sigma = 2.644972e-07 + 3.289898e-13*mnk + 3.262568e-12*mn + 3.674321e-11*mk + 2.242468e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 39: // node 19
            mu    = 2.141750e-06 + 6.461223e-11*mnk + 1.806515e-10*mn + 1.815777e-09*mk + 3.100431e-09*nk;
            sigma = 5.203626e-07 + 1.694313e-13*mnk + 5.948582e-12*mn + 2.917513e-11*mk + 3.102934e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 40: // node 20
            mu    = 1.865688e-06 + 6.545990e-11*mnk + 1.775456e-10*mn + 1.873070e-09*mk + 3.093043e-09*nk;
            sigma = 2.772743e-07 + 5.494137e-13*mnk + 3.727996e-12*mn + 2.842684e-11*mk + 3.715435e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 41: // node 20
            mu    = 1.844062e-06 + 6.457189e-11*mnk + 1.765124e-10*mn + 1.793335e-09*mk + 3.132016e-09*nk;
            sigma = 3.830493e-07 + 2.143595e-13*mnk + -5.900886e-13*mn + 2.196805e-11*mk + 2.230576e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 42: // node 21
            mu    = 1.899687e-06 + 6.633246e-11*mnk + 1.823384e-10*mn + 1.934606e-09*mk + 3.045230e-09*nk;
            sigma = 2.948335e-07 + 7.421231e-13*mnk + 3.267378e-12*mn + 1.649370e-11*mk + 3.111276e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 43: // node 21
            mu    = 1.940625e-06 + 6.475003e-11*mnk + 1.823388e-10*mn + 1.810218e-09*mk + 3.145768e-09*nk;
            sigma = 1.735386e-07 + 1.882534e-13*mnk + 6.707123e-12*mn + 2.474575e-11*mk + 3.197115e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 44: // node 22
            mu    = 1.919000e-06 + 6.483385e-11*mnk + 1.807678e-10*mn + 1.830860e-09*mk + 3.168339e-09*nk;
            sigma = 2.798954e-07 + 2.693333e-13*mnk + 5.173278e-12*mn + 3.026642e-11*mk + 2.443786e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 45: // node 22
            mu    = 1.875250e-06 + 6.506807e-11*mnk + 1.851101e-10*mn + 1.817828e-09*mk + 3.148130e-09*nk;
            sigma = 2.939988e-07 + 2.578584e-13*mnk + 1.082036e-11*mn + 2.882633e-11*mk + 2.512614e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 46: // node 23
            mu    = 1.888875e-06 + 6.480356e-11*mnk + 1.900429e-10*mn + 1.840601e-09*mk + 3.170487e-09*nk;
            sigma = 3.385195e-07 + 2.459952e-13*mnk + 3.035029e-12*mn + 2.264050e-11*mk + 2.355389e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 47: // node 23
            mu    = 1.901437e-06 + 6.434747e-11*mnk + 1.830547e-10*mn + 1.746030e-09*mk + 3.068161e-09*nk;
            sigma = 3.148678e-07 + 1.557446e-13*mnk + 6.254224e-12*mn + 2.339877e-11*mk + 3.333573e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 48: // node 24
            mu    = 1.881375e-06 + 6.485936e-11*mnk + 1.847761e-10*mn + 1.815567e-09*mk + 3.157869e-09*nk;
            sigma = 2.714981e-07 + 2.016489e-13*mnk + 1.018895e-11*mn + 3.048033e-11*mk + 2.878991e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 49: // node 24
            mu    = 2.094625e-06 + 6.454334e-11*mnk + 1.819293e-10*mn + 1.771843e-09*mk + 3.082092e-09*nk;
            sigma = 6.198059e-07 + 1.362692e-13*mnk + 7.659468e-12*mn + 2.986360e-11*mk + 3.252807e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 50: // node 25
            mu    = 2.032188e-06 + 6.961552e-11*mnk + 2.129481e-10*mn + 1.960147e-09*mk + 3.104120e-09*nk;
            sigma = 3.254480e-07 + 6.309041e-13*mnk + 2.279188e-12*mn + 1.972113e-11*mk + 5.946323e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 51: // node 25
            mu    = 1.833188e-06 + 6.424904e-11*mnk + 1.879407e-10*mn + 1.734425e-09*mk + 3.124894e-09*nk;
            sigma = 3.375893e-07 + 1.355014e-13*mnk + 5.629758e-12*mn + 2.100043e-11*mk + 2.561372e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 52: // node 26
            mu    = 1.880250e-06 + 6.467900e-11*mnk + 1.861967e-10*mn + 1.818913e-09*mk + 3.112147e-09*nk;
            sigma = 2.736377e-07 + 2.074884e-13*mnk + 6.304114e-12*mn + 2.695679e-11*mk + 2.317838e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 53: // node 26
            mu    = 1.970313e-06 + 6.504954e-11*mnk + 1.947658e-10*mn + 1.858818e-09*mk + 3.161671e-09*nk;
            sigma = 4.829437e-07 + 4.090640e-13*mnk + 6.153819e-12*mn + 3.070785e-11*mk + 4.566329e-12*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 54: // node 27
            mu    = 1.980687e-06 + 6.489013e-11*mnk + 1.833492e-10*mn + 1.859267e-09*mk + 3.144332e-09*nk;
            sigma = 2.983720e-07 + 3.012809e-13*mnk + 9.325486e-12*mn + 3.245553e-11*mk + 1.813336e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 55: // node 27
            mu    = 1.974188e-06 + 6.503106e-11*mnk + 1.845489e-10*mn + 1.865478e-09*mk + 3.117690e-09*nk;
            sigma = 2.836010e-07 + 4.342045e-13*mnk + 9.958234e-12*mn + 3.340051e-11*mk + 1.455744e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 56: // node 28
            mu    = 1.977875e-06 + 6.695951e-11*mnk + 1.957752e-10*mn + 1.899119e-09*mk + 3.049061e-09*nk;
            sigma = 2.558424e-07 + 7.509952e-13*mnk + -3.574783e-12*mn + 1.404136e-11*mk + 1.629091e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 57: // node 28
            mu    = 1.857938e-06 + 6.454369e-11*mnk + 1.840635e-10*mn + 1.772487e-09*mk + 3.118998e-09*nk;
            sigma = 3.404247e-07 + 1.500717e-13*mnk + 5.969057e-12*mn + 2.903712e-11*mk + 3.406740e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 58: // node 29
            mu    = 1.896375e-06 + 6.469370e-11*mnk + 1.817856e-10*mn + 1.812727e-09*mk + 3.160018e-09*nk;
            sigma = 2.431139e-07 + 1.937044e-13*mnk + 4.741775e-12*mn + 2.678295e-11*mk + 2.371321e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 59: // node 29
            mu    = 1.762000e-06 + 6.473257e-11*mnk + 1.839720e-10*mn + 1.780512e-09*mk + 3.115920e-09*nk;
            sigma = 2.861419e-07 + 1.747565e-13*mnk + 7.637732e-12*mn + 2.585554e-11*mk + 2.696759e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 60: // node 30
            mu    = 1.794250e-06 + 6.454037e-11*mnk + 1.836723e-10*mn + 1.785702e-09*mk + 3.081991e-09*nk;
            sigma = 3.752261e-07 + 2.282857e-13*mnk + 7.378904e-12*mn + 3.325473e-11*mk + 2.647763e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 61: // node 30
            mu    = 1.848187e-06 + 6.441980e-11*mnk + 1.813596e-10*mn + 1.757265e-09*mk + 3.074501e-09*nk;
            sigma = 4.537737e-07 + 2.015059e-13*mnk + 2.786832e-14*mn + 2.290080e-11*mk + 2.853973e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 62: // node 31
            mu    = 1.764000e-06 + 6.480890e-11*mnk + 1.824950e-10*mn + 1.770882e-09*mk + 3.135694e-09*nk;
            sigma = 2.465163e-07 + 2.162590e-13*mnk + 7.407017e-12*mn + 2.312908e-11*mk + 3.363025e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 63: // node 31
            mu    = 1.943188e-06 + 6.471888e-11*mnk + 1.846575e-10*mn + 1.794906e-09*mk + 3.165110e-09*nk;
            sigma = 3.981035e-07 + 2.650383e-13*mnk + 8.325095e-12*mn + 2.837281e-11*mk + 2.706379e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 64: // node 32
            mu    = 1.809313e-06 + 6.456844e-11*mnk + 1.788395e-10*mn + 1.740258e-09*mk + 3.081749e-09*nk;
            sigma = 3.340864e-07 + 1.869923e-13*mnk + 5.135284e-12*mn + 2.212050e-11*mk + 2.931581e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 65: // node 32
            mu    = 1.880875e-06 + 6.450284e-11*mnk + 1.834250e-10*mn + 1.777955e-09*mk + 3.049778e-09*nk;
            sigma = 3.672168e-07 + 1.827344e-13*mnk + 7.442627e-12*mn + 2.936419e-11*mk + 2.450741e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        default:
            mu    = 1.901050e-06 + 1.830098e-09*mk + 1.849726e-10*mn + 6.532377e-11*mnk + 3.113753e-09*nk;
            sigma = 3.266040e-07 + 2.553385e-11*mk + 4.807115e-12*mn + 3.315110e-13*mnk + 2.725599e-11*nk;
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
