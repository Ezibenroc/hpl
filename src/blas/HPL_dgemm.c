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
            mu    = 5.176631e-07 + 6.350885e-11*mnk + 9.545662e-11*mn + 2.114035e-09*mk + 3.257169e-09*nk;
            sigma = 4.425990e-07 + 3.678641e-13*mnk + 1.552287e-11*mn + 3.575659e-11*mk + 5.896758e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 3: // node 1
            mu    = 4.868351e-07 + 6.184670e-11*mnk + 1.514162e-10*mn + 2.066126e-09*mk + 3.256477e-09*nk;
            sigma = 3.832751e-07 + 8.351036e-14*mnk + 1.460052e-11*mn + 3.756080e-11*mk + 6.175627e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 4: // node 2
            mu    = 5.023491e-07 + 6.236673e-11*mnk + 1.316263e-10*mn + 2.080258e-09*mk + 3.315941e-09*nk;
            sigma = 3.940485e-07 + 2.407008e-13*mnk + 8.341990e-12*mn + 2.953215e-11*mk + 3.031578e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 5: // node 2
            mu    = 4.974726e-07 + 6.205674e-11*mnk + 1.471030e-10*mn + 2.061982e-09*mk + 3.324590e-09*nk;
            sigma = 3.928921e-07 + 1.207550e-13*mnk + 1.482112e-11*mn + 4.329756e-11*mk + 5.732950e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 6: // node 3
            mu    = 5.033389e-07 + 6.232760e-11*mnk + 1.242248e-10*mn + 2.090036e-09*mk + 3.350375e-09*nk;
            sigma = 3.950192e-07 + 1.985379e-13*mnk + -8.736210e-13*mn + 4.999914e-11*mk + 4.367242e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 7: // node 3
            mu    = 4.818443e-07 + 6.181782e-11*mnk + 1.464827e-10*mn + 2.049636e-09*mk + 3.310328e-09*nk;
            sigma = 3.718770e-07 + 5.236684e-14*mnk + 2.055300e-11*mn + 6.184270e-11*mk + 5.224286e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 8: // node 4
            mu    = 4.940256e-07 + 6.178801e-11*mnk + 1.568555e-10*mn + 2.019072e-09*mk + 3.324394e-09*nk;
            sigma = 4.014520e-07 + 7.477089e-14*mnk + 1.343630e-11*mn + 4.749938e-11*mk + 5.315425e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 9: // node 4
            mu    = 4.811863e-07 + 6.201270e-11*mnk + 1.627870e-10*mn + 2.029928e-09*mk + 3.319502e-09*nk;
            sigma = 3.639953e-07 + 8.976655e-14*mnk + 1.129242e-11*mn + 4.157610e-11*mk + 3.377701e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 12: // node 6
            mu    = 4.932771e-07 + 6.404389e-11*mnk + 1.094798e-10*mn + 2.115466e-09*mk + 3.254405e-09*nk;
            sigma = 3.737196e-07 + 3.307311e-13*mnk + 5.095776e-12*mn + 4.167380e-11*mk + 5.242706e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 13: // node 6
            mu    = 4.952999e-07 + 6.156356e-11*mnk + 1.619846e-10*mn + 2.010381e-09*mk + 3.260128e-09*nk;
            sigma = 3.919210e-07 + 4.685596e-14*mnk + 1.597936e-11*mn + 5.314991e-11*mk + 4.447439e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 14: // node 7
            mu    = 5.042268e-07 + 6.194369e-11*mnk + 1.547242e-10*mn + 2.072948e-09*mk + 3.316321e-09*nk;
            sigma = 4.045263e-07 + 6.879347e-14*mnk + 1.617777e-11*mn + 5.380012e-11*mk + 6.251309e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 15: // node 7
            mu    = 4.832945e-07 + 6.133599e-11*mnk + 1.617922e-10*mn + 1.987801e-09*mk + 3.263344e-09*nk;
            sigma = 3.675208e-07 + 5.961817e-14*mnk + 1.239695e-11*mn + 5.125212e-11*mk + 4.581797e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 16: // node 8
            mu    = 4.935647e-07 + 6.305588e-11*mnk + 8.097412e-11*mn + 2.136391e-09*mk + 3.296024e-09*nk;
            sigma = 3.732866e-07 + 3.074514e-13*mnk + -1.437719e-12*mn + 5.488336e-11*mk + 2.873476e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 17: // node 8
            mu    = 4.924134e-07 + 6.185977e-11*mnk + 1.539445e-10*mn + 2.020350e-09*mk + 3.242825e-09*nk;
            sigma = 3.891436e-07 + 9.336698e-14*mnk + 1.088144e-11*mn + 5.702408e-11*mk + 4.484375e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 18: // node 9
            mu    = 4.892414e-07 + 6.171223e-11*mnk + 1.595345e-10*mn + 2.034004e-09*mk + 3.332226e-09*nk;
            sigma = 3.878699e-07 + 4.386713e-14*mnk + 1.441969e-11*mn + 5.279831e-11*mk + 6.446940e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 19: // node 9
            mu    = 4.843786e-07 + 6.179913e-11*mnk + 1.555239e-10*mn + 2.000865e-09*mk + 3.266388e-09*nk;
            sigma = 3.890611e-07 + 6.240094e-14*mnk + 1.385222e-11*mn + 4.388016e-11*mk + 5.956400e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 20: // node 10
            mu    = 4.947499e-07 + 6.177217e-11*mnk + 1.574658e-10*mn + 2.028801e-09*mk + 3.331912e-09*nk;
            sigma = 3.890667e-07 + 1.007106e-13*mnk + 1.003245e-11*mn + 3.787737e-11*mk + 4.712221e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 21: // node 10
            mu    = 5.065937e-07 + 6.194689e-11*mnk + 1.482196e-10*mn + 2.040617e-09*mk + 3.359802e-09*nk;
            sigma = 4.285429e-07 + 7.958848e-14*mnk + 1.011307e-11*mn + 5.080979e-11*mk + 4.923180e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 22: // node 11
            mu    = 4.918953e-07 + 6.166460e-11*mnk + 1.557420e-10*mn + 1.995393e-09*mk + 3.237157e-09*nk;
            sigma = 3.920778e-07 + 6.835200e-14*mnk + 1.274438e-11*mn + 4.930389e-11*mk + 3.991733e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 23: // node 11
            mu    = 4.882358e-07 + 6.259746e-11*mnk + 1.026824e-10*mn + 2.133298e-09*mk + 3.274179e-09*nk;
            sigma = 3.714272e-07 + 2.486803e-13*mnk + 9.217720e-12*mn + 6.577017e-11*mk + 3.444806e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 24: // node 12
            mu    = 4.798733e-07 + 6.180736e-11*mnk + 1.619451e-10*mn + 2.024514e-09*mk + 3.302979e-09*nk;
            sigma = 3.667449e-07 + 5.482824e-14*mnk + 1.475172e-11*mn + 6.180322e-11*mk + 5.013946e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 25: // node 12
            mu    = 4.860476e-07 + 6.169386e-11*mnk + 1.533953e-10*mn + 1.981956e-09*mk + 3.248501e-09*nk;
            sigma = 3.819153e-07 + 1.135578e-13*mnk + 6.770309e-12*mn + 3.140087e-11*mk + 3.445871e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 34: // node 17
            mu    = 5.124815e-07 + 6.416433e-11*mnk + 9.647405e-11*mn + 2.198422e-09*mk + 3.234558e-09*nk;
            sigma = 4.265800e-07 + 5.245396e-13*mnk + 2.282277e-11*mn + 2.308210e-11*mk + 6.405347e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 35: // node 17
            mu    = 4.885716e-07 + 6.196703e-11*mnk + 1.517621e-10*mn + 2.082042e-09*mk + 3.325126e-09*nk;
            sigma = 3.981721e-07 + 3.590415e-14*mnk + 1.902220e-11*mn + 5.693618e-11*mk + 7.530758e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 36: // node 18
            mu    = 5.108739e-07 + 6.392515e-11*mnk + 6.192583e-11*mn + 2.189185e-09*mk + 3.177964e-09*nk;
            sigma = 4.071176e-07 + 3.893158e-13*mnk + 2.963399e-11*mn + 4.066395e-11*mk + 6.123203e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 37: // node 18
            mu    = 4.820189e-07 + 6.158827e-11*mnk + 1.616031e-10*mn + 1.994777e-09*mk + 3.237100e-09*nk;
            sigma = 3.871231e-07 + 8.158790e-14*mnk + 9.505923e-12*mn + 4.721878e-11*mk + 3.142333e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 38: // node 19
            mu    = 4.923455e-07 + 6.196489e-11*mnk + 1.543992e-10*mn + 2.057428e-09*mk + 3.351357e-09*nk;
            sigma = 3.972481e-07 + 8.658301e-14*mnk + 1.747657e-11*mn + 5.091317e-11*mk + 5.874473e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 39: // node 19
            mu    = 4.915001e-07 + 6.183613e-11*mnk + 1.534767e-10*mn + 2.034158e-09*mk + 3.303009e-09*nk;
            sigma = 3.970659e-07 + 7.572456e-14*mnk + 1.091060e-11*mn + 3.755636e-11*mk + 6.291291e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 40: // node 20
            mu    = 4.847009e-07 + 6.204414e-11*mnk + 1.578919e-10*mn + 2.090352e-09*mk + 3.300387e-09*nk;
            sigma = 3.793946e-07 + 7.942706e-14*mnk + 1.671621e-11*mn + 5.365856e-11*mk + 5.252419e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 41: // node 20
            mu    = 4.829631e-07 + 6.179681e-11*mnk + 1.482244e-10*mn + 2.038635e-09*mk + 3.293478e-09*nk;
            sigma = 3.722483e-07 + 8.793656e-14*mnk + 9.969439e-12*mn + 3.742316e-11*mk + 4.653886e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 42: // node 21
            mu    = 4.909472e-07 + 6.200865e-11*mnk + 1.543290e-10*mn + 2.052205e-09*mk + 3.382308e-09*nk;
            sigma = 3.837337e-07 + 7.486165e-14*mnk + 1.804121e-11*mn + 8.140267e-11*mk + 4.687852e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 43: // node 21
            mu    = 4.890330e-07 + 6.192177e-11*mnk + 1.573438e-10*mn + 2.051455e-09*mk + 3.320061e-09*nk;
            sigma = 3.936516e-07 + 5.312143e-14*mnk + 1.572031e-11*mn + 6.261635e-11*mk + 5.819328e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 44: // node 22
            mu    = 4.878822e-07 + 6.172976e-11*mnk + 1.482621e-10*mn + 2.028372e-09*mk + 3.279886e-09*nk;
            sigma = 4.026244e-07 + 7.197645e-14*mnk + 1.082554e-11*mn + 5.178771e-11*mk + 4.865570e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 45: // node 22
            mu    = 4.936822e-07 + 6.186714e-11*mnk + 1.543108e-10*mn + 2.052711e-09*mk + 3.316411e-09*nk;
            sigma = 3.946072e-07 + 8.449272e-14*mnk + 1.064643e-11*mn + 3.933274e-11*mk + 5.684593e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 46: // node 23
            mu    = 4.941188e-07 + 6.198228e-11*mnk + 1.545497e-10*mn + 2.033903e-09*mk + 3.380732e-09*nk;
            sigma = 4.121263e-07 + 6.894363e-14*mnk + 1.158814e-11*mn + 4.405521e-11*mk + 5.359804e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 47: // node 23
            mu    = 4.975189e-07 + 6.154566e-11*mnk + 1.625192e-10*mn + 2.018039e-09*mk + 3.285374e-09*nk;
            sigma = 4.195640e-07 + 4.712705e-14*mnk + 1.683734e-11*mn + 4.785140e-11*mk + 4.312214e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 48: // node 24
            mu    = 4.890067e-07 + 6.185867e-11*mnk + 1.613742e-10*mn + 2.021834e-09*mk + 3.310428e-09*nk;
            sigma = 3.860074e-07 + 4.992063e-14*mnk + 1.860987e-11*mn + 4.774086e-11*mk + 5.871249e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 49: // node 24
            mu    = 4.926251e-07 + 6.173476e-11*mnk + 1.621909e-10*mn + 2.039362e-09*mk + 3.288933e-09*nk;
            sigma = 4.090749e-07 + 5.865557e-14*mnk + 1.463527e-11*mn + 6.486346e-11*mk + 4.158526e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 50: // node 25
            mu    = 5.224717e-07 + 6.684686e-11*mnk + 9.476546e-11*mn + 2.243565e-09*mk + 3.267669e-09*nk;
            sigma = 4.343758e-07 + 2.947679e-13*mnk + 1.967469e-11*mn + 3.099421e-11*mk + 1.202927e-10*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 51: // node 25
            mu    = 4.759662e-07 + 6.142307e-11*mnk + 1.666604e-10*mn + 1.978105e-09*mk + 3.311607e-09*nk;
            sigma = 3.651041e-07 + 4.321739e-14*mnk + 1.196260e-11*mn + 4.376371e-11*mk + 3.882221e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 52: // node 26
            mu    = 5.071691e-07 + 6.203203e-11*mnk + 1.440150e-10*mn + 2.060903e-09*mk + 3.360838e-09*nk;
            sigma = 4.156681e-07 + 1.026177e-13*mnk + 6.671157e-12*mn + 3.998545e-11*mk + 4.208227e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 53: // node 26
            mu    = 4.945868e-07 + 6.233651e-11*mnk + 1.172611e-10*mn + 2.124213e-09*mk + 3.305920e-09*nk;
            sigma = 3.712657e-07 + 1.648723e-13*mnk + 5.609670e-12*mn + 5.424832e-11*mk + 4.403454e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 54: // node 27
            mu    = 5.115211e-07 + 6.221490e-11*mnk + 1.418988e-10*mn + 2.085160e-09*mk + 3.388985e-09*nk;
            sigma = 4.127484e-07 + 8.150039e-14*mnk + 1.576886e-11*mn + 5.331785e-11*mk + 5.355455e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 55: // node 27
            mu    = 4.846581e-07 + 6.244143e-11*mnk + 1.171356e-10*mn + 2.127831e-09*mk + 3.321703e-09*nk;
            sigma = 3.716838e-07 + 2.154875e-13*mnk + 6.279426e-12*mn + 4.683788e-11*mk + 6.230127e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 58: // node 29
            mu    = 4.789564e-07 + 6.196234e-11*mnk + 1.525791e-10*mn + 2.013158e-09*mk + 3.336648e-09*nk;
            sigma = 3.658987e-07 + 1.291703e-13*mnk + 2.057693e-12*mn + 3.168401e-11*mk + 4.089649e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 59: // node 29
            mu    = 5.014018e-07 + 6.207055e-11*mnk + 1.559168e-10*mn + 2.048223e-09*mk + 3.311984e-09*nk;
            sigma = 4.042239e-07 + 1.007000e-13*mnk + 1.496691e-11*mn + 4.434940e-11*mk + 4.721640e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 60: // node 30
            mu    = 4.864184e-07 + 6.184368e-11*mnk + 1.602625e-10*mn + 2.029841e-09*mk + 3.318098e-09*nk;
            sigma = 3.745791e-07 + 4.860992e-14*mnk + 1.490017e-11*mn + 5.429349e-11*mk + 6.204667e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 61: // node 30
            mu    = 4.863936e-07 + 6.185760e-11*mnk + 1.619071e-10*mn + 1.998977e-09*mk + 3.237839e-09*nk;
            sigma = 3.888527e-07 + 7.220267e-14*mnk + 2.159438e-11*mn + 4.859564e-11*mk + 4.914690e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 62: // node 31
            mu    = 4.888675e-07 + 6.163329e-11*mnk + 1.596314e-10*mn + 2.008430e-09*mk + 3.278759e-09*nk;
            sigma = 3.868797e-07 + 5.425850e-14*mnk + 1.243494e-11*mn + 5.090185e-11*mk + 4.053472e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 63: // node 31
            mu    = 4.858868e-07 + 6.176961e-11*mnk + 1.619049e-10*mn + 2.019454e-09*mk + 3.331812e-09*nk;
            sigma = 3.734420e-07 + 6.591220e-14*mnk + 1.310326e-11*mn + 4.231096e-11*mk + 6.049612e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 64: // node 32
            mu    = 4.881115e-07 + 6.158081e-11*mnk + 1.619842e-10*mn + 1.985753e-09*mk + 3.240081e-09*nk;
            sigma = 3.844051e-07 + 7.794425e-14*mnk + 1.042054e-11*mn + 3.865148e-11*mk + 3.775177e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 65: // node 32
            mu    = 4.913571e-07 + 6.164220e-11*mnk + 1.673913e-10*mn + 1.987885e-09*mk + 3.258608e-09*nk;
            sigma = 3.896617e-07 + 7.888882e-14*mnk + 1.109705e-11*mn + 3.792334e-11*mk + 4.943609e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        default:
            mu    = 4.929201e-07 + 2.053620e-09*mk + 1.449483e-10*mn + 6.215600e-11*mnk + 3.298128e-09*nk;
            sigma = 3.914829e-07 + 4.725869e-11*mk + 1.284986e-11*mn + 1.212951e-13*mnk + 5.112152e-11*nk;
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
