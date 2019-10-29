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
            mu    = 2.564878e-06 + 2.319514e-09*mk + -1.443585e-10*mn + 7.316512e-11*mnk + 3.156805e-09*nk;
            sigma = 3.942041e-07 + 2.363765e-11*mk + 3.785636e-11*mn + 5.748196e-13*mnk + 6.172746e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 3: // node 1
            mu    = 2.607108e-06 + 2.281112e-09*mk + -2.210295e-10*mn + 7.007846e-11*mnk + 2.959197e-09*nk;
            sigma = 5.097946e-07 + 2.101547e-11*mk + 2.172955e-11*mn + 8.158216e-13*mnk + 1.759677e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 4: // node 2
            mu    = 2.631743e-06 + 2.332165e-09*mk + -1.452183e-10*mn + 7.191529e-11*mnk + 3.165252e-09*nk;
            sigma = 3.609668e-07 + 2.844224e-11*mk + 2.055670e-11*mn + 5.172397e-13*mnk + 5.346529e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 5: // node 2
            mu    = 2.591826e-06 + 2.274483e-09*mk + -1.421251e-10*mn + 6.904870e-11*mnk + 3.170111e-09*nk;
            sigma = 4.539582e-07 + 2.802920e-11*mk + 2.416444e-11*mn + 5.359532e-13*mnk + 4.400807e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 6: // node 3
            mu    = 2.569149e-06 + 2.285577e-09*mk + -1.654071e-10*mn + 7.194334e-11*mnk + 3.003229e-09*nk;
            sigma = 3.870118e-07 + 1.395972e-11*mk + 4.113603e-11*mn + 7.291432e-13*mnk + 5.909440e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 7: // node 3
            mu    = 2.468934e-06 + 2.200326e-09*mk + -1.562658e-10*mn + 6.885396e-11*mnk + 3.018826e-09*nk;
            sigma = 4.581137e-07 + 2.291028e-11*mk + 2.815451e-11*mn + 5.805186e-13*mnk + 4.753530e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 8: // node 4
            mu    = 2.496083e-06 + 2.222754e-09*mk + -1.291794e-10*mn + 6.784384e-11*mnk + 3.089735e-09*nk;
            sigma = 3.717400e-07 + 2.483144e-11*mk + 3.171860e-11*mn + 4.992757e-13*mnk + 6.118932e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 9: // node 4
            mu    = 2.603316e-06 + 2.241018e-09*mk + -1.493441e-10*mn + 6.905514e-11*mnk + 3.066078e-09*nk;
            sigma = 4.678765e-07 + 2.846487e-11*mk + 2.669391e-11*mn + 5.851108e-13*mnk + 6.046990e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 10: // node 5
            mu    = 2.526198e-06 + 2.236968e-09*mk + -1.306704e-10*mn + 7.185974e-11*mnk + 3.055954e-09*nk;
            sigma = 3.597642e-07 + 1.203877e-11*mk + 3.700992e-11*mn + 5.663642e-13*mnk + 5.981417e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 11: // node 5
            mu    = 2.463368e-06 + 2.160675e-09*mk + -1.473619e-10*mn + 6.672683e-11*mnk + 3.007531e-09*nk;
            sigma = 4.524880e-07 + 1.589965e-11*mk + 2.730889e-11*mn + 5.007870e-13*mnk + 5.034473e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 12: // node 6
            mu    = 2.564333e-06 + 2.246125e-09*mk + -8.674316e-11*mn + 7.349503e-11*mnk + 3.170152e-09*nk;
            sigma = 4.041332e-07 + 2.280505e-11*mk + 2.960538e-11*mn + 4.201877e-13*mnk + 5.834344e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 13: // node 6
            mu    = 2.387962e-06 + 2.157037e-09*mk + -1.560458e-10*mn + 6.589321e-11*mnk + 2.884688e-09*nk;
            sigma = 4.305571e-07 + 7.299890e-12*mk + 4.218050e-11*mn + 6.788246e-13*mnk + 5.380998e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 14: // node 7
            mu    = 2.565354e-06 + 2.228662e-09*mk + -1.706556e-10*mn + 6.958212e-11*mnk + 2.997618e-09*nk;
            sigma = 3.785604e-07 + 2.561017e-11*mk + 3.539984e-11*mn + 6.201882e-13*mnk + 7.432215e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 15: // node 7
            mu    = 2.497812e-06 + 2.118337e-09*mk + -1.085073e-10*mn + 6.425946e-11*mnk + 3.032245e-09*nk;
            sigma = 4.315405e-07 + 3.317241e-11*mk + 2.564721e-11*mn + 4.535752e-13*mnk + 4.718171e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 16: // node 8
            mu    = 2.505340e-06 + 2.260800e-09*mk + -1.849049e-10*mn + 7.230615e-11*mnk + 3.006606e-09*nk;
            sigma = 3.863429e-07 + 1.308244e-11*mk + 3.975849e-11*mn + 6.198164e-13*mnk + 5.028555e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 17: // node 8
            mu    = 2.494042e-06 + 2.216870e-09*mk + -1.701574e-10*mn + 6.780364e-11*mnk + 2.912890e-09*nk;
            sigma = 4.848342e-07 + 3.309100e-11*mk + 3.491150e-11*mn + 6.550628e-13*mnk + 5.284657e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 18: // node 9
            mu    = 2.619425e-06 + 2.280683e-09*mk + -1.867096e-10*mn + 6.719161e-11*mnk + 2.934858e-09*nk;
            sigma = 4.097842e-07 + 2.525403e-11*mk + 1.706905e-11*mn + 7.264568e-13*mnk + 2.422280e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 19: // node 9
            mu    = 2.562125e-06 + 2.240709e-09*mk + -8.700213e-11*mn + 6.730293e-11*mnk + 3.145183e-09*nk;
            sigma = 4.537638e-07 + 1.992627e-11*mk + 2.886638e-11*mn + 5.376262e-13*mnk + 6.196557e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 20: // node 10
            mu    = 2.612733e-06 + 2.231419e-09*mk + -1.224582e-10*mn + 6.766796e-11*mnk + 3.079652e-09*nk;
            sigma = 4.086542e-07 + 3.214410e-11*mk + 2.787780e-11*mn + 4.993162e-13*mnk + 6.739603e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 21: // node 10
            mu    = 2.508236e-06 + 2.229072e-09*mk + -1.727272e-10*mn + 6.876203e-11*mnk + 3.038093e-09*nk;
            sigma = 4.687469e-07 + 2.542548e-11*mk + 3.082872e-11*mn + 5.943197e-13*mnk + 4.673214e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 22: // node 11
            mu    = 2.518250e-06 + 2.201216e-09*mk + -1.816969e-10*mn + 6.735811e-11*mnk + 2.975266e-09*nk;
            sigma = 3.398434e-07 + 2.414182e-11*mk + 2.242584e-11*mn + 6.670968e-13*mnk + 4.350173e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 23: // node 11
            mu    = 2.502448e-06 + 2.294001e-09*mk + -1.726822e-10*mn + 7.224332e-11*mnk + 3.042957e-09*nk;
            sigma = 4.796716e-07 + 1.415755e-11*mk + 4.022496e-11*mn + 7.077318e-13*mnk + 6.776312e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 24: // node 12
            mu    = 2.526698e-06 + 2.187974e-09*mk + -1.432686e-10*mn + 6.778821e-11*mnk + 3.022096e-09*nk;
            sigma = 3.537503e-07 + 2.242495e-11*mk + 2.398628e-11*mn + 5.460593e-13*mnk + 4.966691e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 25: // node 12
            mu    = 2.466976e-06 + 2.185173e-09*mk + -1.567443e-10*mn + 6.688891e-11*mnk + 3.037509e-09*nk;
            sigma = 4.068829e-07 + 2.813848e-11*mk + 2.539939e-11*mn + 5.615360e-13*mnk + 5.165258e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 26: // node 13
            mu    = 2.752476e-06 + 2.301487e-09*mk + -4.006373e-12*mn + 7.892263e-11*mnk + 3.417846e-09*nk;
            sigma = 4.675315e-07 + -3.865964e-11*mk + 3.607889e-11*mn + 3.440300e-12*mnk + -8.665233e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 27: // node 13
            mu    = 2.528562e-06 + 2.289597e-09*mk + -1.038439e-10*mn + 7.255856e-11*mnk + 3.436770e-09*nk;
            sigma = 4.747249e-07 + 4.397357e-11*mk + 5.762800e-11*mn + 1.424138e-12*mnk + 9.530768e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 28: // node 14
            mu    = 2.792097e-06 + 2.264896e-09*mk + -2.524748e-11*mn + 7.793373e-11*mnk + 3.257760e-09*nk;
            sigma = 4.084944e-07 + -1.582118e-10*mk + 1.274614e-10*mn + 3.270377e-12*mnk + -9.962597e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 29: // node 14
            mu    = 2.549757e-06 + 2.275910e-09*mk + -2.087559e-10*mn + 7.334130e-11*mnk + 3.061918e-09*nk;
            sigma = 4.595648e-07 + 2.155502e-11*mk + 7.450070e-11*mn + 1.218624e-12*mnk + 5.115943e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 30: // node 15
            mu    = 2.792736e-06 + 2.588937e-09*mk + -1.316311e-10*mn + 8.477900e-11*mnk + 3.403027e-09*nk;
            sigma = 3.937171e-07 + 3.429498e-11*mk + 1.257945e-11*mn + 9.658743e-13*mnk + 5.900742e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 31: // node 15
            mu    = 2.576031e-06 + 2.133646e-09*mk + -4.762863e-11*mn + 7.304379e-11*mnk + 2.938259e-09*nk;
            sigma = 5.107940e-07 + 8.495101e-11*mk + 6.578668e-11*mn + 2.384839e-12*mnk + 4.711649e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 32: // node 16
            mu    = 2.852108e-06 + 2.411571e-09*mk + -6.695578e-11*mn + 8.106829e-11*mnk + 3.322094e-09*nk;
            sigma = 4.456235e-07 + -7.212156e-11*mk + 1.028364e-10*mn + 1.952141e-12*mnk + -5.671276e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 33: // node 16
            mu    = 2.603281e-06 + 2.270566e-09*mk + -3.214576e-10*mn + 7.185298e-11*mnk + 2.852341e-09*nk;
            sigma = 5.568266e-07 + 3.652544e-11*mk + 2.634672e-11*mn + 1.088705e-12*mnk + 3.146780e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 34: // node 17
            mu    = 2.680639e-06 + 2.312999e-09*mk + -1.640092e-10*mn + 7.319093e-11*mnk + 3.100469e-09*nk;
            sigma = 4.125988e-07 + 1.612645e-11*mk + 3.384418e-11*mn + 5.894599e-13*mnk + 5.968139e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 35: // node 17
            mu    = 2.491219e-06 + 2.250349e-09*mk + -1.845086e-10*mn + 6.998302e-11*mnk + 2.982478e-09*nk;
            sigma = 4.875519e-07 + 1.679117e-11*mk + 3.498314e-11*mn + 6.972107e-13*mnk + 4.686682e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 36: // node 18
            mu    = 2.491177e-06 + 2.259291e-09*mk + -3.843287e-10*mn + 7.313197e-11*mnk + 2.956694e-09*nk;
            sigma = 3.538768e-07 + 1.450910e-11*mk + 5.227662e-11*mn + 4.682852e-13*mnk + 8.924465e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 37: // node 18
            mu    = 2.516083e-06 + 2.167992e-09*mk + -4.634177e-10*mn + 6.733235e-11*mnk + 2.726376e-09*nk;
            sigma = 4.674064e-07 + 3.158981e-11*mk + 2.947318e-11*mn + 5.658375e-13*mnk + 4.190402e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 38: // node 19
            mu    = 2.518736e-06 + 2.282711e-09*mk + -2.102911e-10*mn + 6.900796e-11*mnk + 2.975085e-09*nk;
            sigma = 3.615002e-07 + 2.514365e-11*mk + 2.613082e-11*mn + 7.207902e-13*mnk + 1.978688e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 39: // node 19
            mu    = 2.666372e-06 + 2.242356e-09*mk + -1.440802e-10*mn + 6.871071e-11*mnk + 3.134598e-09*nk;
            sigma = 5.064346e-07 + 2.828969e-11*mk + 3.126015e-11*mn + 4.958007e-13*mnk + 4.269200e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 40: // node 20
            mu    = 2.538087e-06 + 2.246728e-09*mk + -1.593620e-10*mn + 7.073904e-11*mnk + 3.043830e-09*nk;
            sigma = 3.628876e-07 + 2.083225e-11*mk + 2.161969e-11*mn + 6.150084e-13*mnk + 3.240651e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 41: // node 20
            mu    = 2.591566e-06 + 2.189672e-09*mk + -1.915895e-10*mn + 6.763355e-11*mnk + 2.975638e-09*nk;
            sigma = 4.662968e-07 + 2.178124e-11*mk + 3.549896e-11*mn + 5.610112e-13*mnk + 4.904525e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 42: // node 21
            mu    = 2.633302e-06 + 2.269042e-09*mk + -1.198376e-10*mn + 7.006368e-11*mnk + 3.162329e-09*nk;
            sigma = 3.839649e-07 + 2.826214e-11*mk + 2.939568e-11*mn + 5.213940e-13*mnk + 6.127316e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 43: // node 21
            mu    = 2.532420e-06 + 2.269744e-09*mk + -1.741651e-10*mn + 6.916830e-11*mnk + 3.022265e-09*nk;
            sigma = 4.599424e-07 + 1.808170e-11*mk + 2.436743e-11*mn + 7.209242e-13*mnk + 3.694896e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 44: // node 22
            mu    = 2.519059e-06 + 2.211499e-09*mk + -1.612096e-10*mn + 6.803016e-11*mnk + 3.006770e-09*nk;
            sigma = 3.471874e-07 + 2.361603e-11*mk + 2.569706e-11*mn + 6.219196e-13*mnk + 4.847087e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 45: // node 22
            mu    = 2.600361e-06 + 2.244985e-09*mk + -1.689315e-10*mn + 6.963974e-11*mnk + 3.045402e-09*nk;
            sigma = 4.935254e-07 + 2.965409e-11*mk + 3.240179e-11*mn + 6.393423e-13*mnk + 4.396316e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 46: // node 23
            mu    = 2.466819e-06 + 2.222876e-09*mk + -1.540193e-10*mn + 6.935710e-11*mnk + 3.046249e-09*nk;
            sigma = 3.638180e-07 + 2.674582e-11*mk + 2.792014e-11*mn + 5.831043e-13*mnk + 3.527729e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 47: // node 23
            mu    = 2.575813e-06 + 2.174068e-09*mk + -1.834576e-10*mn + 6.643075e-11*mnk + 2.936201e-09*nk;
            sigma = 4.863532e-07 + 2.397628e-11*mk + 3.436811e-11*mn + 5.618521e-13*mnk + 5.871765e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 48: // node 24
            mu    = 2.433671e-06 + 2.171235e-09*mk + -1.505702e-10*mn + 6.879598e-11*mnk + 3.003827e-09*nk;
            sigma = 3.418350e-07 + 2.161812e-11*mk + 2.849344e-11*mn + 5.995886e-13*mnk + 4.621013e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 49: // node 24
            mu    = 2.544671e-06 + 2.129685e-09*mk + -1.295043e-10*mn + 6.715201e-11*mnk + 3.046265e-09*nk;
            sigma = 4.973893e-07 + 1.738315e-11*mk + 2.723886e-11*mn + 5.181305e-13*mnk + 6.791229e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 50: // node 25
            mu    = 2.597486e-06 + 2.311296e-09*mk + -1.594176e-10*mn + 7.613608e-11*mnk + 3.055496e-09*nk;
            sigma = 4.012121e-07 + 2.657218e-11*mk + 4.855944e-11*mn + 5.212174e-13*mnk + 9.152750e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 51: // node 25
            mu    = 2.395854e-06 + 2.122906e-09*mk + -2.093406e-10*mn + 6.532887e-11*mnk + 2.840593e-09*nk;
            sigma = 4.560458e-07 + 1.976670e-11*mk + 2.139928e-11*mn + 7.499219e-13*mnk + 1.558464e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 52: // node 26
            mu    = 2.599465e-06 + 2.273816e-09*mk + -2.337139e-10*mn + 6.857245e-11*mnk + 2.944236e-09*nk;
            sigma = 3.999103e-07 + 2.696201e-11*mk + 3.027443e-11*mn + 7.577428e-13*mnk + 3.455684e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 53: // node 26
            mu    = 2.531479e-06 + 2.284333e-09*mk + -1.502799e-10*mn + 7.089453e-11*mnk + 3.103020e-09*nk;
            sigma = 4.511930e-07 + 2.363511e-11*mk + 2.862170e-11*mn + 5.810298e-13*mnk + 6.241887e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 54: // node 27
            mu    = 2.574156e-06 + 2.270580e-09*mk + -2.094846e-10*mn + 7.119769e-11*mnk + 2.933743e-09*nk;
            sigma = 3.712821e-07 + 2.844750e-11*mk + 4.248123e-11*mn + 7.365222e-13*mnk + 6.631982e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 55: // node 27
            mu    = 2.578608e-06 + 2.281135e-09*mk + -2.586911e-10*mn + 7.160017e-11*mnk + 2.831666e-09*nk;
            sigma = 4.614923e-07 + -3.284890e-12*mk + 4.915038e-11*mn + 7.752040e-13*mnk + 5.386469e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 56: // node 28
            mu    = 2.636816e-06 + 2.303201e-09*mk + -1.020302e-10*mn + 7.245640e-11*mnk + 3.208493e-09*nk;
            sigma = 3.658511e-07 + 2.176596e-11*mk + 2.704087e-11*mn + 4.989467e-13*mnk + 6.448892e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 57: // node 28
            mu    = 2.532406e-06 + 2.197884e-09*mk + -1.705121e-10*mn + 6.696472e-11*mnk + 3.011324e-09*nk;
            sigma = 4.841870e-07 + 1.784345e-11*mk + 2.942653e-11*mn + 5.892037e-13*mnk + 4.912946e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 58: // node 29
            mu    = 2.559524e-06 + 2.229634e-09*mk + -1.120649e-10*mn + 6.839814e-11*mnk + 3.149648e-09*nk;
            sigma = 3.768358e-07 + 2.985499e-11*mk + 3.242007e-11*mn + 4.690949e-13*mnk + 7.452242e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 59: // node 29
            mu    = 2.517903e-06 + 2.232818e-09*mk + -1.893774e-10*mn + 6.805100e-11*mnk + 2.973032e-09*nk;
            sigma = 4.396113e-07 + 1.478794e-11*mk + 2.738337e-11*mn + 6.422637e-13*mnk + 3.739380e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 60: // node 30
            mu    = 2.540733e-06 + 2.198599e-09*mk + -1.581561e-10*mn + 6.693421e-11*mnk + 3.048898e-09*nk;
            sigma = 3.537730e-07 + 2.459842e-11*mk + 2.203601e-11*mn + 5.090605e-13*mnk + 5.364999e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 61: // node 30
            mu    = 2.619628e-06 + 2.194255e-09*mk + -1.489253e-10*mn + 6.619825e-11*mnk + 3.063774e-09*nk;
            sigma = 4.744879e-07 + 3.090047e-11*mk + 2.730797e-11*mn + 4.997365e-13*mnk + 7.907536e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 62: // node 31
            mu    = 2.426604e-06 + 2.154489e-09*mk + -1.334948e-10*mn + 6.675336e-11*mnk + 3.003345e-09*nk;
            sigma = 3.593449e-07 + 1.247247e-11*mk + 3.470295e-11*mn + 4.954417e-13*mnk + 6.091315e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 63: // node 31
            mu    = 2.499378e-06 + 2.174820e-09*mk + -1.462657e-10*mn + 6.861519e-11*mnk + 2.966180e-09*nk;
            sigma = 4.305984e-07 + 2.484167e-11*mk + 2.635688e-11*mn + 5.208267e-13*mnk + 4.538180e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 64: // node 32
            mu    = 2.587604e-06 + 2.184720e-09*mk + -6.170374e-11*mn + 6.601543e-11*mnk + 3.117080e-09*nk;
            sigma = 4.028614e-07 + 2.036909e-11*mk + 2.054382e-11*mn + 4.840251e-13*mnk + 5.513812e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        case 65: // node 32
            mu    = 2.516758e-06 + 2.210315e-09*mk + -1.138195e-10*mn + 6.670692e-11*mnk + 3.005317e-09*nk;
            sigma = 4.901299e-07 + 1.454003e-11*mk + 2.927636e-11*mn + 5.137512e-13*mnk + 6.290253e-11*nk;
            return mu + random_halfnormal_shifted(0, sigma);
        default:
            mu    = 2.559622e-06 + 2.241239e-09*mk + -1.600211e-10*mn + 7.002694e-11*mnk + 3.048170e-09*nk;
            sigma = 4.262757e-07 + 1.869818e-11*mk + 3.461998e-11*mn + 7.618981e-13*mnk + 4.676988e-11*nk;
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
