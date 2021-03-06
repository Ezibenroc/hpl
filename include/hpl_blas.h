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
 */ 
#ifndef HPL_BLAS_H
#define HPL_BLAS_H
/*
 * ---------------------------------------------------------------------
 * Include files
 * ---------------------------------------------------------------------
 */
#include "hpl_misc.h"
/*
 * ---------------------------------------------------------------------
 * typedef definitions
 * ---------------------------------------------------------------------
 */
enum HPL_ORDER
{  HplRowMajor = 101,  HplColumnMajor  = 102 };
enum HPL_TRANS
{  HplNoTrans  = 111,  HplTrans        = 112,  HplConjTrans    = 113 };
enum HPL_UPLO
{  HplUpper    = 121,  HplLower        = 122 };
enum HPL_DIAG
{  HplNonUnit  = 131,  HplUnit         = 132 };
enum HPL_SIDE
{  HplLeft     = 141,  HplRight        = 142 }; 

#ifdef HPL_CALL_CBLAS
/*
 * ---------------------------------------------------------------------
 * The C interface of the BLAS is available ...
 * ---------------------------------------------------------------------
 * #define macro constants
 * ---------------------------------------------------------------------
 */
#define    CBLAS_INDEX         int
 
#define    CBLAS_ORDER         HPL_ORDER
#define    CblasRowMajor       HplRowMajor
#define    CblasColMajor       HplColMajor
 
#define    CBLAS_TRANSPOSE     HPL_TRANS
#define    CblasNoTrans        HplNoTrans
#define    CblasTrans          HplTrans
#define    CblasConjTrans      HplConjTrans
 
#define    CBLAS_UPLO          HPL_UPLO
#define    CblasUpper          HplUpper
#define    CblasLower          HplLower
 
#define    CBLAS_DIAG          HPL_DIAG
#define    CblasNonUnit        HplNonUnit
#define    CblasUnit           HplUnit
 
#define    CBLAS_SIDE          HPL_SIDE
#define    CblasLeft           HplLeft
#define    CblasRight          HplRight
/*
 * ---------------------------------------------------------------------
 * CBLAS Function prototypes
 * ---------------------------------------------------------------------
 */
CBLAS_INDEX       cblas_idamax
STDC_ARGS(
(  const int,       const double *,  const int ) );
void              cblas_dswap
STDC_ARGS(
(  const int,       double *,        const int,       double *,
   const int ) );
void              cblas_dcopy
STDC_ARGS(
(  const int,       const double *,  const int,       double *,
   const int ) );
void              cblas_daxpy
STDC_ARGS(
(  const int,       const double,    const double *,  const int,
   double *,        const int ) );
void              cblas_dscal
STDC_ARGS(
(  const int,       const double,    double *,        const int ) );

void              cblas_dgemv
STDC_ARGS(
(  const enum CBLAS_ORDER,           const enum CBLAS_TRANSPOSE,
   const int,       const int,       const double,    const double *,
   const int,       const double *,  const int,       const double,
   double *,        const int ) );

void              cblas_dger
STDC_ARGS(
(  const enum CBLAS_ORDER,           const int,       const int,
   const double,    const double *,  const int,       const double *,
   const int,       double *,        const int ) );
void              cblas_dtrsv
STDC_ARGS(
(  const enum CBLAS_ORDER,           const enum CBLAS_UPLO,
   const enum CBLAS_TRANSPOSE,       const enum CBLAS_DIAG,
   const int,       const double *,  const int,       double *,
   const int ) );

void              cblas_dgemm
STDC_ARGS(
(  const enum CBLAS_ORDER,           const enum CBLAS_TRANSPOSE,
   const enum CBLAS_TRANSPOSE,       const int,       const int,
   const int,       const double,    const double *,  const int,
   const double *,  const int,       const double,    double *,
   const int ) );
void              cblas_dtrsm
STDC_ARGS(
(  const enum CBLAS_ORDER,           const enum CBLAS_SIDE,
   const enum CBLAS_UPLO,            const enum CBLAS_TRANSPOSE,
   const enum CBLAS_DIAG,            const int,       const int,
   const double,    const double *,  const int,       double *,
   const int ) );
/*
 * ---------------------------------------------------------------------
 * HPL C BLAS macro definition
 * ---------------------------------------------------------------------
 */
#define    HPL_dcopy           cblas_dcopy

#if SMPI_OPTIMIZATION_LEVEL >= 2
#define    HPL_dswap(...)      {}
#define    HPL_dgemv(...)      {}
#define    HPL_daxpy(...)      {}
#define    HPL_dscal(...)      {}
#define    HPL_idamax(N, X, incX) (rand()%N)
#define    HPL_dtrsv(...)      {}
#define    HPL_dger(...)       {}
#pragma message "[SMPI] Using no-op for the cheapest BLAS functions."
#else
#define    HPL_dswap           cblas_dswap
#define    HPL_dgemv           cblas_dgemv
#define    HPL_daxpy           cblas_daxpy
#define    HPL_dscal           cblas_dscal
#define    HPL_idamax          cblas_idamax
#define    HPL_dtrsv           cblas_dtrsv
#define    HPL_dger            cblas_dger
#pragma message "[SMPI] Using cblas for the cheapest BLAS functions."
#endif

// From http://stackoverflow.com/a/10227059/4110059
#define VALUE_TO_STRING(x) #x
#define VALUE(x) VALUE_TO_STRING(x)
#define VAR_NAME_VALUE(var) #var "="  VALUE(var)

FILE *get_measure_file();
typedef unsigned long long timestamp_t;
timestamp_t get_timestamp(void);
void record_measure(const char *file, int line, const char *function, timestamp_t start, timestamp_t duration, int n_args, int *args);
double random_halfnormal_shifted(double exp, double std);
void smpi_execute_normal(double mu, double sigma);
void smpi_execute_normal_size(double mu, double sigma, double size);
void smpi_execute_dgemm(int M, int N, int K);

static double get_param(const char *name) {
    char *val_str = getenv(name);
    if(!val_str) {
        fprintf(stderr, "Environment variable '%s' not defined.\n", name);
        exit(1);
    }
    return atof(val_str);
}

#if SMPI_OPTIMIZATION_LEVEL >= 1
#define  HPL_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)  ({\
    timestamp_t start = get_timestamp();\
    smpi_execute_dgemm(M, N, K);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dgemm", start, duration, 3, (int []){M, N, K});\
})
#define HPL_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) ({\
    timestamp_t start = get_timestamp();\
    double expected_time;\
    double size=-1;\
    if((Side) == HplLeft) {\
        size = ((double)(M))*((double)(M))*((double)(N));\
    } else {\
        size = ((double)(M))*((double)(N))*((double)(N));\
    }\
    smpi_execute_normal_size(5.445540e-11, 3.093410e-12, size);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dtrsm", start, duration, 2, (int []){M, N});\
})
#else // SMPI_OPTIMIZATION_LEVEL
#define  HPL_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)  ({\
    timestamp_t start = get_timestamp();\
    cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dgemm", start, duration, 3, (int []){M, N, K});\
})
#define HPL_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) ({\
    timestamp_t start = get_timestamp();\
    cblas_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dtrsm", start, duration, 2, (int []){M, N});\
})
#endif // SMPI_OPTIMIZATION_LEVEL

#if SMPI_OPTIMIZATION_LEVEL >= 2
#define HPL_dswap(N, X, incX, Y, incY) ({\
    timestamp_t start = get_timestamp();\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dswap", start, duration, 1, (int []){N});\
})
// do not remove HPL_dcopy, otherwise you get a deadlock
#define HPL_dcopy(N, X, incX, Y, incY) ({\
    timestamp_t start = get_timestamp();\
    cblas_dcopy(N, X, incX, Y, incY);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dcopy", start, duration, 1, (int []){N});\
})
#define HPL_daxpy(N, alpha, X, incX, Y, incY) ({\
    timestamp_t start = get_timestamp();\
    smpi_execute_normal(6.680085e-07, 9.265829e-07);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "daxpy", start, duration, 1, (int []){N});\
})
#define HPL_dscal(N, alpha, X, incX) ({\
    timestamp_t start = get_timestamp();\
    smpi_execute_normal_size(3.798686e-10, 1.490121e-10, (double)N);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dscal", start, duration, 1, (int []){N});\
})
#define HPL_idamax(N, X, incX) ({\
    timestamp_t start = get_timestamp();\
    int result = rand()%N;\
    smpi_execute_normal_size(6.052262e-10, 2.380847e-10, (double)N);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "idamax", start, duration, 1, (int []){N});\
    result;\
})
#define HPL_dgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY) ({\
    timestamp_t start = get_timestamp();\
    smpi_execute_normal_size(1.278804e-09, 1.645759e-11, ((double)M)*N);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dgemv", start, duration, 2, (int []){M, N});\
})
#define HPL_dtrsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX) ({\
    timestamp_t start = get_timestamp();\
    smpi_execute_normal(2.438651e-05, 4.452153e-06);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dtrsv", start, duration, 1, (int []){N});\
})
#define HPL_dger(layout, M, N, alpha, X, incX, Y, incY, A, ldA) ({\
    timestamp_t start = get_timestamp();\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dger", start, duration, 2, (int []){M, N});\
})
#else // SMPI_OPTIMIZATION_LEVEL
#define HPL_dswap(N, X, incX, Y, incY) ({\
    timestamp_t start = get_timestamp();\
    cblas_dswap(N, X, incX, Y, incY);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dswap", start, duration, 1, (int []){N});\
})
#define HPL_dcopy(N, X, incX, Y, incY) ({\
    timestamp_t start = get_timestamp();\
    cblas_dcopy(N, X, incX, Y, incY);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dcopy", start, duration, 1, (int []){N});\
})
#define HPL_daxpy(N, alpha, X, incX, Y, incY) ({\
    timestamp_t start = get_timestamp();\
    cblas_daxpy(N, alpha, X, incX, Y, incY);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "daxpy", start, duration, 1, (int []){N});\
})
#define HPL_dscal(N, alpha, X, incX) ({\
    timestamp_t start = get_timestamp();\
    cblas_dscal(N, alpha, X, incX);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dscal", start, duration, 1, (int []){N});\
})
#define HPL_idamax(N, X, incX) ({\
    timestamp_t start = get_timestamp();\
    int result = cblas_idamax(N, X, incX);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "idamax", start, duration, 1, (int []){N});\
    result;\
})
#define HPL_dgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY) ({\
    timestamp_t start = get_timestamp();\
    cblas_dgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dgemv", start, duration, 2, (int []){M, N});\
})
#define HPL_dtrsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX) ({\
    timestamp_t start = get_timestamp();\
    cblas_dtrsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dtrsv", start, duration, 1, (int []){N});\
})
#define HPL_dger(layout, M, N, alpha, X, incX, Y, incY, A, ldA) ({\
    timestamp_t start = get_timestamp();\
    cblas_dger(layout, M, N, alpha, X, incX, Y, incY, A, ldA);\
    timestamp_t duration = get_timestamp() - start;\
    record_measure(__FILE__, __LINE__, "dger", start, duration, 2, (int []){M, N});\
})
#endif // SMPI_OPTIMIZATION_LEVEL

#endif

#ifdef HPL_CALL_FBLAS
/*
 * ---------------------------------------------------------------------
 * Use the Fortran 77 interface of the BLAS ...
 * ---------------------------------------------------------------------
 * Defaults: Add_, F77_INTEGER=int, StringSunStyle
 * ---------------------------------------------------------------------
 */
#ifndef NoChange
#ifndef UpCase
#ifndef Add__
#ifndef Add_

#define Add_

#endif
#endif
#endif
#endif

#ifndef F77_INTEGER
#define    F77_INTEGER         int
#else
#define    HPL_USE_F77_INTEGER_DEF
#endif

#ifndef StringCrayStyle
#ifndef StringStructVal
#ifndef StringStructPtr
#ifndef StringSunStyle

#define StringSunStyle

#endif
#endif
#endif
#endif
/*
 * ---------------------------------------------------------------------
 * Fortran 77 <-> C interface
 * ---------------------------------------------------------------------
 *
 * These macros identifies how Fortran routines will be called.
 *
 * Add_     : the Fortran compiler expects the name of C functions to be
 * in all lower case and to have an underscore postfixed it (Suns, Intel
 * compilers expect this).
 *
 * NoChange : the Fortran compiler expects the name of C functions to be
 * in all lower case (IBM RS6K compilers do this).
 *
 * UpCase   : the Fortran compiler expects the name of C functions to be
 * in all upcase. (Cray compilers expect this).
 *
 * Add__    : the Fortran compiler in use is f2c, a Fortran to C conver-
 * ter.
 */
#ifdef NoChange
/*
 * These defines  set  up  the  naming scheme required to have a FORTRAN
 * routine called by a C routine with the following  FORTRAN to C inter-
 * face:
 *
 *          FORTRAN DECLARATION            C CALL
 *          SUBROUTINE DGEMM(...)          dgemm(...)
 */
#define    F77dswap               dswap
#define    F77dscal               dscal
#define    F77dcopy               dcopy
#define    F77daxpy               daxpy
#define    F77idamax              idamax

#define    F77dgemv               dgemv
#define    F77dtrsv               dtrsv
#define    F77dger                dger

#define    F77dgemm               dgemm
#define    F77dtrsm               dtrsm

#endif

#ifdef UpCase
/*
 * These defines  set  up  the  naming scheme required to have a FORTRAN
 * routine called by a C routine with the following  FORTRAN to C inter-
 * face:
 *
 *          FORTRAN DECLARATION            C CALL
 *          SUBROUTINE DGEMM(...)          DGEMM(...)
 */
#ifdef CRAY_BLAS
                                                                                
#define    F77dswap               SSWAP
#define    F77dscal               SSCAL
#define    F77dcopy               SCOPY
#define    F77daxpy               SAXPY
#define    F77idamax              ISAMAX
                                                                                
#define    F77dgemv               SGEMV
#define    F77dtrsv               STRSV
#define    F77dger                SGER
                                                                                
#define    F77dgemm               SGEMM
#define    F77dtrsm               STRSM
                                                                                
#else

#define    F77dswap               DSWAP
#define    F77dscal               DSCAL
#define    F77dcopy               DCOPY
#define    F77daxpy               DAXPY
#define    F77idamax              IDAMAX

#define    F77dgemv               DGEMV
#define    F77dtrsv               DTRSV
#define    F77dger                DGER

#define    F77dgemm               DGEMM
#define    F77dtrsm               DTRSM

#endif

#endif

#ifdef Add_
/*
 * These defines  set  up  the  naming scheme required to have a FORTRAN
 * routine called by a C routine  with the following  FORTRAN to C inter-
 * face:
 *
 *          FORTRAN DECLARATION            C CALL
 *          SUBROUTINE DGEMM(...)          dgemm_(...)
 */
#define    F77dswap               dswap_
#define    F77dscal               dscal_
#define    F77dcopy               dcopy_
#define    F77daxpy               daxpy_
#define    F77idamax              idamax_

#define    F77dgemv               dgemv_
#define    F77dtrsv               dtrsv_
#define    F77dger                dger_

#define    F77dgemm               dgemm_
#define    F77dtrsm               dtrsm_

#endif

#ifdef Add__
/*
 * These defines  set  up  the  naming scheme required to have a FORTRAN
 * routine called by a C routine  with the following  FORTRAN to C inter-
 * face:
 *
 *          FORTRAN DECLARATION            C CALL
 *          SUBROUTINE DGEMM(...)          dgemm_(...)
 */
#define    F77dswap               dswap_
#define    F77dscal               dscal_
#define    F77dcopy               dcopy_
#define    F77daxpy               daxpy_
#define    F77idamax              idamax_
 
#define    F77dgemv               dgemv_
#define    F77dtrsv               dtrsv_
#define    F77dger                dger_
 
#define    F77dgemm               dgemm_
#define    F77dtrsm               dtrsm_
 
#endif
/*
 * ---------------------------------------------------------------------
 * Typedef definitions and conversion utilities
 * ---------------------------------------------------------------------
 */
#ifdef StringCrayStyle

#include <fortran.h>
                      /* Type of character argument in a FORTRAN call */
#define    F77_CHAR            _fcd
                                    /* Character conversion utilities */
#define    HPL_F2C_CHAR(c)     (*(_fcdtocp(c) ))
#define    HPL_C2F_CHAR(c)     (_cptofcd(&(c), 1))

#define    F77_CHAR_DECL       F77_CHAR          /* input CHARACTER*1 */

#endif
/* ------------------------------------------------------------------ */
#ifdef StringStructVal
                      /* Type of character argument in a FORTRAN call */
typedef struct { char *cp; F77_INTEGER len; } F77_CHAR;
                                    /* Character conversion utilities */
#define    HPL_F2C_CHAR(c)     (*(c.cp))

#define    F77_CHAR_DECL       F77_CHAR          /* input CHARACTER*1 */

#endif
/* ------------------------------------------------------------------ */
#ifdef StringStructPtr
                      /* Type of character argument in a FORTRAN call */
typedef struct { char *cp; F77_INTEGER len; } F77_CHAR;
                                    /* Character conversion utilities */
#define    HPL_F2C_CHAR(c)     (*(c->cp))

#define    F77_CHAR_DECL       F77_CHAR *        /* input CHARACTER*1 */

#endif
/* ------------------------------------------------------------------ */
#ifdef StringSunStyle
                      /* Type of character argument in a FORTRAN call */
#define    F77_CHAR            char *
                                    /* Character conversion utilities */
#define    HPL_F2C_CHAR(c)     (*(c))
#define    HPL_C2F_CHAR(c)     (&(c))

#define    F77_CHAR_DECL       F77_CHAR          /* input CHARACTER*1 */
#define    F77_1_CHAR          , F77_INTEGER
#define    F77_2_CHAR          F77_1_CHAR F77_1_CHAR
#define    F77_3_CHAR          F77_2_CHAR F77_1_CHAR
#define    F77_4_CHAR          F77_3_CHAR F77_1_CHAR

#endif
/* ------------------------------------------------------------------ */

#ifndef F77_1_CHAR
#define    F77_1_CHAR
#define    F77_2_CHAR
#define    F77_3_CHAR
#define    F77_4_CHAR
#endif

#define    F77_INT_DECL        const F77_INTEGER *   /* input integer */
#define    F77_SIN_DECL        const double *         /* input scalar */
#define    F77_VIN_DECL        const double *         /* input vector */
#define    F77_VINOUT_DECL     double *        /* input/output matrix */
#define    F77_MIN_DECL        const double *         /* input matrix */
#define    F77_MINOUT_DECL     double *        /* input/output matrix */
 
#ifdef CRAY_PVP_ENV                      /* Type of FORTRAN functions */
#define    F77_VOID_FUN        extern fortran void      /* subroutine */
#define    F77_INT_FUN         extern fortran int /* integer function */
#else
#define    F77_VOID_FUN        extern void              /* subroutine */
#define    F77_INT_FUN         extern int         /* integer function */
#endif
/*
 * ---------------------------------------------------------------------
 * Fortran 77 BLAS function prototypes
 * ---------------------------------------------------------------------
 */
F77_VOID_FUN    F77dswap
STDC_ARGS(
(  F77_INT_DECL,    F77_VINOUT_DECL, F77_INT_DECL,    F77_VINOUT_DECL,
   F77_INT_DECL ) );
F77_VOID_FUN    F77dscal
STDC_ARGS(
(  F77_INT_DECL,    F77_SIN_DECL,    F77_VINOUT_DECL, F77_INT_DECL ) );
F77_VOID_FUN    F77dcopy
STDC_ARGS(
(  F77_INT_DECL,    F77_VIN_DECL,    F77_INT_DECL,    F77_VINOUT_DECL,
   F77_INT_DECL ) );
F77_VOID_FUN    F77daxpy
STDC_ARGS(
(  F77_INT_DECL,    F77_SIN_DECL,    F77_VIN_DECL,    F77_INT_DECL,
   F77_VINOUT_DECL, F77_INT_DECL ) );
F77_INT_FUN     F77idamax
STDC_ARGS(
(  F77_INT_DECL,    F77_VIN_DECL,    F77_INT_DECL ) );

F77_VOID_FUN    F77dgemv
STDC_ARGS(
(  F77_CHAR_DECL,   F77_INT_DECL,    F77_INT_DECL,    F77_SIN_DECL,
   F77_MIN_DECL,    F77_INT_DECL,    F77_VIN_DECL,    F77_INT_DECL,
   F77_SIN_DECL,    F77_VINOUT_DECL, F77_INT_DECL     F77_1_CHAR ) );
F77_VOID_FUN    F77dger
STDC_ARGS(
(  F77_INT_DECL,    F77_INT_DECL,    F77_SIN_DECL,    F77_VIN_DECL,
   F77_INT_DECL,    F77_VIN_DECL,    F77_INT_DECL,    F77_MINOUT_DECL,
   F77_INT_DECL ) );
F77_VOID_FUN    F77dtrsv
STDC_ARGS(
(  F77_CHAR_DECL,   F77_CHAR_DECL,   F77_CHAR_DECL,   F77_INT_DECL,
   F77_MIN_DECL,    F77_INT_DECL,    F77_VINOUT_DECL, F77_INT_DECL
   F77_3_CHAR ) );

F77_VOID_FUN    F77dgemm
STDC_ARGS(
(  F77_CHAR_DECL,   F77_CHAR_DECL,   F77_INT_DECL,    F77_INT_DECL,
   F77_INT_DECL,    F77_SIN_DECL,    F77_MIN_DECL,    F77_INT_DECL,
   F77_MIN_DECL,    F77_INT_DECL,    F77_SIN_DECL,    F77_MINOUT_DECL,
   F77_INT_DECL     F77_2_CHAR ) );
F77_VOID_FUN    F77dtrsm
STDC_ARGS(
(  F77_CHAR_DECL,   F77_CHAR_DECL,   F77_CHAR_DECL,   F77_CHAR_DECL,
   F77_INT_DECL,    F77_INT_DECL,    F77_SIN_DECL,    F77_MIN_DECL,
   F77_INT_DECL,    F77_MINOUT_DECL, F77_INT_DECL     F77_4_CHAR ) );

#endif
/*
 * ---------------------------------------------------------------------
 * HPL BLAS Function prototypes
 * ---------------------------------------------------------------------
 */
#ifndef HPL_CALL_CBLAS

int                              HPL_idamax
STDC_ARGS( (
   const int,
   const double *,
   const int
) );
void                             HPL_daxpy
STDC_ARGS( (
   const int,
   const double,
   const double *,
   const int,
   double *,
   const int
) );
void                             HPL_dcopy
STDC_ARGS( (
   const int,
   const double *,
   const int,
   double *,
   const int
) );
void                             HPL_dscal
STDC_ARGS( (
   const int,
   const double,
   double *,
   const int
) );
void                             HPL_dswap
STDC_ARGS( (
   const int,
   double *,
   const int,
   double *,
   const int
) );
void                             HPL_dgemv
STDC_ARGS( (
   const enum HPL_ORDER,
   const enum HPL_TRANS,
   const int,
   const int,
   const double,
   const double *,
   const int,
   const double *,
   const int,
   const double,
   double *,
   const int
) );
void                             HPL_dger
STDC_ARGS( (
   const enum HPL_ORDER,
   const int,
   const int,
   const double,
   const double *,
   const int,
   double *,
   const int,
   double *,
   const int
) );
void                             HPL_dtrsv
STDC_ARGS( (
   const enum HPL_ORDER,
   const enum HPL_UPLO,
   const enum HPL_TRANS,
   const enum HPL_DIAG,
   const int,
   const double *,
   const int,
   double *,
   const int
) );
void                             HPL_dgemm
STDC_ARGS( (
   const enum HPL_ORDER,
   const enum HPL_TRANS,
   const enum HPL_TRANS,
   const int,
   const int,
   const int,
   const double,
   const double *,
   const int,
   const double *,
   const int,
   const double,
   double *,
   const int
) );
void                             HPL_dtrsm
STDC_ARGS( (
   const enum HPL_ORDER,
   const enum HPL_SIDE,
   const enum HPL_UPLO,
   const enum HPL_TRANS,
   const enum HPL_DIAG,
   const int,
   const int,
   const double,
   const double *,
   const int,
   double *,
   const int
) );

#endif

#endif
/*
 * hpl_blas.h
 */
