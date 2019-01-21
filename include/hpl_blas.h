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
double get_timestamp(struct timeval timestamp);

#ifdef SMPI_MEASURE
#pragma message "[SMPI] Tracing the calls to BLAS functions."
#define START_MEASURE(before) ({\
    gettimeofday(&before, NULL);\
})
#define STOP_MEASURE(before, function, M, N, K, lda, ldb, ldc)  ({\
    struct timeval after = {};\
    gettimeofday(&after, NULL);\
    double duration = (after.tv_sec-before.tv_sec) + 1e-6*(after.tv_usec-before.tv_usec);\
    int my_rank, buff=0;\
    double timestamp = get_timestamp(before);\
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);\
    FILE *measure_file = get_measure_file();\
    if(!measure_file) {fprintf(stderr, "error with measure_file\n"); exit(1);}\
    fprintf(measure_file, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d, %g, %g\n", function, __FILE__, __LINE__, my_rank, M, N, K, lda, ldb, ldc, duration, timestamp);\
})
#else
#pragma message "[SMPI] Not tracing the calls to BLAS functions."
#define START_MEASURE(...)   {}
#define STOP_MEASURE(...)    {}
#endif

static double get_param(const char *name) {
    char *val_str = getenv(name);
    if(!val_str) {
        fprintf(stderr, "Environment variable '%s' not defined.\n", name);
        exit(1);
    }
    return atof(val_str);
}

// DGEMM
#if SMPI_OPTIMIZATION_LEVEL >= 1
static double dgemm_coefficient = -1;
static double dgemm_intercept = -1;
#pragma message "[SMPI] Using smpi_execute for HPL_dgemm."
#define  HPL_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)  ({\
    if(dgemm_coefficient < 0 || dgemm_intercept < 0) {\
        dgemm_coefficient = get_param("SMPI_DGEMM_COEFFICIENT");\
        dgemm_intercept = get_param("SMPI_DGEMM_INTERCEPT");\
    }\
    double expected_time;\
    expected_time = dgemm_coefficient*((double)(M))*((double)(N))*((double)(K)) + dgemm_intercept;\
    struct timeval before = {};\
    START_MEASURE(before);\
    if(expected_time > 0)\
        smpi_execute_benched(expected_time);\
    STOP_MEASURE(before, "dgemm", M, N, K, lda, ldb, ldc);\
})
#else
#pragma message "[SMPI] Using cblas_dgemm for HPL_dgemm."
#define  HPL_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)  ({\
    struct timeval before = {};\
    START_MEASURE(before);\
    cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);\
    STOP_MEASURE(before, "dgemm", M, N, K, lda, ldb, ldc);\
})
#endif

// DTRSM
#if SMPI_OPTIMIZATION_LEVEL >= 1
static double dtrsm_coefficient = -1;
static double dtrsm_intercept = -1;
#pragma message "[SMPI] Using smpi_execute for HPL_dtrsm."
#define HPL_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) ({\
    if(dtrsm_coefficient < 0 || dtrsm_intercept < 0) {\
        dtrsm_coefficient = get_param("SMPI_DTRSM_COEFFICIENT");\
        dtrsm_intercept = get_param("SMPI_DTRSM_INTERCEPT");\
    }\
    double expected_time;\
    if((Side) == HplLeft) {\
        expected_time = dtrsm_coefficient*((double)(M))*((double)(M))*((double)(N)) + dtrsm_intercept;\
    } else {\
        expected_time = dtrsm_coefficient*((double)(M))*((double)(N))*((double)(N)) + dtrsm_intercept;\
    }\
    struct timeval before = {};\
    START_MEASURE(before);\
    if(expected_time > 0)\
        smpi_execute_benched(expected_time);\
    STOP_MEASURE(before, "dtrsm", M, N, -1, lda, ldb, -1);\
})
#else
#pragma message "[SMPI] Using cblas_dtrsm for HPL_dtrsm."
#define HPL_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) ({\
    struct timeval before = {};\
    START_MEASURE(before);\
    cblas_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);\
    STOP_MEASURE(before, "dtrsm", M, N, -1, lda, ldb, -1);\
})
#endif

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
