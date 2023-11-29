#ifndef MYPROJECT_H_
#define MYPROJECT_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include <cmath>

#define NPARTICLES  10
#define NPARTICLES2 12
#define NHIDDEN 2
#define NOUT 1
#define NORMBITS 6 // 2^NORMBITS
#define N_TABLE_COS 1024
#define N_TABLE_SINH 1024
#define N_TABLE_COSH 1024
#define TABLE_FRACS 10

typedef ap_fixed<16,6> input_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<16,6> internal_t;
typedef ap_fixed<16,6> weight_t;
typedef ap_fixed<16,6> bias_t;
typedef ap_ufixed<10,0> coslut_t;
typedef ap_ufixed<10,4> sinhlut_t;
typedef ap_ufixed<10,4> coshlut_t;

template<class data_T, int N_TABLE>
static void lut_cos_init(data_T table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
        float phi = float( ii << TABLE_FRACS )/float(1<<(TABLE_FRACS-2));
        data_T real_val = (data_T) (cos(phi));
        table_out[ii] = real_val;
    }
};

template<class data_T, int N_TABLE>
static void lut_sinh_init(data_T table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
        float eta = float( ii << TABLE_FRACS )/float(1<<(TABLE_FRACS-2));
        data_T real_val = (data_T) (sinh(eta));
        table_out[ii] = real_val;
    }
};

template<class data_T, int N_TABLE>
static void lut_cosh_init(data_T table_out[N_TABLE])
{
    for (int ii = 0; ii < N_TABLE; ii++) {
        float eta = float( ii << TABLE_FRACS )/float(1<<(TABLE_FRACS-2));
        data_T real_val = (data_T) (cosh(eta));
        table_out[ii] = real_val;
    }
};

void dot4(input_t p1[4], input_t p2[4], internal_t& dot);

void myproject(
    input_t model_input[(NPARTICLES)*4],
    result_t model_out[1]
);

#endif
