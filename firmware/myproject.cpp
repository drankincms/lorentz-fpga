#include <iostream>

#include "myproject.h"

coslut_t coslut(int index){
  static coslut_t _table[N_TABLE_COS];
  lut_cos_init<coslut_t,N_TABLE_COS>(_table);
  return _table[index];
}

coshlut_t coshlut(int index){
  static coshlut_t _table[N_TABLE_COSH];
  lut_cosh_init<coshlut_t,N_TABLE_COSH>(_table);
  return _table[index];
}

sinhlut_t sinhlut(int index){
  static sinhlut_t _table[N_TABLE_SINH];
  lut_sinh_init<sinhlut_t,N_TABLE_SINH>(_table);
  return _table[index];
}

void dot4(input_t p1[4], input_t p2[4], internal_t& dot) {
//#pragma HLS INLINE
//#pragma function instatiate

// if input could be px, py, pz, E
  //dot = p1[0]*p2[0]-p1[1]*p2[1]-p1[2]*p2[2]-p1[3]*p2[3];

// if input could be pT, eta, phi
  input_t dphi = (p1[2]-p2[2]);
  if (dphi < (input_t)(0.)) dphi = -dphi;
  if (dphi > (input_t)(2.*3.141592654)) dphi = dphi - (input_t)(2.*3.141592654);
  dphi = dphi - (input_t)(3.141592654);
  if (dphi < (input_t)(0.)) dphi = -dphi;
  internal_t cossign = 1.;
  if (dphi > (input_t)(3.141592654/2.)) {
    dphi = dphi - (input_t)(3.141592654/2.);
    cossign = -1.;
  }
  input_t eta1sign = 1.;
  if (p1[1]<0.) eta1sign = -1.;
  input_t eta2sign = 1.;
  if (p2[1]<0.) eta2sign = -1.;
  dot = p1[0]*p2[0];
  internal_t dotmod = coshlut(((eta1sign)*(p1[1]))>>(TABLE_FRACS-2))*coshlut(((eta2sign)*(p2[1]))>>(TABLE_FRACS-2))
                     -cossign*coslut(dphi>>(TABLE_FRACS-2))
                     -eta1sign*eta2sign*sinhlut(((eta1sign)*(p1[1]))>>(TABLE_FRACS-2))*sinhlut(((eta2sign)*(p2[1]))>>(TABLE_FRACS-2));
  dot *= dotmod;
}

void myproject(
    input_t model_input[(NPARTICLES)*4],
    result_t model_out[1]
) {

    #pragma HLS ARRAY_RESHAPE variable=model_input complete dim=0
    #pragma HLS ARRAY_PARTITION variable=model_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=model_input,model_out 
//    #pragma HLS DATAFLOW 
    #pragma HLS PIPELINE II=1

    internal_t dots[(NPARTICLES2)*(NPARTICLES2)];
    #pragma HLS ARRAY_PARTITION variable=dots complete dim=0
    input_t p1[(NPARTICLES2)][4];
    #pragma HLS ARRAY_PARTITION variable=p1 complete dim=0
    P1Prep: for (unsigned int i = 0; i < NPARTICLES; i++) {
    #pragma HLS unroll
      for (unsigned int k = 0; k < 4; k++){
      #pragma HLS unroll
        p1[i][k] = model_input[i*(4)+k];
      }
    }
    p1[NPARTICLES][0]   = 1.; p1[NPARTICLES][1]   = 0.; p1[NPARTICLES][2]   = 0.; p1[NPARTICLES][3]   = 1.;
    p1[NPARTICLES+1][0] = 1.; p1[NPARTICLES+1][1] = 0.; p1[NPARTICLES+1][2] = 0.; p1[NPARTICLES+1][3] = -1.;

    int indices[(NPARTICLES2)*(NPARTICLES+1)/2][4];
    #pragma HLS ARRAY_PARTITION variable=indices complete dim=0
    //this avoids computing the other half of the dots since its symmetric
    int cnt = 0;
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
      #pragma HLS unroll
        if (j<i) {
          indices[cnt][0] = i;
          indices[cnt][1] = j;
          indices[cnt][2] = i*(NPARTICLES2)+j;
          indices[cnt][3] = j*(NPARTICLES2)+i;
          cnt++;
        }
      }
    }

    //compute dot product
    for (unsigned int i = 0; i < (NPARTICLES2)*(NPARTICLES+1)/2; i++) {
    #pragma HLS unroll
      Dot: dot4(p1[indices[i][0]],p1[indices[i][1]],dots[indices[i][2]]);
    }
    for (unsigned int i = 0; i < (NPARTICLES2)*(NPARTICLES+1)/2; i++) {
    #pragma HLS unroll
      dots[indices[i][3]] = dots[indices[i][2]];
    }
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      dots[i*(NPARTICLES2)+i] = 0;
    }

// B1 = delta_{ik}delta_{jl}
// B2 = 1
// B3 = delta_{ij}
// B4 = delta_{ij}delta_{ik}
// B5 = delta_{ik}
// B6 = delta_{jk}

//this requires too many loops, and repeats a lot of the computations

    /*ap_uint<1> B[NPARTICLES2][NPARTICLES2][NPARTICLES2][NPARTICLES2][6];
    #pragma HLS ARRAY_PARTITION variable=B complete dim=0
    
    SetB: for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int k = 0; k < NPARTICLES2; k++) {
    #pragma HLS unroll
          for (unsigned int l = 0; l < NPARTICLES2; l++) {
    #pragma HLS unroll
            B[i][j][k][l][0] = (i==k and j==l) ? 1 : 0; //I x dots
            B[i][j][k][l][1] = 1; //jet mass for all NxN
            B[i][j][k][l][2] = (i==j) ? 1 : 0; //jet mass x I
            B[i][j][k][l][3] = (i==j and j==k) ? 1 : 0; //p(J) . p(i) along diagonal
            B[i][j][k][l][4] = (i==k) ? 1 : 0; //p(J) . p(i) for all NxN
            B[i][j][k][l][5] = (j==k) ? 1 : 0; //p(J) . p(j) for all NxN
          }
        }
      }
    }*/
    
    internal_t jmass = 0.;
    internal_t jdotp[NPARTICLES2] = {0};
    #pragma HLS ARRAY_PARTITION variable=jdotp complete dim=0

// m(J) = sum(dots)
// p(J).p(i) = sum(dots row i)
// these are the only things (besides the dots themselves) that we need from the aggregation
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        AggMJ: jmass += dots[j*NPARTICLES2+i];
        AggJdot: jdotp[j] += dots[j*NPARTICLES2+i];
      }
    }
    
    internal_t T[NPARTICLES2][NPARTICLES2][6];
    #pragma HLS ARRAY_PARTITION variable=T complete dim=0
    
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int b = 0; b < 6; b++) {
    #pragma HLS unroll
          T[i][j][b] = 0;
        }
      }
    }
// this is only necessary when using B_{ijkl} instead of the shortcut above
    /*LinEq2to2: for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int k = 0; k < NPARTICLES2; k++) {
    #pragma HLS unroll
          for (unsigned int l = 0; l < NPARTICLES2; l++) {
    #pragma HLS unroll
            for (unsigned int b = 0; b < 6; b++) {
    #pragma HLS unroll
              T[i][j][b] += dots[k*NPARTICLES2+l]*B[i][j][k][l][b];
            }
          }
        }
      }
    }*/

// B1 = delta_{ik}delta_{jl}
// B2 = 1
// B3 = delta_{ij}
// B4 = delta_{ij}delta_{ik}
// B5 = delta_{ik}
// B6 = delta_{jk}

    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        LinEq2to2_0: T[i][j][0] = dots[i*NPARTICLES2+j];
        LinEq2to2_1: T[i][j][1] = jmass;
        LinEq2to2_2: T[i][j][2] = (i==j) ? jmass : internal_t(0);
        LinEq2to2_3: T[i][j][3] = (i==j) ? jdotp[i] : internal_t(0);
        LinEq2to2_4: T[i][j][4] = jdotp[i];
        LinEq2to2_5: T[i][j][5] = jdotp[j];
      }
    }
    
// dmumy values for now
    weight_t w1_2to2[NHIDDEN*6] = {
      0.125, 0.7375, 0.225, 0.5375, -0.55, 1.33,
      -0.21, -0.77,  0.82,  -1.4,   0.1,  0.98
    };
    #pragma HLS ARRAY_PARTITION variable=w1_2to2 complete dim=0
    bias_t b1_2to2[NHIDDEN] = {0.12, -0.37};
    #pragma HLS ARRAY_PARTITION variable=b1_2to2 complete dim=0
    
    internal_t Tp[NPARTICLES2][NPARTICLES2][NHIDDEN];
    #pragma HLS ARRAY_PARTITION variable=Tp complete dim=0
    
// initialize with bias
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
          Tp[i][j][h] = b1_2to2[h];
        }
      }
    }
    
// 2->2 weights
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
          for (unsigned int b = 0; b < 6; b++) {
    #pragma HLS unroll
            Mult2to2: Tp[i][j][h] += w1_2to2[(h*6)+b]*T[i][j][b];
          }
        }
      }
    }
    
// ReLU
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
          if (Tp[i][j][h]<0.) {
            Tp[i][j][h] = 0;
          }
        }
      }
    }
    
// A1 = delta_{kl}
// A2 = 1
    ap_uint<1> A[NPARTICLES2][NPARTICLES2][2];
    #pragma HLS ARRAY_PARTITION variable=A complete dim=0
    
// this works here since its smaller by N^2
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        A[i][j][0] = 1;
        A[i][j][1] = (i==j);
      }
    }
    
    internal_t R[NHIDDEN][2];
    #pragma HLS ARRAY_PARTITION variable=R complete dim=0
    
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int a = 0; a < 2; a++) {
    #pragma HLS unroll
        R[h][a] = 0;
      }
    }
    
    for (unsigned int i = 0; i < NPARTICLES2; i++) {
    #pragma HLS unroll
      for (unsigned int j = 0; j < NPARTICLES2; j++) {
    #pragma HLS unroll
        for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
          for (unsigned int a = 0; a < 2; a++) {
    #pragma HLS unroll
            LinEq2to0: R[h][a] += (Tp[i][j][h]*A[i][j][a])>>((a == 0) ? NORMBITS*2 : NORMBITS);
          }
        }
      }
    }
    
// dummy values
    weight_t w2_2to0[NHIDDEN*2*NOUT] = {
      -1.65, 4.85,
      3.397, 2.465
    };
    #pragma HLS ARRAY_PARTITION variable=w2_2to0 complete dim=0
    bias_t b2_2to0[NOUT] = {0.5923};
    #pragma HLS ARRAY_PARTITION variable=b2_2to0 complete dim=0
    
    internal_t Rp[NOUT];
    #pragma HLS ARRAY_PARTITION variable=Rp complete dim=0
    
// initialize with bias
    for (unsigned int o = 0; o < NOUT; o++) {
    #pragma HLS unroll
      Rp[o] = b2_2to0[o];
    }
    
// 2->0 weights
    for (unsigned int h = 0; h < NHIDDEN; h++) {
    #pragma HLS unroll
      for (unsigned int a = 0; a < 2; a++) {
    #pragma HLS unroll
        for (unsigned int o = 0; o < NOUT; o++) {
    #pragma HLS unroll
          Mult2to0: Rp[o] += w2_2to0[(h*2)+a*(NOUT)+o]*R[h][a];
        }
      }
    }

    model_out[0] = Rp[0];
}
