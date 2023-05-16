#include "uk.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


// example_sgemm_a1True_b1False(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[12, 8] @DRAM
// )
inline void uk_1xX_a1True_b1False( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
for (int k = 0; k < KC; k++) {
  for (int j = 0; j < 12; j++) {
    for (int i = 0; i < 8; i++) {
      C[(j) * (8) + (i) * (1)] += A[(k) * (8) + (i) * (1)] * B[(k) * (12) + (j) * (1)];
    }
  }
}
}

// example_sgemm_a1True_b1True(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[12, 8] @DRAM
// )
void example_sgemm_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
float *Cb = malloc(12 * 8 * sizeof(*Cb));
free(Cb);
for (int k = 0; k < KC; k++) {
  for (int j = 0; j < 12; j++) {
    for (int i = 0; i < 8; i++) {
      C[(j) * (8) + (i) * (1)] += A[(k) * (8) + (i) * (1)] * B[(k) * (12) + (j) * (1)];
    }
  }
}
}


/* relying on the following instruction..."
neon_broadcast_4xf32(dst,src)
{dst_data} = vld1q_dup_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vfmla_4xf32_4xf32(dst,lhs,rhs,lane)
{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {lane});
*/

/* relying on the following instruction..."
neon_vld_4xf32(dst,src)
{dst_data} = vld1q_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vmul_4xf32(dst,lhs,rhs)
{dst_data} = vmulq_f32({lhs_data}, {rhs_data});
*/

/* relying on the following instruction..."
neon_vst_4xf32(dst,src)
vst1q_f32(&{dst_data}, {src_data});
*/
// uk_8x12_a1True_b1False(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[12, 8] @DRAM
// )
inline void uk_8x12_a1True_b1False( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
float *Cb = malloc(12 * 8 * sizeof(*Cb));
float32x4_t Cb_reg[12][2];
float32x4_t beta_reg;
beta_reg = vld1q_dup_f32(&beta[(0) * (1)]);
float32x4_t C_reg[2];
for (int cj = 0; cj < 12; cj++) {
  for (int cit = 0; cit < 2; cit++) {
    C_reg[cit] = vld1q_f32(&C[(cj) * (8) + (4 * cit) * (1)]);
  }
  for (int cit = 0; cit < 2; cit++) {
    Cb_reg[cj][cit] = vmulq_f32(C_reg[cit], beta_reg);
  }
}
for (int cj = 0; cj < 12; cj++) {
  for (int cit = 0; cit < 2; cit++) {
    vst1q_f32(&Cb[(cj) * (8) + (4 * cit) * (1)], Cb_reg[cj][cit]);
  }
}
Cb_reg[0][0] = vld1q_f32(&Cb[(0) * (8) + (0) * (1)]);
Cb_reg[0][1] = vld1q_f32(&Cb[(0) * (8) + (4) * (1)]);
Cb_reg[1][0] = vld1q_f32(&Cb[(1) * (8) + (0) * (1)]);
Cb_reg[1][1] = vld1q_f32(&Cb[(1) * (8) + (4) * (1)]);
Cb_reg[2][0] = vld1q_f32(&Cb[(2) * (8) + (0) * (1)]);
Cb_reg[2][1] = vld1q_f32(&Cb[(2) * (8) + (4) * (1)]);
Cb_reg[3][0] = vld1q_f32(&Cb[(3) * (8) + (0) * (1)]);
Cb_reg[3][1] = vld1q_f32(&Cb[(3) * (8) + (4) * (1)]);
Cb_reg[4][0] = vld1q_f32(&Cb[(4) * (8) + (0) * (1)]);
Cb_reg[4][1] = vld1q_f32(&Cb[(4) * (8) + (4) * (1)]);
Cb_reg[5][0] = vld1q_f32(&Cb[(5) * (8) + (0) * (1)]);
Cb_reg[5][1] = vld1q_f32(&Cb[(5) * (8) + (4) * (1)]);
Cb_reg[6][0] = vld1q_f32(&Cb[(6) * (8) + (0) * (1)]);
Cb_reg[6][1] = vld1q_f32(&Cb[(6) * (8) + (4) * (1)]);
Cb_reg[7][0] = vld1q_f32(&Cb[(7) * (8) + (0) * (1)]);
Cb_reg[7][1] = vld1q_f32(&Cb[(7) * (8) + (4) * (1)]);
Cb_reg[8][0] = vld1q_f32(&Cb[(8) * (8) + (0) * (1)]);
Cb_reg[8][1] = vld1q_f32(&Cb[(8) * (8) + (4) * (1)]);
Cb_reg[9][0] = vld1q_f32(&Cb[(9) * (8) + (0) * (1)]);
Cb_reg[9][1] = vld1q_f32(&Cb[(9) * (8) + (4) * (1)]);
Cb_reg[10][0] = vld1q_f32(&Cb[(10) * (8) + (0) * (1)]);
Cb_reg[10][1] = vld1q_f32(&Cb[(10) * (8) + (4) * (1)]);
Cb_reg[11][0] = vld1q_f32(&Cb[(11) * (8) + (0) * (1)]);
Cb_reg[11][1] = vld1q_f32(&Cb[(11) * (8) + (4) * (1)]);
float32x4_t A_reg[2];
float32x4_t B_reg[3];
for (int k = 0; k < KC; k++) {
  A_reg[0] = vld1q_f32(&A[(k) * (8) + (4 * 0) * (1)]);
  A_reg[1] = vld1q_f32(&A[(k) * (8) + (4 * 1) * (1)]);
  B_reg[0] = vld1q_f32(&B[(k) * (12) + (4 * 0) * (1)]);
  B_reg[1] = vld1q_f32(&B[(k) * (12) + (4 * 1) * (1)]);
  B_reg[2] = vld1q_f32(&B[(k) * (12) + (4 * 2) * (1)]);
  Cb_reg[0 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][0], A_reg[0], B_reg[0], (0));
  Cb_reg[1 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][0], A_reg[0], B_reg[0], (1));
  Cb_reg[2 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][0], A_reg[0], B_reg[0], (2));
  Cb_reg[3 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][0], A_reg[0], B_reg[0], (3));
  Cb_reg[0 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][1], A_reg[1], B_reg[0], (0));
  Cb_reg[1 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][1], A_reg[1], B_reg[0], (1));
  Cb_reg[2 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][1], A_reg[1], B_reg[0], (2));
  Cb_reg[3 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][1], A_reg[1], B_reg[0], (3));
  Cb_reg[0 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 1][0], A_reg[0], B_reg[1], (0));
  Cb_reg[1 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 1][0], A_reg[0], B_reg[1], (1));
  Cb_reg[2 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 1][0], A_reg[0], B_reg[1], (2));
  Cb_reg[3 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 1][0], A_reg[0], B_reg[1], (3));
  Cb_reg[0 + 4 * 1][1] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 1][1], A_reg[1], B_reg[1], (0));
  Cb_reg[1 + 4 * 1][1] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 1][1], A_reg[1], B_reg[1], (1));
  Cb_reg[2 + 4 * 1][1] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 1][1], A_reg[1], B_reg[1], (2));
  Cb_reg[3 + 4 * 1][1] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 1][1], A_reg[1], B_reg[1], (3));
  Cb_reg[0 + 4 * 2][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 2][0], A_reg[0], B_reg[2], (0));
  Cb_reg[1 + 4 * 2][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 2][0], A_reg[0], B_reg[2], (1));
  Cb_reg[2 + 4 * 2][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 2][0], A_reg[0], B_reg[2], (2));
  Cb_reg[3 + 4 * 2][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 2][0], A_reg[0], B_reg[2], (3));
  Cb_reg[0 + 4 * 2][1] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 2][1], A_reg[1], B_reg[2], (0));
  Cb_reg[1 + 4 * 2][1] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 2][1], A_reg[1], B_reg[2], (1));
  Cb_reg[2 + 4 * 2][1] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 2][1], A_reg[1], B_reg[2], (2));
  Cb_reg[3 + 4 * 2][1] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 2][1], A_reg[1], B_reg[2], (3));
}
vst1q_f32(&Cb[(0 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[0 + 4 * 0][0]);
vst1q_f32(&Cb[(0 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[0 + 4 * 0][1]);
vst1q_f32(&Cb[(1 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[1 + 4 * 0][0]);
vst1q_f32(&Cb[(1 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[1 + 4 * 0][1]);
vst1q_f32(&Cb[(2 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[2 + 4 * 0][0]);
vst1q_f32(&Cb[(2 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[2 + 4 * 0][1]);
vst1q_f32(&Cb[(3 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[3 + 4 * 0][0]);
vst1q_f32(&Cb[(3 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[3 + 4 * 0][1]);
vst1q_f32(&Cb[(0 + 4 * 1) * (8) + (4 * 0) * (1)], Cb_reg[0 + 4 * 1][0]);
vst1q_f32(&Cb[(0 + 4 * 1) * (8) + (4 * 1) * (1)], Cb_reg[0 + 4 * 1][1]);
vst1q_f32(&Cb[(1 + 4 * 1) * (8) + (4 * 0) * (1)], Cb_reg[1 + 4 * 1][0]);
vst1q_f32(&Cb[(1 + 4 * 1) * (8) + (4 * 1) * (1)], Cb_reg[1 + 4 * 1][1]);
vst1q_f32(&Cb[(2 + 4 * 1) * (8) + (4 * 0) * (1)], Cb_reg[2 + 4 * 1][0]);
vst1q_f32(&Cb[(2 + 4 * 1) * (8) + (4 * 1) * (1)], Cb_reg[2 + 4 * 1][1]);
vst1q_f32(&Cb[(3 + 4 * 1) * (8) + (4 * 0) * (1)], Cb_reg[3 + 4 * 1][0]);
vst1q_f32(&Cb[(3 + 4 * 1) * (8) + (4 * 1) * (1)], Cb_reg[3 + 4 * 1][1]);
vst1q_f32(&Cb[(0 + 4 * 2) * (8) + (4 * 0) * (1)], Cb_reg[0 + 4 * 2][0]);
vst1q_f32(&Cb[(0 + 4 * 2) * (8) + (4 * 1) * (1)], Cb_reg[0 + 4 * 2][1]);
vst1q_f32(&Cb[(1 + 4 * 2) * (8) + (4 * 0) * (1)], Cb_reg[1 + 4 * 2][0]);
vst1q_f32(&Cb[(1 + 4 * 2) * (8) + (4 * 1) * (1)], Cb_reg[1 + 4 * 2][1]);
vst1q_f32(&Cb[(2 + 4 * 2) * (8) + (4 * 0) * (1)], Cb_reg[2 + 4 * 2][0]);
vst1q_f32(&Cb[(2 + 4 * 2) * (8) + (4 * 1) * (1)], Cb_reg[2 + 4 * 2][1]);
vst1q_f32(&Cb[(3 + 4 * 2) * (8) + (4 * 0) * (1)], Cb_reg[3 + 4 * 2][0]);
vst1q_f32(&Cb[(3 + 4 * 2) * (8) + (4 * 1) * (1)], Cb_reg[3 + 4 * 2][1]);
for (int cj = 0; cj < 12; cj++) {
  for (int ci = 0; ci < 8; ci++) {
    C[(cj) * (8) + (ci) * (1)] = Cb[(cj) * (8) + (ci) * (1)];
  }
}
free(Cb);
}

// uk_8x12_a1True_b1True(
//     KC : size,
//     alpha : f32[1] @DRAM,
//     A : f32[KC, 8] @DRAM,
//     B : f32[KC, 12] @DRAM,
//     beta : f32[1] @DRAM,
//     C : f32[12, 8] @DRAM
// )
inline void uk_8x12_a1True_b1True( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
float *Cb = malloc(12 * 8 * sizeof(*Cb));
free(Cb);
float32x4_t C_reg[12][2];
C_reg[0][0] = vld1q_f32(&C[(0) * (8) + (0) * (1)]);
C_reg[0][1] = vld1q_f32(&C[(0) * (8) + (4) * (1)]);
C_reg[1][0] = vld1q_f32(&C[(1) * (8) + (0) * (1)]);
C_reg[1][1] = vld1q_f32(&C[(1) * (8) + (4) * (1)]);
C_reg[2][0] = vld1q_f32(&C[(2) * (8) + (0) * (1)]);
C_reg[2][1] = vld1q_f32(&C[(2) * (8) + (4) * (1)]);
C_reg[3][0] = vld1q_f32(&C[(3) * (8) + (0) * (1)]);
C_reg[3][1] = vld1q_f32(&C[(3) * (8) + (4) * (1)]);
C_reg[4][0] = vld1q_f32(&C[(4) * (8) + (0) * (1)]);
C_reg[4][1] = vld1q_f32(&C[(4) * (8) + (4) * (1)]);
C_reg[5][0] = vld1q_f32(&C[(5) * (8) + (0) * (1)]);
C_reg[5][1] = vld1q_f32(&C[(5) * (8) + (4) * (1)]);
C_reg[6][0] = vld1q_f32(&C[(6) * (8) + (0) * (1)]);
C_reg[6][1] = vld1q_f32(&C[(6) * (8) + (4) * (1)]);
C_reg[7][0] = vld1q_f32(&C[(7) * (8) + (0) * (1)]);
C_reg[7][1] = vld1q_f32(&C[(7) * (8) + (4) * (1)]);
C_reg[8][0] = vld1q_f32(&C[(8) * (8) + (0) * (1)]);
C_reg[8][1] = vld1q_f32(&C[(8) * (8) + (4) * (1)]);
C_reg[9][0] = vld1q_f32(&C[(9) * (8) + (0) * (1)]);
C_reg[9][1] = vld1q_f32(&C[(9) * (8) + (4) * (1)]);
C_reg[10][0] = vld1q_f32(&C[(10) * (8) + (0) * (1)]);
C_reg[10][1] = vld1q_f32(&C[(10) * (8) + (4) * (1)]);
C_reg[11][0] = vld1q_f32(&C[(11) * (8) + (0) * (1)]);
C_reg[11][1] = vld1q_f32(&C[(11) * (8) + (4) * (1)]);
float32x4_t A_reg[2];
float32x4_t B_reg[3];
for (int k = 0; k < KC; k++) {
  A_reg[0] = vld1q_f32(&A[(k) * (8) + (4 * 0) * (1)]);
  A_reg[1] = vld1q_f32(&A[(k) * (8) + (4 * 1) * (1)]);
  B_reg[0] = vld1q_f32(&B[(k) * (12) + (4 * 0) * (1)]);
  B_reg[1] = vld1q_f32(&B[(k) * (12) + (4 * 1) * (1)]);
  B_reg[2] = vld1q_f32(&B[(k) * (12) + (4 * 2) * (1)]);
  C_reg[0 + 4 * 0][0] = vfmaq_laneq_f32(C_reg[0 + 4 * 0][0], A_reg[0], B_reg[0], (0));
  C_reg[1 + 4 * 0][0] = vfmaq_laneq_f32(C_reg[1 + 4 * 0][0], A_reg[0], B_reg[0], (1));
  C_reg[2 + 4 * 0][0] = vfmaq_laneq_f32(C_reg[2 + 4 * 0][0], A_reg[0], B_reg[0], (2));
  C_reg[3 + 4 * 0][0] = vfmaq_laneq_f32(C_reg[3 + 4 * 0][0], A_reg[0], B_reg[0], (3));
  C_reg[0 + 4 * 0][1] = vfmaq_laneq_f32(C_reg[0 + 4 * 0][1], A_reg[1], B_reg[0], (0));
  C_reg[1 + 4 * 0][1] = vfmaq_laneq_f32(C_reg[1 + 4 * 0][1], A_reg[1], B_reg[0], (1));
  C_reg[2 + 4 * 0][1] = vfmaq_laneq_f32(C_reg[2 + 4 * 0][1], A_reg[1], B_reg[0], (2));
  C_reg[3 + 4 * 0][1] = vfmaq_laneq_f32(C_reg[3 + 4 * 0][1], A_reg[1], B_reg[0], (3));
  C_reg[0 + 4 * 1][0] = vfmaq_laneq_f32(C_reg[0 + 4 * 1][0], A_reg[0], B_reg[1], (0));
  C_reg[1 + 4 * 1][0] = vfmaq_laneq_f32(C_reg[1 + 4 * 1][0], A_reg[0], B_reg[1], (1));
  C_reg[2 + 4 * 1][0] = vfmaq_laneq_f32(C_reg[2 + 4 * 1][0], A_reg[0], B_reg[1], (2));
  C_reg[3 + 4 * 1][0] = vfmaq_laneq_f32(C_reg[3 + 4 * 1][0], A_reg[0], B_reg[1], (3));
  C_reg[0 + 4 * 1][1] = vfmaq_laneq_f32(C_reg[0 + 4 * 1][1], A_reg[1], B_reg[1], (0));
  C_reg[1 + 4 * 1][1] = vfmaq_laneq_f32(C_reg[1 + 4 * 1][1], A_reg[1], B_reg[1], (1));
  C_reg[2 + 4 * 1][1] = vfmaq_laneq_f32(C_reg[2 + 4 * 1][1], A_reg[1], B_reg[1], (2));
  C_reg[3 + 4 * 1][1] = vfmaq_laneq_f32(C_reg[3 + 4 * 1][1], A_reg[1], B_reg[1], (3));
  C_reg[0 + 4 * 2][0] = vfmaq_laneq_f32(C_reg[0 + 4 * 2][0], A_reg[0], B_reg[2], (0));
  C_reg[1 + 4 * 2][0] = vfmaq_laneq_f32(C_reg[1 + 4 * 2][0], A_reg[0], B_reg[2], (1));
  C_reg[2 + 4 * 2][0] = vfmaq_laneq_f32(C_reg[2 + 4 * 2][0], A_reg[0], B_reg[2], (2));
  C_reg[3 + 4 * 2][0] = vfmaq_laneq_f32(C_reg[3 + 4 * 2][0], A_reg[0], B_reg[2], (3));
  C_reg[0 + 4 * 2][1] = vfmaq_laneq_f32(C_reg[0 + 4 * 2][1], A_reg[1], B_reg[2], (0));
  C_reg[1 + 4 * 2][1] = vfmaq_laneq_f32(C_reg[1 + 4 * 2][1], A_reg[1], B_reg[2], (1));
  C_reg[2 + 4 * 2][1] = vfmaq_laneq_f32(C_reg[2 + 4 * 2][1], A_reg[1], B_reg[2], (2));
  C_reg[3 + 4 * 2][1] = vfmaq_laneq_f32(C_reg[3 + 4 * 2][1], A_reg[1], B_reg[2], (3));
}
vst1q_f32(&C[(0 + 4 * 0) * (8) + (4 * 0) * (1)], C_reg[0 + 4 * 0][0]);
vst1q_f32(&C[(0 + 4 * 0) * (8) + (4 * 1) * (1)], C_reg[0 + 4 * 0][1]);
vst1q_f32(&C[(1 + 4 * 0) * (8) + (4 * 0) * (1)], C_reg[1 + 4 * 0][0]);
vst1q_f32(&C[(1 + 4 * 0) * (8) + (4 * 1) * (1)], C_reg[1 + 4 * 0][1]);
vst1q_f32(&C[(2 + 4 * 0) * (8) + (4 * 0) * (1)], C_reg[2 + 4 * 0][0]);
vst1q_f32(&C[(2 + 4 * 0) * (8) + (4 * 1) * (1)], C_reg[2 + 4 * 0][1]);
vst1q_f32(&C[(3 + 4 * 0) * (8) + (4 * 0) * (1)], C_reg[3 + 4 * 0][0]);
vst1q_f32(&C[(3 + 4 * 0) * (8) + (4 * 1) * (1)], C_reg[3 + 4 * 0][1]);
vst1q_f32(&C[(0 + 4 * 1) * (8) + (4 * 0) * (1)], C_reg[0 + 4 * 1][0]);
vst1q_f32(&C[(0 + 4 * 1) * (8) + (4 * 1) * (1)], C_reg[0 + 4 * 1][1]);
vst1q_f32(&C[(1 + 4 * 1) * (8) + (4 * 0) * (1)], C_reg[1 + 4 * 1][0]);
vst1q_f32(&C[(1 + 4 * 1) * (8) + (4 * 1) * (1)], C_reg[1 + 4 * 1][1]);
vst1q_f32(&C[(2 + 4 * 1) * (8) + (4 * 0) * (1)], C_reg[2 + 4 * 1][0]);
vst1q_f32(&C[(2 + 4 * 1) * (8) + (4 * 1) * (1)], C_reg[2 + 4 * 1][1]);
vst1q_f32(&C[(3 + 4 * 1) * (8) + (4 * 0) * (1)], C_reg[3 + 4 * 1][0]);
vst1q_f32(&C[(3 + 4 * 1) * (8) + (4 * 1) * (1)], C_reg[3 + 4 * 1][1]);
vst1q_f32(&C[(0 + 4 * 2) * (8) + (4 * 0) * (1)], C_reg[0 + 4 * 2][0]);
vst1q_f32(&C[(0 + 4 * 2) * (8) + (4 * 1) * (1)], C_reg[0 + 4 * 2][1]);
vst1q_f32(&C[(1 + 4 * 2) * (8) + (4 * 0) * (1)], C_reg[1 + 4 * 2][0]);
vst1q_f32(&C[(1 + 4 * 2) * (8) + (4 * 1) * (1)], C_reg[1 + 4 * 2][1]);
vst1q_f32(&C[(2 + 4 * 2) * (8) + (4 * 0) * (1)], C_reg[2 + 4 * 2][0]);
vst1q_f32(&C[(2 + 4 * 2) * (8) + (4 * 1) * (1)], C_reg[2 + 4 * 2][1]);
vst1q_f32(&C[(3 + 4 * 2) * (8) + (4 * 0) * (1)], C_reg[3 + 4 * 2][0]);
vst1q_f32(&C[(3 + 4 * 2) * (8) + (4 * 1) * (1)], C_reg[3 + 4 * 2][1]);
}

inline void uk_8x4_a1True_b1False( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
	float *Cb = malloc(4 * 8 * sizeof(*Cb));
	float32x4_t Cb_reg[4][2];
	float32x4_t beta_reg;
	beta_reg = vld1q_dup_f32(&beta[(0) * (1)]);
	float32x4_t C_reg[2];
	for (int cj = 0; cj < 4; cj++) {
		  for (int cit = 0; cit < 2; cit++) {
			      C_reg[cit] = vld1q_f32(&C[(cj) * (8) + (4 * cit) * (1)]);
			        }
		    for (int cit = 0; cit < 2; cit++) {
			        Cb_reg[cj][cit] = vmulq_f32(C_reg[cit], beta_reg);
				  }
	}
	for (int cj = 0; cj < 4; cj++) {
		  for (int cit = 0; cit < 2; cit++) {
			      vst1q_f32(&Cb[(cj) * (8) + (4 * cit) * (1)], Cb_reg[cj][cit]);
			        }
	}
	Cb_reg[0][0] = vld1q_f32(&Cb[(0) * (8) + (0) * (1)]);
	Cb_reg[0][1] = vld1q_f32(&Cb[(0) * (8) + (4) * (1)]);
	Cb_reg[1][0] = vld1q_f32(&Cb[(1) * (8) + (0) * (1)]);
	Cb_reg[1][1] = vld1q_f32(&Cb[(1) * (8) + (4) * (1)]);
	Cb_reg[2][0] = vld1q_f32(&Cb[(2) * (8) + (0) * (1)]);
	Cb_reg[2][1] = vld1q_f32(&Cb[(2) * (8) + (4) * (1)]);
	Cb_reg[3][0] = vld1q_f32(&Cb[(3) * (8) + (0) * (1)]);
	Cb_reg[3][1] = vld1q_f32(&Cb[(3) * (8) + (4) * (1)]);
	float32x4_t A_reg[2];
	float32x4_t B_reg[1];
	for (int k = 0; k < KC; k++) {
		  A_reg[0] = vld1q_f32(&A[(k) * (8) + (4 * 0) * (1)]);
		    A_reg[1] = vld1q_f32(&A[(k) * (8) + (4 * 1) * (1)]);
		      B_reg[0] = vld1q_f32(&B[(k) * (4) + (4 * 0) * (1)]);
		        Cb_reg[0 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][0], A_reg[0], B_reg[0], (0));
			  Cb_reg[1 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][0], A_reg[0], B_reg[0], (1));
			    Cb_reg[2 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][0], A_reg[0], B_reg[0], (2));
			      Cb_reg[3 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][0], A_reg[0], B_reg[0], (3));
			        Cb_reg[0 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][1], A_reg[1], B_reg[0], (0));
				  Cb_reg[1 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][1], A_reg[1], B_reg[0], (1));
				    Cb_reg[2 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][1], A_reg[1], B_reg[0], (2));
				      Cb_reg[3 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][1], A_reg[1], B_reg[0], (3));
	}
	vst1q_f32(&Cb[(0 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[0 + 4 * 0][0]);
	vst1q_f32(&Cb[(0 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[0 + 4 * 0][1]);
	vst1q_f32(&Cb[(1 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[1 + 4 * 0][0]);
	vst1q_f32(&Cb[(1 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[1 + 4 * 0][1]);
	vst1q_f32(&Cb[(2 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[2 + 4 * 0][0]);
	vst1q_f32(&Cb[(2 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[2 + 4 * 0][1]);
	vst1q_f32(&Cb[(3 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[3 + 4 * 0][0]);
	vst1q_f32(&Cb[(3 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[3 + 4 * 0][1]);
	for (int cj = 0; cj < 4; cj++) {
		  for (int ci = 0; ci < 8; ci++) {
			      C[(cj) * (8) + (ci) * (1)] = Cb[(cj) * (8) + (ci) * (1)];
			        }
	}
	free(Cb);
}

inline void uk_8x8_a1True_b1False( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
	float *Cb = malloc(8 * 8 * sizeof(*Cb));
	float32x4_t Cb_reg[8][2];
	float32x4_t beta_reg;
	beta_reg = vld1q_dup_f32(&beta[(0) * (1)]);
	float32x4_t C_reg[2];
	for (int cj = 0; cj < 8; cj++) {
		  for (int cit = 0; cit < 2; cit++) {
			      C_reg[cit] = vld1q_f32(&C[(cj) * (8) + (4 * cit) * (1)]);
			        }
		    for (int cit = 0; cit < 2; cit++) {
			        Cb_reg[cj][cit] = vmulq_f32(C_reg[cit], beta_reg);
				  }
	}
	for (int cj = 0; cj < 8; cj++) {
		  for (int cit = 0; cit < 2; cit++) {
			      vst1q_f32(&Cb[(cj) * (8) + (4 * cit) * (1)], Cb_reg[cj][cit]);
			        }
	}
	Cb_reg[0][0] = vld1q_f32(&Cb[(0) * (8) + (0) * (1)]);
	Cb_reg[0][1] = vld1q_f32(&Cb[(0) * (8) + (4) * (1)]);
	Cb_reg[1][0] = vld1q_f32(&Cb[(1) * (8) + (0) * (1)]);
	Cb_reg[1][1] = vld1q_f32(&Cb[(1) * (8) + (4) * (1)]);
	Cb_reg[2][0] = vld1q_f32(&Cb[(2) * (8) + (0) * (1)]);
	Cb_reg[2][1] = vld1q_f32(&Cb[(2) * (8) + (4) * (1)]);
	Cb_reg[3][0] = vld1q_f32(&Cb[(3) * (8) + (0) * (1)]);
	Cb_reg[3][1] = vld1q_f32(&Cb[(3) * (8) + (4) * (1)]);
	Cb_reg[4][0] = vld1q_f32(&Cb[(4) * (8) + (0) * (1)]);
	Cb_reg[4][1] = vld1q_f32(&Cb[(4) * (8) + (4) * (1)]);
	Cb_reg[5][0] = vld1q_f32(&Cb[(5) * (8) + (0) * (1)]);
	Cb_reg[5][1] = vld1q_f32(&Cb[(5) * (8) + (4) * (1)]);
	Cb_reg[6][0] = vld1q_f32(&Cb[(6) * (8) + (0) * (1)]);
	Cb_reg[6][1] = vld1q_f32(&Cb[(6) * (8) + (4) * (1)]);
	Cb_reg[7][0] = vld1q_f32(&Cb[(7) * (8) + (0) * (1)]);
	Cb_reg[7][1] = vld1q_f32(&Cb[(7) * (8) + (4) * (1)]);
	float32x4_t A_reg[2];
	float32x4_t B_reg[2];
	for (int k = 0; k < KC; k++) {
		  A_reg[0] = vld1q_f32(&A[(k) * (8) + (4 * 0) * (1)]);
		    A_reg[1] = vld1q_f32(&A[(k) * (8) + (4 * 1) * (1)]);
		      B_reg[0] = vld1q_f32(&B[(k) * (8) + (4 * 0) * (1)]);
		        B_reg[1] = vld1q_f32(&B[(k) * (8) + (4 * 1) * (1)]);
			  Cb_reg[0 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][0], A_reg[0], B_reg[0], (0));
			    Cb_reg[1 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][0], A_reg[0], B_reg[0], (0));
			      Cb_reg[2 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][0], A_reg[0], B_reg[0], (0));
			        Cb_reg[3 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][0], A_reg[0], B_reg[0], (0));
				  Cb_reg[0 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][1], A_reg[1], B_reg[0], (0));
				    Cb_reg[1 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][1], A_reg[1], B_reg[0], (0));
				      Cb_reg[2 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][1], A_reg[1], B_reg[0], (0));
				        Cb_reg[3 + 4 * 0][1] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][1], A_reg[1], B_reg[0], (0));
					  Cb_reg[0 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 1][0], A_reg[0], B_reg[0], (1));
					    Cb_reg[1 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 1][0], A_reg[0], B_reg[0], (1));
					      Cb_reg[2 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 1][0], A_reg[0], B_reg[0], (1));
					        Cb_reg[3 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 1][0], A_reg[0], B_reg[0], (1));
						  Cb_reg[0 + 4 * 1][1] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 1][1], A_reg[1], B_reg[0], (1));
						    Cb_reg[1 + 4 * 1][1] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 1][1], A_reg[1], B_reg[0], (1));
						      Cb_reg[2 + 4 * 1][1] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 1][1], A_reg[1], B_reg[0], (1));
						        Cb_reg[3 + 4 * 1][1] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 1][1], A_reg[1], B_reg[0], (1));
	}
	vst1q_f32(&Cb[(0 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[0 + 4 * 0][0]);
	vst1q_f32(&Cb[(0 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[0 + 4 * 0][1]);
	vst1q_f32(&Cb[(1 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[1 + 4 * 0][0]);
	vst1q_f32(&Cb[(1 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[1 + 4 * 0][1]);
	vst1q_f32(&Cb[(2 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[2 + 4 * 0][0]);
	vst1q_f32(&Cb[(2 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[2 + 4 * 0][1]);
	vst1q_f32(&Cb[(3 + 4 * 0) * (8) + (4 * 0) * (1)], Cb_reg[3 + 4 * 0][0]);
	vst1q_f32(&Cb[(3 + 4 * 0) * (8) + (4 * 1) * (1)], Cb_reg[3 + 4 * 0][1]);
	vst1q_f32(&Cb[(0 + 4 * 1) * (8) + (4 * 0) * (1)], Cb_reg[0 + 4 * 1][0]);
	vst1q_f32(&Cb[(0 + 4 * 1) * (8) + (4 * 1) * (1)], Cb_reg[0 + 4 * 1][1]);
	vst1q_f32(&Cb[(1 + 4 * 1) * (8) + (4 * 0) * (1)], Cb_reg[1 + 4 * 1][0]);
	vst1q_f32(&Cb[(1 + 4 * 1) * (8) + (4 * 1) * (1)], Cb_reg[1 + 4 * 1][1]);
	vst1q_f32(&Cb[(2 + 4 * 1) * (8) + (4 * 0) * (1)], Cb_reg[2 + 4 * 1][0]);
	vst1q_f32(&Cb[(2 + 4 * 1) * (8) + (4 * 1) * (1)], Cb_reg[2 + 4 * 1][1]);
	vst1q_f32(&Cb[(3 + 4 * 1) * (8) + (4 * 0) * (1)], Cb_reg[3 + 4 * 1][0]);
	vst1q_f32(&Cb[(3 + 4 * 1) * (8) + (4 * 1) * (1)], Cb_reg[3 + 4 * 1][1]);
	for (int cj = 0; cj < 8; cj++) {
		  for (int ci = 0; ci < 8; ci++) {
			      C[(cj) * (8) + (ci) * (1)] = Cb[(cj) * (8) + (ci) * (1)];
			        }
	}
	free(Cb);
}

inline void uk_4x8_a1True_b1False( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
	float *Cb = malloc(8 * 4 * sizeof(*Cb));
	float32x4_t Cb_reg[8][1];
	float32x4_t beta_reg;
	beta_reg = vld1q_dup_f32(&beta[(0) * (1)]);
	float32x4_t C_reg[1];
	for (int cj = 0; cj < 8; cj++) {
		  for (int cit = 0; cit < 1; cit++) {
			      C_reg[cit] = vld1q_f32(&C[(cj) * (4) + (4 * cit) * (1)]);
			        }
		    for (int cit = 0; cit < 1; cit++) {
			        Cb_reg[cj][cit] = vmulq_f32(C_reg[cit], beta_reg);
				  }
	}
	for (int cj = 0; cj < 8; cj++) {
		  for (int cit = 0; cit < 1; cit++) {
			      vst1q_f32(&Cb[(cj) * (4) + (4 * cit) * (1)], Cb_reg[cj][cit]);
			        }
	}
	Cb_reg[0][0] = vld1q_f32(&Cb[(0) * (4) + (0) * (1)]);
	Cb_reg[1][0] = vld1q_f32(&Cb[(1) * (4) + (0) * (1)]);
	Cb_reg[2][0] = vld1q_f32(&Cb[(2) * (4) + (0) * (1)]);
	Cb_reg[3][0] = vld1q_f32(&Cb[(3) * (4) + (0) * (1)]);
	Cb_reg[4][0] = vld1q_f32(&Cb[(4) * (4) + (0) * (1)]);
	Cb_reg[5][0] = vld1q_f32(&Cb[(5) * (4) + (0) * (1)]);
	Cb_reg[6][0] = vld1q_f32(&Cb[(6) * (4) + (0) * (1)]);
	Cb_reg[7][0] = vld1q_f32(&Cb[(7) * (4) + (0) * (1)]);
	float32x4_t A_reg[1];
	float32x4_t B_reg[2];
	for (int k = 0; k < KC; k++) {
		  A_reg[0] = vld1q_f32(&A[(k) * (4) + (4 * 0) * (1)]);
		    B_reg[0] = vld1q_f32(&B[(k) * (8) + (4 * 0) * (1)]);
		      B_reg[1] = vld1q_f32(&B[(k) * (8) + (4 * 1) * (1)]);
		        Cb_reg[0 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][0], A_reg[0], B_reg[0], (0));
			  Cb_reg[1 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][0], A_reg[0], B_reg[0], (1));
			    Cb_reg[2 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][0], A_reg[0], B_reg[0], (2));
			      Cb_reg[3 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][0], A_reg[0], B_reg[0], (3));
			        Cb_reg[0 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 1][0], A_reg[0], B_reg[1], (0));
				  Cb_reg[1 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 1][0], A_reg[0], B_reg[1], (1));
				    Cb_reg[2 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 1][0], A_reg[0], B_reg[1], (2));
				      Cb_reg[3 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 1][0], A_reg[0], B_reg[1], (3));
	}
	vst1q_f32(&Cb[(0 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[0 + 4 * 0][0]);
	vst1q_f32(&Cb[(1 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[1 + 4 * 0][0]);
	vst1q_f32(&Cb[(2 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[2 + 4 * 0][0]);
	vst1q_f32(&Cb[(3 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[3 + 4 * 0][0]);
	vst1q_f32(&Cb[(0 + 4 * 1) * (4) + (4 * 0) * (1)], Cb_reg[0 + 4 * 1][0]);
	vst1q_f32(&Cb[(1 + 4 * 1) * (4) + (4 * 0) * (1)], Cb_reg[1 + 4 * 1][0]);
	vst1q_f32(&Cb[(2 + 4 * 1) * (4) + (4 * 0) * (1)], Cb_reg[2 + 4 * 1][0]);
	vst1q_f32(&Cb[(3 + 4 * 1) * (4) + (4 * 0) * (1)], Cb_reg[3 + 4 * 1][0]);
	for (int cj = 0; cj < 8; cj++) {
		  for (int ci = 0; ci < 4; ci++) {
			      C[(cj) * (4) + (ci) * (1)] = Cb[(cj) * (4) + (ci) * (1)];
			        }
	}
	free(Cb);
}

inline void uk_4x4_a1True_b1False( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
	float *Cb = malloc(4 * 4 * sizeof(*Cb));
	float32x4_t Cb_reg[4][1];
	float32x4_t beta_reg;
	beta_reg = vld1q_dup_f32(&beta[(0) * (1)]);
	float32x4_t C_reg[1];
	for (int cj = 0; cj < 4; cj++) {
		  for (int cit = 0; cit < 1; cit++) {
			      C_reg[cit] = vld1q_f32(&C[(cj) * (4) + (4 * cit) * (1)]);
			        }
		    for (int cit = 0; cit < 1; cit++) {
			        Cb_reg[cj][cit] = vmulq_f32(C_reg[cit], beta_reg);
				  }
	}
	for (int cj = 0; cj < 4; cj++) {
		  for (int cit = 0; cit < 1; cit++) {
			      vst1q_f32(&Cb[(cj) * (4) + (4 * cit) * (1)], Cb_reg[cj][cit]);
			        }
	}
	Cb_reg[0][0] = vld1q_f32(&Cb[(0) * (4) + (0) * (1)]);
	Cb_reg[1][0] = vld1q_f32(&Cb[(1) * (4) + (0) * (1)]);
	Cb_reg[2][0] = vld1q_f32(&Cb[(2) * (4) + (0) * (1)]);
	Cb_reg[3][0] = vld1q_f32(&Cb[(3) * (4) + (0) * (1)]);
	float32x4_t A_reg[1];
	float32x4_t B_reg[1];
	for (int k = 0; k < KC; k++) {
		  A_reg[0] = vld1q_f32(&A[(k) * (4) + (4 * 0) * (1)]);
		    B_reg[0] = vld1q_f32(&B[(k) * (4) + (4 * 0) * (1)]);
		      Cb_reg[0 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][0], A_reg[0], B_reg[0], (0));
		        Cb_reg[1 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][0], A_reg[0], B_reg[0], (1));
			  Cb_reg[2 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][0], A_reg[0], B_reg[0], (2));
			    Cb_reg[3 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][0], A_reg[0], B_reg[0], (3));
	}
	vst1q_f32(&Cb[(0 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[0 + 4 * 0][0]);
	vst1q_f32(&Cb[(1 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[1 + 4 * 0][0]);
	vst1q_f32(&Cb[(2 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[2 + 4 * 0][0]);
	vst1q_f32(&Cb[(3 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[3 + 4 * 0][0]);
	for (int cj = 0; cj < 4; cj++) {
		  for (int ci = 0; ci < 4; ci++) {
			      C[(cj) * (4) + (ci) * (1)] = Cb[(cj) * (4) + (ci) * (1)];
			        }
	}
	free(Cb);
}

inline void uk_4x12_a1True_b1False( void *ctxt, int_fast32_t KC, const float* alpha, const float* A, const float* B, const float* beta, float* C ) {
	float *Cb = malloc(12 * 4 * sizeof(*Cb));
	float32x4_t Cb_reg[12][1];
	float32x4_t beta_reg;
	beta_reg = vld1q_dup_f32(&beta[(0) * (1)]);
	float32x4_t C_reg[1];
	for (int cj = 0; cj < 12; cj++) {
		  for (int cit = 0; cit < 1; cit++) {
			      C_reg[cit] = vld1q_f32(&C[(cj) * (4) + (4 * cit) * (1)]);
			        }
		    for (int cit = 0; cit < 1; cit++) {
			        Cb_reg[cj][cit] = vmulq_f32(C_reg[cit], beta_reg);
				  }
	}
	for (int cj = 0; cj < 12; cj++) {
		  for (int cit = 0; cit < 1; cit++) {
			      vst1q_f32(&Cb[(cj) * (4) + (4 * cit) * (1)], Cb_reg[cj][cit]);
			        }
	}
	Cb_reg[0][0] = vld1q_f32(&Cb[(0) * (4) + (0) * (1)]);
	Cb_reg[1][0] = vld1q_f32(&Cb[(1) * (4) + (0) * (1)]);
	Cb_reg[2][0] = vld1q_f32(&Cb[(2) * (4) + (0) * (1)]);
	Cb_reg[3][0] = vld1q_f32(&Cb[(3) * (4) + (0) * (1)]);
	Cb_reg[4][0] = vld1q_f32(&Cb[(4) * (4) + (0) * (1)]);
	Cb_reg[5][0] = vld1q_f32(&Cb[(5) * (4) + (0) * (1)]);
	Cb_reg[6][0] = vld1q_f32(&Cb[(6) * (4) + (0) * (1)]);
	Cb_reg[7][0] = vld1q_f32(&Cb[(7) * (4) + (0) * (1)]);
	Cb_reg[8][0] = vld1q_f32(&Cb[(8) * (4) + (0) * (1)]);
	Cb_reg[9][0] = vld1q_f32(&Cb[(9) * (4) + (0) * (1)]);
	Cb_reg[10][0] = vld1q_f32(&Cb[(10) * (4) + (0) * (1)]);
	Cb_reg[11][0] = vld1q_f32(&Cb[(11) * (4) + (0) * (1)]);
	float32x4_t A_reg[1];
	float32x4_t B_reg[3];
	for (int k = 0; k < KC; k++) {
		  A_reg[0] = vld1q_f32(&A[(k) * (4) + (4 * 0) * (1)]);
		    B_reg[0] = vld1q_f32(&B[(k) * (12) + (4 * 0) * (1)]);
		      B_reg[1] = vld1q_f32(&B[(k) * (12) + (4 * 1) * (1)]);
		        B_reg[2] = vld1q_f32(&B[(k) * (12) + (4 * 2) * (1)]);
			  Cb_reg[0 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 0][0], A_reg[0], B_reg[0], (0));
			    Cb_reg[1 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 0][0], A_reg[0], B_reg[0], (0));
			      Cb_reg[2 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 0][0], A_reg[0], B_reg[0], (0));
			        Cb_reg[3 + 4 * 0][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 0][0], A_reg[0], B_reg[0], (0));
				  Cb_reg[0 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 1][0], A_reg[0], B_reg[0], (1));
				    Cb_reg[1 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 1][0], A_reg[0], B_reg[0], (1));
				      Cb_reg[2 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 1][0], A_reg[0], B_reg[0], (1));
				        Cb_reg[3 + 4 * 1][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 1][0], A_reg[0], B_reg[0], (1));
					  Cb_reg[0 + 4 * 2][0] = vfmaq_laneq_f32(Cb_reg[0 + 4 * 2][0], A_reg[0], B_reg[0], (2));
					    Cb_reg[1 + 4 * 2][0] = vfmaq_laneq_f32(Cb_reg[1 + 4 * 2][0], A_reg[0], B_reg[0], (2));
					      Cb_reg[2 + 4 * 2][0] = vfmaq_laneq_f32(Cb_reg[2 + 4 * 2][0], A_reg[0], B_reg[0], (2));
					        Cb_reg[3 + 4 * 2][0] = vfmaq_laneq_f32(Cb_reg[3 + 4 * 2][0], A_reg[0], B_reg[0], (2));
	}
	vst1q_f32(&Cb[(0 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[0 + 4 * 0][0]);
	vst1q_f32(&Cb[(1 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[1 + 4 * 0][0]);
	vst1q_f32(&Cb[(2 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[2 + 4 * 0][0]);
	vst1q_f32(&Cb[(3 + 4 * 0) * (4) + (4 * 0) * (1)], Cb_reg[3 + 4 * 0][0]);
	vst1q_f32(&Cb[(0 + 4 * 1) * (4) + (4 * 0) * (1)], Cb_reg[0 + 4 * 1][0]);
	vst1q_f32(&Cb[(1 + 4 * 1) * (4) + (4 * 0) * (1)], Cb_reg[1 + 4 * 1][0]);
	vst1q_f32(&Cb[(2 + 4 * 1) * (4) + (4 * 0) * (1)], Cb_reg[2 + 4 * 1][0]);
	vst1q_f32(&Cb[(3 + 4 * 1) * (4) + (4 * 0) * (1)], Cb_reg[3 + 4 * 1][0]);
	vst1q_f32(&Cb[(0 + 4 * 2) * (4) + (4 * 0) * (1)], Cb_reg[0 + 4 * 2][0]);
	vst1q_f32(&Cb[(1 + 4 * 2) * (4) + (4 * 0) * (1)], Cb_reg[1 + 4 * 2][0]);
	vst1q_f32(&Cb[(2 + 4 * 2) * (4) + (4 * 0) * (1)], Cb_reg[2 + 4 * 2][0]);
	vst1q_f32(&Cb[(3 + 4 * 2) * (4) + (4 * 0) * (1)], Cb_reg[3 + 4 * 2][0]);
	for (int cj = 0; cj < 12; cj++) {
		  for (int ci = 0; ci < 4; ci++) {
			      C[(cj) * (4) + (ci) * (1)] = Cb[(cj) * (4) + (ci) * (1)];
			        }
	}
	free(Cb);
}





