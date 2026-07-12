#pragma once

#include "stage1_mamba2_payload.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <vector>

namespace fhemamba::stage1 {

inline constexpr double kPlaintextCoefficientFloor = 1e-8;
inline constexpr int kBabyStepIn = 64;
inline constexpr int kBabyStepOut = 64;

struct ReplicatedShape {
  int replicas = 1;
  int window = 0;
  int reps = 0;         // window / n input tiles per window
  int per_replica = 0;  // ceil(n / r) diagonals (= masks = encodes)
  int baby_step = 1;    // 1 = direct group rotations; >1 = true BSGS
};


struct PackingDims {
  int batch = 0;
  int group_count = 0;   // number of state ciphertexts (3)
  int group_heads = 0;   // heads per state ciphertext (8)
  int group_block = 0;   // group_heads * head_dim (512)
  int xbc0 = 0;          // start of xBC inside proj layout (d_inner)
  int dt0 = 0;           // start of dt inside proj layout (d_inner + conv_dim)
  int b_base = 0;        // start of B inside packed conv layout (d_inner)
  int c_base = 0;        // start of C inside packed conv layout (d_inner + state)
};


auto resolve_replicated_shape(int output_dim, int input_dim, int batch,
                              int force_r) -> ReplicatedShape;
auto replicated_bsgs_mask(const std::vector<double>& weights, int output_dim,
                          int input_dim, int k,
                          const ReplicatedShape& shape, int batch_size)
    -> std::vector<double>;
auto replicated_bsgs_pre_mask(const std::vector<double>& weights,
                              int output_dim, int input_dim, int k,
                              const ReplicatedShape& shape, int batch_size)
    -> std::vector<double>;
auto python_mod(int value, int modulus) -> int;
auto slot_bsgs_giant_with_zero(int input_dim, int output_dim, int baby_step)
    -> std::vector<int>;
auto derive_packing(const M1Payload& payload, int batch) -> PackingDims;
auto int_log2(int value) -> int;
auto required_rotations(const M1Payload& payload, const PackingDims& dims,
                        const ReplicatedShape& rep_in,
                        const ReplicatedShape& rep_out,
                        bool replicated_state_blocks = false)
    -> std::vector<int32_t>;
auto naf_steps(int value) -> std::vector<int>;
void verify_naf(const std::vector<int32_t>& indices);
auto rotation_frequencies(const M1Payload& payload, const PackingDims& dims,
                          int layers, int streams, int stream_stride,
                          const ReplicatedShape& rep_in,
                          const ReplicatedShape& rep_out,
                          bool replicated_state_blocks = false)
    -> std::map<int32_t, double>;
auto rotation_key_gib_estimate(int ring_dim, int multiplicative_depth)
    -> double;

}  // namespace fhemamba::stage1
