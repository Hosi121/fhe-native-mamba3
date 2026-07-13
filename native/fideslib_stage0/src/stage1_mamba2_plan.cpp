#include "stage1_mamba2_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace fhemamba::stage1 {

auto resolve_replicated_shape(int output_dim, int input_dim, int batch, int force_r)
    -> ReplicatedShape {
  if (output_dim <= 0 || input_dim <= 0 || batch <= 0 || force_r < 0) {
    throw std::invalid_argument("replicated BSGS dimensions must be positive");
  }
  ReplicatedShape shape;
  const int window =
      input_dim * ((output_dim + input_dim + input_dim - 1) / input_dim);
  int replicas = batch / window;
  if (force_r > 0) {
    replicas = std::min(replicas, force_r);
  }
  if (replicas <= 1) {
    return shape;  // does not fit or no benefit: legacy path
  }
  shape.replicas = replicas;
  shape.window = window;
  shape.reps = window / input_dim;
  shape.per_replica = (input_dim + replicas - 1) / replicas;
  return shape;
}

auto resolve_interleaved_replicated_shape(int output_dim, int input_dim,
                                          int batch, int force_r)
    -> ReplicatedShape {
  if (output_dim <= 0 || input_dim <= 0 || batch <= 0 || force_r < 0) {
    throw std::invalid_argument(
        "interleaved replicated BSGS dimensions must be positive");
  }
  const int maximum = force_r > 0
                          ? std::min(batch / input_dim - 1, force_r)
                          : batch / input_dim - 1;
  for (int replicas = maximum; replicas > 1; --replicas) {
    const int required = output_dim + replicas - 1;
    const int window =
        input_dim * ((required + input_dim - 1) / input_dim);
    if ((replicas + 1) * window > batch) {
      continue;
    }
    ReplicatedShape shape;
    shape.replicas = replicas;
    shape.window = window;
    shape.reps = window / input_dim;
    shape.per_replica = (input_dim + replicas - 1) / replicas;
    shape.guard_windows = 1;
    return shape;
  }
  return resolve_replicated_shape(output_dim, input_dim, batch, force_r);
}

// Combined per-k mask (one encoded plaintext serves all replicas).
auto replicated_bsgs_mask(const std::vector<double>& weights, int output_dim, int input_dim,
                          int k, const ReplicatedShape& shape, int batch_size)
    -> std::vector<double> {
  if (output_dim <= 0 || input_dim <= 0 || k < 0 || batch_size <= 0 ||
      weights.size() != static_cast<std::size_t>(output_dim) * input_dim ||
      shape.replicas <= 1 || shape.window <= 0 || shape.per_replica <= 0 ||
      (shape.replicas + shape.guard_windows) * shape.window > batch_size ||
      k >= shape.per_replica) {
    throw std::invalid_argument("invalid replicated BSGS mask geometry");
  }
  std::vector<double> mask(static_cast<std::size_t>(batch_size), 0.0);
  for (int j = 0; j < shape.replicas; ++j) {
    const int d = j + k * shape.replicas;
    if (d >= input_dim) {
      continue;
    }
    for (int i = 0; i < output_dim; ++i) {
      const double value =
          weights[static_cast<std::size_t>(i) * input_dim + ((i + d) % input_dim)];
      mask[static_cast<std::size_t>(j * shape.window + i + j)] =
          std::abs(value) < kPlaintextCoefficientFloor ? 0.0 : value;
    }
  }
  return mask;
}

auto replicated_bsgs_pre_mask(const std::vector<double>& weights, int output_dim,
                              int input_dim, int k,
                              const ReplicatedShape& shape, int batch_size)
    -> std::vector<double> {
  auto mask = replicated_bsgs_mask(weights, output_dim, input_dim, k, shape,
                                   batch_size);
  if (shape.baby_step <= 1) {
    return mask;
  }
  const int giant = (k / shape.baby_step) * shape.baby_step * shape.replicas;
  if (giant == 0) {
    return mask;
  }
  std::vector<double> shifted(mask.size(), 0.0);
  for (int slot = 0; slot < batch_size; ++slot) {
    shifted[static_cast<std::size_t>((slot + giant) % batch_size)] =
        mask[static_cast<std::size_t>(slot)];
  }
  return shifted;
}

// Rotation indices for one replicated matmul: input self-extension, window
// fill, per-k diagonal rolls, and the fold (doubling strides when r is a
// power of two, sequential otherwise) — all NAF-composable.
void insert_replicated_rotations(std::set<int32_t>& rotations, int input_dim,
                                 const ReplicatedShape& shape) {
  for (int t = 1; t < shape.reps; ++t) {
    rotations.insert(static_cast<int32_t>(-t * input_dim));
  }
  for (int j = 1; j < shape.replicas + shape.guard_windows; ++j) {
    rotations.insert(static_cast<int32_t>(-j * shape.window));
  }
  if (shape.baby_step > 1) {
    for (int baby = 1; baby < shape.baby_step; ++baby) {
      rotations.insert(static_cast<int32_t>(baby * shape.replicas));
    }
    const int giant_count =
        (shape.per_replica + shape.baby_step - 1) / shape.baby_step;
    for (int giant = 1; giant < giant_count; ++giant) {
      rotations.insert(
          static_cast<int32_t>(giant * shape.baby_step * shape.replicas));
    }
  } else {
    for (int k = 1; k < shape.per_replica; ++k) {
      rotations.insert(static_cast<int32_t>(k * shape.replicas));
    }
  }
  if ((shape.replicas & (shape.replicas - 1)) == 0) {
    for (int step = shape.window + 1; step < shape.replicas * (shape.window + 1);
         step *= 2) {
      rotations.insert(static_cast<int32_t>(step));  // fold rotates LEFT
    }
  } else {
    for (int j = 1; j < shape.replicas; ++j) {
      rotations.insert(static_cast<int32_t>(j * (shape.window + 1)));
    }
  }
}


auto python_mod(int value, int modulus) -> int {
  if (modulus <= 0) {
    throw std::invalid_argument("modulus must be positive");
  }
  const int result = value % modulus;
  return result < 0 ? result + modulus : result;
}

auto slot_bsgs_giant_with_zero(int input_dim, int output_dim, int baby_step) -> std::vector<int> {
  if (input_dim <= 0 || output_dim <= 0 || baby_step <= 0) {
    throw std::invalid_argument("BSGS dimensions and baby step must be positive");
  }
  std::set<int> values;
  const int min_offset = -(output_dim - 1);
  const int max_offset = input_dim - 1;
  for (int offset = min_offset; offset <= max_offset; ++offset) {
    values.insert(offset - python_mod(offset, baby_step));
  }
  return {values.begin(), values.end()};
}

auto slot_bsgs_rotations(int input_dim, int output_dim, int baby_step) -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (int baby = 1; baby < baby_step; ++baby) {
    rotations.insert(static_cast<int32_t>(baby));
  }
  for (const int giant : slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step)) {
    if (giant != 0) {
      rotations.insert(static_cast<int32_t>(giant));
    }
  }
  return {rotations.begin(), rotations.end()};
}


auto derive_packing(const M1Payload& payload, int batch) -> PackingDims {
  if (batch <= 0) {
    throw std::invalid_argument("batch must be positive");
  }
  PackingDims dims;
  dims.batch = batch;
  const int per_head = payload.head_dim * payload.state_size;
  if (per_head <= 0 || batch % per_head != 0) {
    throw std::runtime_error("batch is not a multiple of head_dim * state_size");
  }
  dims.group_heads = batch / per_head;
  if (dims.group_heads <= 0 || payload.num_heads % dims.group_heads != 0) {
    throw std::runtime_error("num_heads is not divisible by heads-per-group");
  }
  dims.group_count = payload.num_heads / dims.group_heads;
  dims.group_block = dims.group_heads * payload.head_dim;
  dims.xbc0 = payload.d_inner;
  dims.dt0 = payload.d_inner + payload.conv_dim;
  dims.b_base = payload.d_inner;
  dims.c_base = payload.d_inner + payload.state_size;
  auto power_of_two = [](int value) { return value > 0 && (value & (value - 1)) == 0; };
  if (!power_of_two(payload.state_size) || !power_of_two(payload.head_dim) ||
      !power_of_two(dims.group_block) || !power_of_two(batch)) {
    throw std::runtime_error("state_size/head_dim/group_block/batch must be powers of two");
  }
  return dims;
}

auto int_log2(int value) -> int {
  if (value <= 0) {
    throw std::invalid_argument("log2 input must be positive");
  }
  int log = 0;
  while ((1 << log) < value) {
    ++log;
  }
  return log;
}

auto required_rotations(const M1Payload& payload, const PackingDims& dims,
                        const ReplicatedShape& rep_in, const ReplicatedShape& rep_out,
                        bool replicated_state_blocks)
    -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  auto insert_all = [&](const std::vector<int32_t>& values) {
    rotations.insert(values.begin(), values.end());
  };
  // in_proj / out_proj: legacy rectangular BSGS babies/giants, or the
  // replicated layout's extension/fill/roll/fold indices.
  if (rep_in.replicas > 1) {
    insert_replicated_rotations(rotations, payload.d_model, rep_in);
  } else {
    insert_all(slot_bsgs_rotations(payload.d_model, payload.proj_dim, kBabyStepIn));
  }
  if (rep_out.replicas > 1) {
    insert_replicated_rotations(rotations, payload.d_inner, rep_out);
  } else {
    insert_all(slot_bsgs_rotations(payload.d_inner, payload.d_model, kBabyStepOut));
  }
  // Full-batch rotate-sum for the two RMS variances (broadcast total to all slots).
  for (int k = 0; k < int_log2(dims.batch); ++k) {
    rotations.insert(static_cast<int32_t>(1 << k));
  }
  // Rotate-add doubling fills: within a 512 block (B/C), across p (dt/decay,
  // subset), across n (x/dt/decay expands).
  for (int k = 0; k < int_log2(dims.group_block); ++k) {
    rotations.insert(static_cast<int32_t>(-(1 << k)));
  }
  for (int k = 0; k < int_log2(payload.state_size); ++k) {
    rotations.insert(static_cast<int32_t>(-(dims.group_block << k)));
  }
  // conv output shift from proj coordinates into the packed layout.
  rotations.insert(static_cast<int32_t>(dims.xbc0));
  // x expand: bring group block to slot 0.
  for (int g = 1; g < dims.group_count; ++g) {
    rotations.insert(static_cast<int32_t>(dims.group_block * g));
  }
  // Readout rotate-sum and group placement.
  for (int k = 0; k < int_log2(payload.state_size); ++k) {
    rotations.insert(static_cast<int32_t>(dims.group_block << k));
  }
  for (int g = 1; g < dims.group_count; ++g) {
    rotations.insert(static_cast<int32_t>(-dims.group_block * g));
  }
  // B/C placement. The replicated schedule selects B+C once, shifts one
  // branch to slot zero, and copies it at (group_block-1)*2^k before keeping
  // block seeds. The legacy schedule masks each state element separately.
  const int stride = dims.group_block - 1;  // 511
  if (replicated_state_blocks) {
    rotations.insert(static_cast<int32_t>(dims.b_base));
    rotations.insert(static_cast<int32_t>(dims.c_base));
    for (int step = 1; step < payload.state_size; step *= 2) {
      rotations.insert(static_cast<int32_t>(-stride * step));
    }
  } else {
    const int giant_stride = dims.group_heads * stride;  // 4088
    for (const int base : {dims.b_base, dims.c_base}) {
      for (int b = 0; b < dims.group_heads; ++b) {
        const int baby = base - stride * b;
        if (baby != 0) {
          rotations.insert(static_cast<int32_t>(baby));
        }
      }
    }
    for (int a = 1; a < payload.state_size / dims.group_heads; ++a) {
      rotations.insert(static_cast<int32_t>(-giant_stride * a));
    }
  }
  // dt/decay head placement: rotate whole dt block of group g to slot 0, then
  // move head h_local from slot h_local to slot h_local*head_dim.
  for (int g = 0; g < dims.group_count; ++g) {
    rotations.insert(static_cast<int32_t>(dims.dt0 + dims.group_heads * g));
  }
  for (int h = 1; h < dims.group_heads; ++h) {
    rotations.insert(static_cast<int32_t>(-(payload.head_dim - 1) * h));
  }
  rotations.erase(0);
  return {rotations.begin(), rotations.end()};
}

// ---------------------------------------------------------------------------
// Composite rotation keys (design: fhemamba/src/fhemamba/rotation_keys.py).
// naf_steps is an exact port of rotation_keys.naf: the signed powers of two
// (non-adjacent form) summing to value, e.g. 28 -> {32, -4}. Every rotation
// then decomposes into applications of base keys +-2^k. EvalRotate is a pure
// key switch in the proven kernels (no rescale, never level-aligned), so a
// k-step composition consumes zero levels and the ledger is unchanged.
// ---------------------------------------------------------------------------

auto naf_steps(int value) -> std::vector<int> {
  std::vector<int> steps;
  if (value == 0) {
    return steps;
  }
  long long v = value;
  int k = 0;
  while (v != 0) {
    if ((v & 1) != 0) {
      const long long digit = 2 - (v & 3);  // +-1, zeroing the low two bits
      steps.push_back(static_cast<int>(digit << k));
      v -= digit;
    }
    v >>= 1;
    ++k;
  }
  return steps;
}

// Startup unit test over the exact index set the kernel will use.
void verify_naf(const std::vector<int32_t>& indices) {
  for (const int index : indices) {
    long long sum = 0;
    for (const int step : naf_steps(index)) {
      const long long magnitude = step < 0 ? -static_cast<long long>(step) : step;
      if (magnitude == 0 || (magnitude & (magnitude - 1)) != 0) {
        throw std::runtime_error("NAF produced a non-power step for index " +
                                 std::to_string(index));
      }
      sum += step;
    }
    if (sum != index) {
      throw std::runtime_error("NAF decomposition does not sum to index " +
                               std::to_string(index));
    }
  }
}

// Static per-token rotation frequencies (applications per token across the
// loaded layers), keyed by exactly the required_rotations index set: the
// per-site counts follow the same generators the evaluation uses, and the
// planner asserts key-set equality so the two cannot drift apart.
auto rotation_frequencies(const M1Payload& payload, const PackingDims& dims, int layers,
                          int streams, int stream_stride,
                          const ReplicatedShape& rep_in, const ReplicatedShape& rep_out,
                          bool replicated_state_blocks)
    -> std::map<int32_t, double> {
  if (layers <= 0 || streams <= 0 || stream_stride <= 0) {
    throw std::invalid_argument("rotation frequency geometry must be positive");
  }
  std::map<int32_t, double> freq;
  const double L = layers;
  const double S = streams;
  const double G = dims.group_count;
  auto add = [&](int index, double count) {
    if (index != 0) {
      freq[static_cast<int32_t>(index)] += count;
    }
  };
  // BSGS families: legacy babies/giants, or replicated ext/fill/roll/fold
  // (each applied once per matmul per token-layer).
  auto add_replicated = [&](int input_dim, const ReplicatedShape& shape) {
    for (int t = 1; t < shape.reps; ++t) {
      add(-t * input_dim, L);
    }
    for (int j = 1; j < shape.replicas + shape.guard_windows; ++j) {
      add(-j * shape.window, L);
    }
    if (shape.baby_step > 1) {
      for (int baby = 1; baby < shape.baby_step; ++baby) {
        add(baby * shape.replicas, L);
      }
      const int giant_count =
          (shape.per_replica + shape.baby_step - 1) / shape.baby_step;
      for (int giant = 1; giant < giant_count; ++giant) {
        add(giant * shape.baby_step * shape.replicas, L);
      }
    } else {
      for (int k = 1; k < shape.per_replica; ++k) {
        add(k * shape.replicas, L);
      }
    }
    if ((shape.replicas & (shape.replicas - 1)) == 0) {
      for (int step = shape.window + 1; step < shape.replicas * (shape.window + 1);
           step *= 2) {
        add(step, L);
      }
    } else {
      for (int j = 1; j < shape.replicas; ++j) {
        add(j * (shape.window + 1), L);
      }
    }
  };
  if (rep_in.replicas > 1) {
    add_replicated(payload.d_model, rep_in);
  } else {
    for (int baby = 1; baby < kBabyStepIn; ++baby) {
      add(baby, L);
    }
    for (const int giant : slot_bsgs_giant_with_zero(payload.d_model, payload.proj_dim,
                                                     kBabyStepIn)) {
      add(giant, L);
    }
  }
  if (rep_out.replicas > 1) {
    add_replicated(payload.d_inner, rep_out);
  } else {
    for (int baby = 1; baby < kBabyStepOut; ++baby) {
      add(baby, L);
    }
    for (const int giant : slot_bsgs_giant_with_zero(payload.d_inner, payload.d_model,
                                                     kBabyStepOut)) {
      add(giant, L);
    }
  }
  // RMS variance reductions: two per layer (block + gated).
  const int sum_bits = streams == 1 ? int_log2(dims.batch) : int_log2(stream_stride);
  for (int k = 0; k < sum_bits; ++k) {
    add(1 << k, 2.0 * L);
  }
  if (streams > 1) {
    for (int k = 0; k < int_log2(stream_stride); ++k) {
      add(-(1 << k), 2.0 * L);  // in-stride broadcast after the base mask
    }
  }
  // Broadcast doubling fills.
  for (int k = 0; k < int_log2(dims.group_block); ++k) {
    add(-(1 << k), 2.0 * S * L);  // B/C block fills
  }
  for (int k = 0; k < int_log2(payload.head_dim); ++k) {
    add(-(1 << k), 2.0 * G * S * L);  // dt/decay head fills
  }
  for (int k = 0; k < int_log2(payload.state_size); ++k) {
    add(-(dims.group_block << k), 3.0 * G * S * L);  // x/dt/decay across n
  }
  // conv shift into the packed layout.
  add(dims.xbc0, L);
  // x expand group shifts, readout sums, group placement.
  for (int g = 1; g < dims.group_count; ++g) {
    add(dims.group_block * g, S * L);
    add(-dims.group_block * g, S * L);
  }
  for (int k = 0; k < int_log2(payload.state_size); ++k) {
    add(dims.group_block << k, G * S * L);
  }
  // B/C placement: logarithmic replication or legacy baby/giant selection.
  const int stride = dims.group_block - 1;
  if (replicated_state_blocks) {
    add(dims.b_base, S * L);
    add(dims.c_base, S * L);
    for (int step = 1; step < payload.state_size; step *= 2) {
      add(-stride * step, 2.0 * S * L);
    }
  } else {
    const int giant_stride = dims.group_heads * stride;
    for (const int base : {dims.b_base, dims.c_base}) {
      for (int b = 0; b < dims.group_heads; ++b) {
        add(base - stride * b, S * L);
      }
    }
    for (int a = 1; a < payload.state_size / dims.group_heads; ++a) {
      add(-giant_stride * a, 2.0 * S * L);
    }
  }
  // dt/decay head placement.
  for (int g = 0; g < dims.group_count; ++g) {
    add(dims.dt0 + dims.group_heads * g, 2.0 * S * L);
  }
  for (int h = 1; h < dims.group_heads; ++h) {
    add(-(payload.head_dim - 1) * h, 2.0 * G * S * L);
  }
  return freq;
}

// Hybrid key-switch key size estimate: dnum digits x 2 polys x ring x
// (Q + Q/dnum) towers x 8 B. Calibrated 0.39 GiB/key at 2^17/d48/dnum3.
constexpr int kRotationKeyDnumEstimate = 3;

auto rotation_key_gib_estimate(int ring_dim, int multiplicative_depth) -> double {
  if (ring_dim <= 0 || multiplicative_depth <= 0) {
    throw std::invalid_argument("rotation-key geometry must be positive");
  }
  const int q_towers = multiplicative_depth + 1;
  const int p_towers =
      (q_towers + kRotationKeyDnumEstimate - 1) / kRotationKeyDnumEstimate;
  return static_cast<double>(kRotationKeyDnumEstimate) * 2.0 * ring_dim *
         (q_towers + p_towers) * 8.0 / (1024.0 * 1024.0 * 1024.0);
}

}  // namespace fhemamba::stage1
