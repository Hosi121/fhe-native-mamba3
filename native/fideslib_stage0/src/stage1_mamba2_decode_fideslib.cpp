// Full-width Mamba-2 decode on FIDESlib/CKKS GPU.
//
// M1 (--input DIR): one layer (mamba2-130m layer 0), looped over tokens.
// M2 (--input-chain DIR): all (or --max-layers) layers chained per token with
// the residual as a ciphertext handoff, per-layer persistent ciphertext state
// and conv FIFO, an auto-bootstrap headroom policy
// (--auto-bootstrap-headroom), and the final RMSNorm when the full chain is
// loaded.
//
// Implements the verified per-layer circuit of fhemamba/src/fhemamba/lowering.py
// looped over tokens with real ciphertext state carry:
//   block RMSNorm -> in_proj BSGS -> mask splits -> conv FIFO + bias ->
//   conv SiLU (Chebyshev) -> dt softplus^2 -> decay exp(dt*A) -> slot expands ->
//   state update (3 head-group state ciphertexts) -> readout -> +D*x ->
//   gate SiLU * y -> gated RMSNorm -> out_proj BSGS -> residual add.
// State and conv FIFO stay ciphertext across tokens. Fixed-vector mode has
// zero intermediate decrypts; autoregressive client-loop mode intentionally
// decrypts each completed final_norm for client-side token selection.
//
// Payload: fhemamba m1 export (meta.json + <name>.bin float32-LE row-major),
// produced by fhemamba/src/fhemamba/m1_payload.py.
//
// Packing:
//   packed layout: vectors in slots 0..k-1 (hidden 768; in_proj output 3352 =
//     gate 1536 | xBC 1792 | dt 24; conv out 1792 = x 1536 | B 128 | C 128).
//   state layout: 3 ciphertexts, one per head group g in {0,1,2} (8 heads
//     each); slot index = n*512 + h_local*64 + p (n<128, h_local<8, p<64).
//
// Level ledger (validated by slot-level simulation against the real payload,
// default poly degrees: rms 47 + 4 Newton, conv 96, gate 64, dt 64, exp 24,
// gated const-Newton 14): the uncut circuit needs ~61 levels for token 0 from
// fresh input and 64+tokens with state carry — which can NEVER run on
// FIDESlib v2.1.0 (MAXP=64 tower cap: empirical max depth 50 at scale 40, 44
// at scale 59). The kernel therefore runs at scale 59 / depth 44 with
// MID-CIRCUIT bootstrap checkpoints (segment map in the pre-run log and JSON
// "segment_requirements"): after in_proj (input segment 17), after conv_silu
// (8), after dt (9), after decay (7 + per-layer squarings, up to 21), after
// the state update (<=5), before the gated norm and out_proj (<=4), plus a
// per-iteration Newton iterate refresh (2/iter). Bootstrap output sits at
// GetLevel 18 (dgx-measured), leaving a 26-level segment budget; every
// segment above fits. Warm GPU bootstrap is 12-14 ms.

#include <fideslib.hpp>

#include <CKKS/Ciphertext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <openfhe.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include "stage1_mamba2_config.hpp"
#include "fideslib_handoff.hpp"
#include "stage1_mamba2_artifact.hpp"
#include "stage1_mamba2_depth.hpp"
#include "stage1_mamba2_payload.hpp"
#include "stage1_mamba2_plan.hpp"
#include "stage1_mamba2_process.hpp"

#include <algorithm>
#include <any>
#include <chrono>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

using namespace fideslib;
namespace fs = std::filesystem;

// FIDESlib's bootstrap uses several CUDA streams, while its release build
// only synchronizes the device at the end when internal PRINT diagnostics are
// enabled. Its own tests bracket bootstrap calls with an explicit device
// synchronization; cuda_runtime.h is included through the serialization
// bridge headers above.

namespace {

using fhemamba::stage1::Config;
using fhemamba::stage1::ChainPayload;
using fhemamba::stage1::M1Payload;
using fhemamba::stage1::parse_args;
using fhemamba::stage1::json_escape;
using fhemamba::stage1::count_server_secret_files;
using fhemamba::stage1::cheb_baby_size;
using fhemamba::stage1::cheb_ps_depth;
using fhemamba::stage1::derive_packing;
using fhemamba::stage1::DepthEstimate;
using fhemamba::stage1::estimate_levels;
using fhemamba::stage1::handoff_paths;
using fhemamba::stage1::int_log2;
using fhemamba::stage1::kBabyStepIn;
using fhemamba::stage1::kBabyStepOut;
using fhemamba::stage1::kAssumedBootstrapOutputLevel;
using fhemamba::stage1::kChebCoefficientFloor;
using fhemamba::stage1::kNewtonSegmentEstimate;
using fhemamba::stage1::kPlaintextCoefficientFloor;
using fhemamba::stage1::naf_steps;
using fhemamba::stage1::PackingDims;
using fhemamba::stage1::payload_file_exists;
using fhemamba::stage1::prepare_client_handoff;
using fhemamba::stage1::read_chain_payload;
using fhemamba::stage1::read_m1_payload;
using fhemamba::stage1::replicated_bsgs_mask;
using fhemamba::stage1::replicated_bsgs_pre_mask;
using fhemamba::stage1::ReplicatedShape;
using fhemamba::stage1::required_rotations;
using fhemamba::stage1::resolve_interleaved_replicated_shape;
using fhemamba::stage1::resolve_replicated_shape;
using fhemamba::stage1::rotation_frequencies;
using fhemamba::stage1::rotation_key_gib_estimate;
using fhemamba::stage1::require_same_layer_dims;
using fhemamba::stage1::require_server_has_no_secret_files;
using fhemamba::stage1::write_artifact_prefix;
using fhemamba::stage1::write_double_map_json;
using fhemamba::stage1::write_double_vector_json;
using fhemamba::stage1::write_int_map_json;
using fhemamba::stage1::write_int_set_json;
using fhemamba::stage1::write_int_vector_json;
using fhemamba::stage1::write_payload;
using fhemamba::stage1::write_runtime_failure_payload;
using fhemamba::stage1::python_mod;
using fhemamba::stage1::slot_bsgs_giant_with_zero;
using fhemamba::stage1::verify_naf;
using fhemamba::stage1::verify_cheb_ps_host;
using fhemamba::handoff::copy_context_device_metadata;
using fhemamba::handoff::deserialize_ciphertext;
using fhemamba::handoff::require_serialized;
using fhemamba::handoff::serialize_ciphertext;
using fhemamba::handoff::serialize_context;

// dgx-measured FIDESlib v2.1.0 bootstrap output: GetLevel() == 18 after
// EvalBootstrap at ring 131072/depth 44/scale 59 and ring 65536/depth 28/
// scale 59 (fhemamba/results/dgx/bootstrap_probe_*.json, which report the
// REMAINING levels: 26 and 10). At the default depth 44 that leaves a
// 26-level segment budget between mid-circuit refresh points. The refresh
// floor kMinBootstrapGain skips refreshes that cannot usefully lower a
// ciphertext (bootstrap output would be at or above its current level).
constexpr int kMinBootstrapGain = 8;
// Parallel miss-path encoding: uncached BSGS diagonals are encoded in look-
// ahead batches of encode_threads * this many entries, bounding the in-flight
// plaintext memory (8 threads * 8 entries ~= 1.4 GiB at ring 65536/d44).
constexpr int kEncodeBatchPerThread = 8;
// Number of diagonals compared serial-vs-parallel by the encode thread-safety
// self test that gates --encode-threads > 1.
constexpr int kEncodeSelfTestDiagonals = 32;


// Per-checkpoint message-magnitude bounds for the pre-bootstrap
// normalization. FIDESlib EvalBootstrap refresh error grows polynomially
// with magnitude on noiseScaleDeg-2 inputs (dgx GB10, 65536/d44/s59:
// |m|=1 -> 2.2e-5, 4 -> 9.4e-4, 16 -> 6.0e-2, 24 -> 0.20/NaN) while deg-1
// inputs tolerate |m|=24 at 2.4e-4. Values are raw --debug-decrypt
// measurements (token 0, mamba2-130m layer 0) where available, otherwise
// conservative fallbacks; the configurable margin multiplies them. The
// rescale-to-deg-1 step in maybe_bootstrap is the primary guard, so a
// moderate bound undershoot degrades gracefully instead of exploding.
auto checkpoint_bound(const std::string& what) -> double {
  static constexpr std::pair<const char*, double> kBounds[] = {
      {"final_norm_scaled", 1.0},  // final residual stays in normalized coordinates
      {"final_norm", 8.0},   // fallback (residual lineage)
      {"residual", 8.0},     // fallback; expected outputs |m| <= ~2.4
      {"output", 8.0},       // fallback; calibrated per layer in new payloads
      {"conv_silu", 9.5},    // measured 9.20
      // Carried-forward lineages (state, conv FIFO): measured |m| maxima from
      // --debug-decrypt telemetry (mamba2-130m layer 0). These are the true
      // bound, not a fallback; the tighter state margin multiplies them. The
      // per-layer |m| ranges O(1-25), so where a layer's state is small the
      // fixed 24 over-scales -- re-measure per layer with --debug-decrypt to
      // tighten further (bound must stay >= that layer's max |m|).
      {"state_post", 24.0},  // updated state; y readout of state measured 23.85
      {"state", 24.0},       // carried state pre-update (matches state_post)
      {"fifo", 6.0},         // FIFO holds proj-lineage cts; proj measured 5.73
      {"proj", 6.0},         // measured 5.73
      {"decay", 1.0},        // measured 1.00 (decay in [0, 1])
      {"dt", 0.5},           // measured 0.103
      {"gated_poly_input", 1.0},  // affine-mapped gated variance
      {"gated_variance", 1.0},  // -0.5 * V * initial_guess^2; calibrated <= 0.125
      {"y_out", 24.0},       // same lineage as y
      {"y", 24.0},           // measured 23.85 (check after "decay": "y" substring)
      {"newton", 4.0},       // fallback; rsqrt iterates (rms measured 0.13)
  };
  for (const auto& [key, bound] : kBounds) {
    if (what.find(key) != std::string::npos) {
      return bound;
    }
  }
  return 24.0;
}

// Carried-forward (cross-token) lineages get the tighter state margin; every
// other checkpoint is a transient per-layer activation.
auto is_carried_checkpoint(const std::string& what) -> bool {
  return what.find("state") != std::string::npos || what.find("fifo") != std::string::npos;
}

struct OperationCounts {
  int rotations = 0;  // logical rotations (mode-independent)
  int rotations_direct = 0;          // served by a direct key
  int rotations_composite_steps = 0;  // NAF key applications for the rest
  int ct_pt_mul = 0;
  int ct_ct_mul = 0;
  int adds = 0;
  int unity_level_align_muls = 0;
  int direct_level_align_drops = 0;
  int bootstraps = 0;
};

using BabyRotationCache = std::map<int, Ciphertext<DCRTPoly>>;
// Every EvalRotate goes through this choke point (built in main): direct key
// if present, NAF-composed base-key applications otherwise.
using RotateFn = std::function<Ciphertext<DCRTPoly>(const Ciphertext<DCRTPoly>&, int)>;

auto now() -> std::chrono::steady_clock::time_point { return std::chrono::steady_clock::now(); }

auto seconds_since(std::chrono::steady_clock::time_point start) -> double {
  return std::chrono::duration<double>(now() - start).count();
}

auto add_counts(OperationCounts lhs, const OperationCounts& rhs) -> OperationCounts {
  lhs.rotations += rhs.rotations;
  lhs.rotations_direct += rhs.rotations_direct;
  lhs.rotations_composite_steps += rhs.rotations_composite_steps;
  lhs.ct_pt_mul += rhs.ct_pt_mul;
  lhs.ct_ct_mul += rhs.ct_ct_mul;
  lhs.adds += rhs.adds;
  lhs.unity_level_align_muls += rhs.unity_level_align_muls;
  lhs.direct_level_align_drops += rhs.direct_level_align_drops;
  lhs.bootstraps += rhs.bootstraps;
  return lhs;
}

auto subtract_counts(const OperationCounts& after, const OperationCounts& before)
    -> OperationCounts {
  return OperationCounts{
      .rotations = after.rotations - before.rotations,
      .rotations_direct = after.rotations_direct - before.rotations_direct,
      .rotations_composite_steps =
          after.rotations_composite_steps - before.rotations_composite_steps,
      .ct_pt_mul = after.ct_pt_mul - before.ct_pt_mul,
      .ct_ct_mul = after.ct_ct_mul - before.ct_ct_mul,
      .adds = after.adds - before.adds,
      .unity_level_align_muls =
          after.unity_level_align_muls - before.unity_level_align_muls,
      .direct_level_align_drops =
          after.direct_level_align_drops - before.direct_level_align_drops,
      .bootstraps = after.bootstraps - before.bootstraps,
  };
}


auto resolve_security(const std::string& value) -> SecurityLevel {
  if (value == "128-classic") {
    return HEStd_128_classic;
  }
  return HEStd_NotSet;
}

auto resolve_secret_key_dist(const std::string& value) -> SecretKeyDist {
  if (value == "uniform-ternary") {
    return UNIFORM_TERNARY;
  }
  if (value == "sparse-encapsulated") {
    return fideslib::SPARSE_ENCAPSULATED;
  }
  return SPARSE_TERNARY;
}

// Per-layer host-side plaintext constants (folded weights, masks, poly
// affines); built once per layer, reused every token.
struct LayerPlan {
  double eps_block = 0.0;
  double eps_gated = 0.0;
  double b_gate = 0.0;
  double b_exp = 0.0;
  double b_rms = 0.0;
  double a_rms_v = 0.0;
  double gated_guess_v = 0.0;
  double a_gated_v = 0.0;
  double b_gated = 0.0;
  double gated_damping_mean = 0.0;
  int rms_iterations = 0;
  int gated_iterations = 0;
  int exp_squarings = 0;
  std::vector<double> rms_coeffs;
  std::vector<double> conv_coeffs;
  std::vector<double> gate_coeffs;
  std::vector<double> dt_coeffs;
  std::vector<double> exp_coeffs;
  std::vector<double> gated_coeffs;
  std::vector<double> in_w_folded;
  std::vector<double> out_w_folded;
  std::vector<std::vector<double>> conv_tap_masks;
  std::vector<double> conv_const;
  std::vector<double> gate_mask;
  std::vector<double> dt_mask;
  std::vector<double> dt_const;
  std::vector<double> a_vec;
  std::vector<double> d_vec;
  std::vector<double> test_layer_output;  // (tokens, d_model) row-major
  // Mid-circuit bootstrap checkpoint requirements: levels the named value
  // still consumes before its lineage reaches the next checkpoint.
  int req_residual = 0;    // layer input -> proj
  int req_proj = 0;        // proj -> conv/gate/dt polynomial outputs
  int req_fifo = 0;        // FIFO entry -> conv_silu output
  int req_conv = 0;        // conv_silu output -> expands/update/readout
  int req_dt = 0;          // dt -> decay polynomial (+ squarings)
  int req_decay = 0;       // decay -> expand + state multiply
  int req_state_pre = 0;   // carried state -> update + readout tail
  int req_state_tail = 0;  // updated state -> readout -> y
  int req_y = 0;           // y -> variance + out_proj entry
  int req_out = 0;         // y at out_proj -> final multiply + residual add
  // Per-layer measured carried-lineage bounds (< 0 -> generic fallback).
  double state_abs_max = -1.0;
  std::vector<double> state_group_abs_max;
  std::vector<double> state_group_scales;
  std::vector<std::vector<double>> normalized_state_x_masks;
  std::vector<std::vector<double>> normalized_state_readout_masks;
  double fifo_abs_max = -1.0;
  double y_scale = 1.0;
  std::map<std::string, double> checkpoint_abs_max;
  // Plaintext-cache wiring: key prefix ("L00." ...) for this layer's
  // token-invariant vectors, and encode-once BSGS diagonal tables (null when
  // the cache mode excludes them).
  std::string cache_prefix;
  const std::vector<Plaintext>* in_proj_table = nullptr;
  const std::vector<Plaintext>* out_proj_table = nullptr;
};

// Per-layer persistent ciphertext state carried across tokens.
struct LayerRuntime {
  std::vector<Ciphertext<DCRTPoly>> state_cts;
  bool has_state = false;
  std::vector<Ciphertext<DCRTPoly>> conv_fifo;
};

// ---------------------------------------------------------------------------
// Input-replicated BSGS (authority: fhemamba/src/fhemamba/bsgs_layout.py).
// window = n * ceil((m+n)/n) (multiple of n, >= m+n: no read crosses a window
// boundary); r = batch/window identical replicas of the period-n input tile.
// Schedule (verified bitwise against the spec simulator): for each k <
// ceil(n/r), ONE global rotation by k*r and ONE combined mask that places
// W[i, (i+d) mod n] for replica j's diagonal d = j + k*r at slot
// j*window + i + j — the +j in-window shift is what lets identical replicas
// share a single roll per k; the fold therefore uses stride (window+1).
// Replication fill and fold are rotate/add only (0 levels); the matmul stays
// one ct-pt level, so the level ledger is unchanged.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// BSGS ct-pt matmul machinery (copied from stage1_rank_gate_fideslib.cpp).
// ---------------------------------------------------------------------------

// stride == batch_size reproduces the single-stream mask exactly. For
// multi-stream packing the diagonal placement is computed modulo the stream
// stride and tiled across all strides: every stream's elements sit at the
// same in-stride offsets, and the shared cyclic rotations move them
// identically, so ONE periodic plaintext serves all streams (requires
// output_dim <= stride, validated by the caller).
auto slot_bsgs_pre_mask(
    const std::vector<double>& weights,
    int input_dim,
    int output_dim,
    int batch_size,
    int stride,
    int giant,
    int offset) -> std::vector<double> {
  std::vector<double> mask(static_cast<size_t>(batch_size), 0.0);
  for (int output = 0; output < output_dim; ++output) {
    const int input = output + offset;
    if (input < 0 || input >= input_dim) {
      continue;
    }
    const int source_slot = python_mod(output + giant, stride);
    const double value = weights[static_cast<size_t>(output) * input_dim + input];
    const double floored = std::abs(value) < kPlaintextCoefficientFloor ? 0.0 : value;
    for (int copy = source_slot; copy < batch_size; copy += stride) {
      mask[static_cast<size_t>(copy)] = floored;
    }
  }
  return mask;
}

auto slot_bsgs_precompute_baby_rotations(
    const RotateFn& rotate_fn,
    const Ciphertext<DCRTPoly>& input_ct,
    int baby_step) -> BabyRotationCache {
  // Cache keys are LOGICAL rotation indices: any composite decomposition
  // happens inside rotate_fn, so the cache (and every consumer indexing it)
  // is transparent to the rotation-key plan.
  BabyRotationCache baby_ct;
  baby_ct[0] = input_ct;
  for (int baby = 1; baby < baby_step; ++baby) {
    baby_ct[baby] = rotate_fn(input_ct, baby);
  }
  return baby_ct;
}

// plain_table (optional): encode-once plaintexts for the token-invariant
// diagonal masks, indexed giant-major/baby-minor over the same enumeration
// (null entries fall back to on-the-fly encoding). cache_hits/cache_misses
// are only counted when a table is supplied. encode_threads > 1 pre-encodes
// the next uncached diagonals in look-ahead worker batches (bounded by
// encode_threads * kEncodeBatchPerThread in-flight plaintexts); the math and
// counters are identical to the serial path — only where and when the host
// NTT encode happens changes. Callers must have passed the encode
// thread-safety self test before requesting threads > 1.
auto slot_bsgs_linear_block0_from_babies(
    const CryptoContext<DCRTPoly>& cc,
    const RotateFn& rotate_fn,
    const BabyRotationCache& baby_ct,
    const std::vector<double>& weights,
    int input_dim,
    int output_dim,
    int baby_step,
    int batch_size,
    int& ct_pt_muls,
    int& adds,
    const std::vector<Plaintext>* plain_table,
    long long* cache_hits,
    long long* cache_misses,
    int encode_threads,
    int stride) -> Ciphertext<DCRTPoly> {
  Ciphertext<DCRTPoly> accumulator;
  bool has_accumulator = false;
  const auto giants = slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step);
  const auto total_positions = giants.size() * static_cast<std::size_t>(baby_step);
  auto in_persistent_table = [&](std::size_t table_index) {
    return plain_table != nullptr && table_index < plain_table->size() &&
           (*plain_table)[table_index];
  };
  // Look-ahead buffer for encode_threads > 1: a present-but-null entry marks
  // an all-zero mask (skip), mirroring the serial path's zero test.
  std::map<std::size_t, Plaintext> prefetched;
  auto prefetch_from = [&](std::size_t start_position) {
    struct PendingEncode {
      std::size_t position = 0;
      int giant = 0;
      int offset = 0;
    };
    std::vector<PendingEncode> pending;
    const auto batch_target =
        static_cast<std::size_t>(encode_threads) * kEncodeBatchPerThread;
    for (std::size_t position = start_position;
         position < total_positions && pending.size() < batch_target; ++position) {
      if (in_persistent_table(position) || prefetched.count(position) > 0) {
        continue;
      }
      const auto giant_index = position / static_cast<std::size_t>(baby_step);
      const int baby = static_cast<int>(position % static_cast<std::size_t>(baby_step));
      pending.push_back(PendingEncode{position, giants[giant_index],
                                      giants[giant_index] + baby});
    }
    if (pending.empty()) {
      return;
    }
    std::vector<Plaintext> results(pending.size());
    const auto worker_count = std::min<std::size_t>(
        static_cast<std::size_t>(encode_threads), pending.size());
    std::vector<std::exception_ptr> worker_errors(worker_count);
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers.emplace_back([&, worker]() {
        try {
          for (std::size_t job = worker; job < pending.size(); job += worker_count) {
            auto mask = slot_bsgs_pre_mask(weights, input_dim, output_dim, batch_size,
                                           stride, pending[job].giant, pending[job].offset);
            if (std::all_of(mask.begin(), mask.end(),
                            [](double value) { return value == 0.0; })) {
              continue;  // leave results[job] null = known-zero marker
            }
            auto plain = cc->MakeCKKSPackedPlaintext(mask);
            plain->SetLength(static_cast<size_t>(batch_size));
            results[job] = plain;
          }
        } catch (...) {
          worker_errors[worker] = std::current_exception();
        }
      });
    }
    for (auto& worker : workers) {
      worker.join();
    }
    for (const auto& error : worker_errors) {
      if (error) {
        std::rethrow_exception(error);
      }
    }
    for (std::size_t job = 0; job < pending.size(); ++job) {
      prefetched[pending[job].position] = results[job];
    }
  };
  for (std::size_t giant_index = 0; giant_index < giants.size(); ++giant_index) {
    const int giant = giants[giant_index];
    Ciphertext<DCRTPoly> inner;
    bool has_inner = false;
    for (int baby = 0; baby < baby_step; ++baby) {
      const int offset = giant + baby;
      const auto table_index = giant_index * static_cast<std::size_t>(baby_step) +
                               static_cast<std::size_t>(baby);
      Plaintext plain;
      if (in_persistent_table(table_index)) {
        plain = (*plain_table)[table_index];
        if (cache_hits != nullptr) {
          ++*cache_hits;
        }
      } else if (encode_threads > 1) {
        auto ready = prefetched.find(table_index);
        if (ready == prefetched.end()) {
          prefetch_from(table_index);
          ready = prefetched.find(table_index);
        }
        if (ready == prefetched.end()) {
          continue;  // defensive: nothing pending at or past this position
        }
        plain = ready->second;
        prefetched.erase(ready);
        if (!plain) {
          continue;  // all-zero mask
        }
        if (cache_misses != nullptr) {
          ++*cache_misses;
        }
      } else {
        auto mask = slot_bsgs_pre_mask(weights, input_dim, output_dim, batch_size, stride,
                                       giant, offset);
        if (std::all_of(mask.begin(), mask.end(), [](double value) { return value == 0.0; })) {
          continue;
        }
        plain = cc->MakeCKKSPackedPlaintext(mask);
        plain->SetLength(static_cast<size_t>(batch_size));
        if (cache_misses != nullptr) {
          ++*cache_misses;
        }
      }
      auto term = cc->EvalMult(baby_ct.at(baby), plain);
      ++ct_pt_muls;
      if (!has_inner) {
        inner = term;
        has_inner = true;
      } else {
        inner = cc->EvalAdd(inner, term);
        ++adds;
      }
    }
    if (!has_inner) {
      continue;
    }
    if (giant != 0) {
      inner = rotate_fn(inner, giant);
    }
    if (!has_accumulator) {
      accumulator = inner;
      has_accumulator = true;
    } else {
      accumulator = cc->EvalAdd(accumulator, inner);
      ++adds;
    }
  }
  if (!has_accumulator) {
    throw std::runtime_error("slot BSGS produced no terms");
  }
  return accumulator;
}

void align_levels(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& lhs,
    Ciphertext<DCRTPoly>& rhs,
    const std::string& mode,
    int& unity_multiplies,
    int& direct_drops) {
  if (mode == "native") {
    return;
  }
  if (mode == "drop") {
    const auto target = std::max(lhs->GetLevel(), rhs->GetLevel());
    if (lhs->GetLevel() < target) {
      lhs->SetLevel(target);
      ++direct_drops;
    }
    if (rhs->GetLevel() < target) {
      rhs->SetLevel(target);
      ++direct_drops;
    }
    return;
  }
  for (int guard = 0; guard < 128 && lhs->GetLevel() < rhs->GetLevel(); ++guard) {
    const auto before = lhs->GetLevel();
    cc->EvalMultInPlace(lhs, 1.0);
    ++unity_multiplies;
    if (lhs->GetLevel() == before) {
      break;
    }
  }
  for (int guard = 0; guard < 128 && rhs->GetLevel() < lhs->GetLevel(); ++guard) {
    const auto before = rhs->GetLevel();
    cc->EvalMultInPlace(rhs, 1.0);
    ++unity_multiplies;
    if (rhs->GetLevel() == before) {
      break;
    }
  }
}

auto decrypt_slots(
    const CryptoContext<DCRTPoly>& cc,
    const PrivateKey<DCRTPoly>& secret_key,
    Ciphertext<DCRTPoly> ciphertext,
    size_t length) -> std::vector<double> {
  Plaintext plaintext;
  auto ciphertext_handle = ciphertext->Clone();
  cc->Decrypt(secret_key, ciphertext_handle, &plaintext);
  plaintext->SetLength(length);
  auto values = plaintext->GetRealPackedValue();
  values.resize(length);
  return values;
}

// ---------------------------------------------------------------------------
// Chebyshev Paterson-Stockmeyer (log-depth) with a host-side self check.
//
// Split at giant k (power of two >= (n+1)/2, multiple of the baby size m):
//   p = A'(u) + T_k(u) * Btil(u)
//   Btil_j = 2*c_{k+j} (j>=1), Btil_0 = c_k
//   A'_i   = c_i (i<k), then A'_{2k-i} -= c_i for i in (k, n]
// Both children have degree <= k-1, so the recursion halves. Baby T_0..T_{m-1}
// and giant T_{m*2^j} ciphertexts are produced with the double-angle rules
// T_{2i} = 2*T_i^2 - 1 and T_{2i+1} = 2*T_{i+1}*T_i - T_1.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Process telemetry + JSON writers (conventions of the rank/gate kernel).
// ---------------------------------------------------------------------------

auto read_status_kib(const std::string& key) -> long long {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind(key + ":", 0) != 0) {
      continue;
    }
    std::stringstream stream(line.substr(key.size() + 1));
    long long value = 0;
    std::string unit;
    stream >> value >> unit;
    return value;
  }
  return 0;
}

auto rss_gib() -> double { return static_cast<double>(read_status_kib("VmRSS")) / (1024.0 * 1024.0); }

auto peak_rss_gib() -> double {
  return static_cast<double>(read_status_kib("VmHWM")) / (1024.0 * 1024.0);
}

void log_phase(const std::string& message) {
  std::cerr << "[stage1_mamba2_decode_fideslib] " << message << " rss_gib=" << rss_gib()
            << " peak_rss_gib=" << peak_rss_gib() << std::endl;
}


void write_operation_counts_json(std::ostringstream& out, const OperationCounts& counts) {
  out << "{";
  out << "\"rotations\":" << counts.rotations << ",";
  out << "\"rotations_direct\":" << counts.rotations_direct << ",";
  out << "\"rotations_composite_steps\":" << counts.rotations_composite_steps << ",";
  out << "\"ct_pt_mul\":" << counts.ct_pt_mul << ",";
  out << "\"ct_ct_mul\":" << counts.ct_ct_mul << ",";
  out << "\"adds\":" << counts.adds << ",";
  out << "\"unity_level_align_muls\":" << counts.unity_level_align_muls << ",";
  out << "\"direct_level_align_drops\":" << counts.direct_level_align_drops << ",";
  out << "\"bootstraps\":" << counts.bootstraps;
  out << "}";
}

void write_phase_operation_counts_json(
    std::ostringstream& out,
    const std::map<std::string, OperationCounts>& values) {
  out << "{";
  bool first = true;
  for (const auto& [name, counts] : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << "\"" << json_escape(name) << "\":";
    write_operation_counts_json(out, counts);
  }
  out << "}";
}


// Per-token depth estimate from the poly degrees (mirrors the ciphertext
// program's level arithmetic; validated against a slot-level simulation).

}  // namespace

auto main(int argc, char* argv[]) -> int {
  Config args;
  bool args_available = false;
  try {
    args = parse_args(argc, argv);
    args_available = true;

    const bool chain_mode = !args.input_chain.empty();
    ChainPayload chain;
    std::vector<M1Payload> layer_payloads;
    if (chain_mode) {
      chain = read_chain_payload(args.input_chain,
                                 args.autoregressive_client_loop);
      const int layers_to_load =
          args.max_layers > 0 ? std::min(args.max_layers, chain.n_layers) : chain.n_layers;
      layer_payloads.reserve(static_cast<std::size_t>(layers_to_load));
      for (int layer = 0; layer < layers_to_load; ++layer) {
        layer_payloads.push_back(read_m1_payload(
            args.input_chain + "/" + chain.layer_dirs[static_cast<std::size_t>(layer)]));
      }
      if (!args.autoregressive_client_loop && args.tokens > chain.n_test_tokens) {
        throw std::runtime_error("--tokens exceeds n_test_tokens in the chain payload");
      }
    } else {
      layer_payloads.push_back(read_m1_payload(args.input));
    }
    const auto& dims_payload = layer_payloads.front();
    const int layers_loaded = static_cast<int>(layer_payloads.size());
    const bool full_chain = chain_mode && layers_loaded == chain.n_layers;
    if (args.autoregressive_client_loop) {
      if (!chain.has_autoregressive) {
        throw std::runtime_error(
            "autoregressive-client-loop requested but payload has no assets");
      }
      if (!full_chain) {
        throw std::runtime_error("autoregressive-client-loop requires the full layer chain");
      }
      if (args.tokens != chain.autoregressive_server_evaluations) {
        throw std::runtime_error(
            "--tokens must equal autoregressive server_evaluations (" +
            std::to_string(chain.autoregressive_server_evaluations) + ")");
      }
      if (chain.autoregressive_server_evaluations !=
          chain.autoregressive_prompt_tokens +
              chain.autoregressive_generate_tokens - 1) {
        throw std::runtime_error("invalid autoregressive evaluation count");
      }
      if (args.debug_layer_errors) {
        throw std::runtime_error(
            "debug-layer-errors has no per-layer autoregressive references");
      }
    }
    for (std::size_t index = 0; index < layer_payloads.size(); ++index) {
      require_same_layer_dims(dims_payload, layer_payloads[index], index);
      if (!args.autoregressive_client_loop &&
          args.tokens > layer_payloads[index].n_test_tokens) {
        throw std::runtime_error("--tokens exceeds n_test_tokens in layer payload " +
                                 std::to_string(index));
      }
      for (const auto& [name, spec] : layer_payloads[index].polys) {
        if (!spec.coeffs.empty()) {
          verify_cheb_ps_host(name, spec.coeffs);
        }
      }
    }
    if (args.autoregressive_client_loop) {
      const auto evaluations =
          static_cast<std::size_t>(chain.autoregressive_server_evaluations);
      const auto width = static_cast<std::size_t>(dims_payload.d_model);
      const auto vocab = static_cast<std::size_t>(chain.autoregressive_vocab_size);
      if (chain.autoregressive_expected_generated_ids.size() !=
              static_cast<std::size_t>(chain.autoregressive_generate_tokens) ||
          chain.autoregressive_embeddings.size() != evaluations * width ||
          chain.autoregressive_expected_poly_final.size() != evaluations * width ||
          chain.autoregressive_expected_exact_final.size() != evaluations * width ||
          chain.client_embedding_w.size() != vocab * width ||
          (!chain.client_lm_head_w.empty() &&
           chain.client_lm_head_w.size() != vocab * width) ||
          (!chain.client_lm_head_b.empty() &&
           chain.client_lm_head_b.size() != vocab)) {
        throw std::runtime_error("autoregressive tensor shape mismatch");
      }
    }
    if (args.process_role == "client-decrypt") {
      const auto paths = handoff_paths(args.handoff_dir);
      CryptoContext<DCRTPoly> client_context;
      PrivateKey<DCRTPoly> client_secret_key;
      require_serialized(
          fideslib::Serial::DeserializeFromFile(
              (paths.client / "context.bin").string(), client_context,
              SerType::BINARY),
          "failed to deserialize client context");
      require_serialized(
          fideslib::Serial::DeserializeFromFile(
              (paths.client / "secret-key.bin").string(), client_secret_key,
              SerType::BINARY),
          "failed to deserialize client secret key");
      const std::vector<double>& exact_reference =
          chain_mode
              ? (full_chain
                     ? chain.expected_final
                     : layer_payloads.back().tensors.at("test_layer_output"))
              : layer_payloads.front().tensors.at("test_layer_output");
      const bool has_poly_reference =
          chain_mode &&
          (full_chain
               ? !chain.expected_poly_final.empty()
               : layer_payloads.back().tensors.count(
                     "test_layer_output_poly") > 0);
      const std::vector<double>& reference =
          has_poly_reference
              ? (full_chain
                     ? chain.expected_poly_final
                     : layer_payloads.back().tensors.at(
                           "test_layer_output_poly"))
              : exact_reference;
      std::vector<double> per_token_errors;
      std::vector<int> per_token_decrypt_ok;
      double max_error = 0.0;
      for (int token = 0; token < args.tokens; ++token) {
        try {
          auto output = deserialize_ciphertext(
              paths.exchange /
                  ("output_t" + std::to_string(token) + ".ct"),
              client_context);
          const auto slots = decrypt_slots(
              client_context, client_secret_key, output,
              static_cast<std::size_t>(dims_payload.d_model));
          double token_error = 0.0;
          for (int slot = 0; slot < dims_payload.d_model; ++slot) {
            const double difference = std::abs(
                slots[static_cast<std::size_t>(slot)] -
                reference[static_cast<std::size_t>(token) *
                              dims_payload.d_model +
                          slot]);
            if (!std::isfinite(difference)) {
              token_error = 1.0e308;
              break;
            }
            token_error = std::max(token_error, difference);
          }
          per_token_decrypt_ok.push_back(1);
          per_token_errors.push_back(token_error);
          max_error = std::max(max_error, token_error);
        } catch (const std::exception& exc) {
          log_phase("client decrypt token " + std::to_string(token) +
                    " failed: " + exc.what());
          per_token_decrypt_ok.push_back(0);
          per_token_errors.push_back(1.0e308);
          max_error = 1.0e308;
        }
      }
      const int server_secret_key_files = count_server_secret_files(paths);
      const bool passed =
          max_error <= args.tolerance && server_secret_key_files == 0;
      std::ostringstream result;
      result << "{";
      write_artifact_prefix(result, args);
      result << "\"status\":\"" << (passed ? "passed" : "failed")
             << "\",";
      result << "\"passed\":" << (passed ? "true" : "false") << ",";
      result << "\"parameters\":{\"process_role\":\"client-decrypt\",";
      result << "\"tokens\":" << args.tokens << ",";
      result << "\"layers\":" << layers_loaded << ",";
      result << "\"tolerance\":" << args.tolerance << "},";
      result << "\"measurements\":{\"max_abs_error\":" << max_error
             << ",\"per_token_max_abs_error\":";
      write_double_vector_json(result, per_token_errors);
      result << ",\"per_token_decrypt_ok\":";
      write_int_vector_json(result, per_token_decrypt_ok);
      result << ",\"server_secret_key_files\":"
             << server_secret_key_files << "},";
      result << "\"measurement_scope\":{";
      result << "\"client_server_process_separation\":true,";
      result << "\"server_secret_key_loaded\":false,";
      result << "\"full_model_correctness_claimed\":"
             << (full_chain ? "true" : "false") << ",";
      result << "\"claim\":\"Client-only decryption and correctness "
                "verification of ciphertext outputs produced by the "
                "process-separated Mamba server.\"}";
      result << "}";
      write_payload(args.output_json, result.str());
      return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const bool all_carried_bounds_calibrated =
        std::all_of(layer_payloads.begin(), layer_payloads.end(), [](const auto& payload) {
          return payload.state_abs_max >= 0.0 && payload.fifo_abs_max >= 0.0;
        });
    const bool all_state_head_bounds_calibrated =
        std::all_of(layer_payloads.begin(), layer_payloads.end(), [](const auto& payload) {
          return payload.state_head_abs_max.size() ==
                 static_cast<std::size_t>(payload.num_heads);
        });
    const bool all_checkpoint_bounds_calibrated =
        std::all_of(layer_payloads.begin(), layer_payloads.end(), [](const auto& payload) {
          return payload.checkpoint_abs_max.size() == 8;
        });
    if (args.normalized_recurrent_state && !all_state_head_bounds_calibrated) {
      throw std::runtime_error(
          "normalized recurrent state requires calibration-text head-wise state bounds");
    }
    const int batch_size = args.ring_dim / 2;
    const auto packing = derive_packing(dims_payload, batch_size);
    // Multi-stream packing geometry: S streams at stride batch/S. Every
    // packed-layout vector (hidden, proj, y) must fit inside one stride so
    // stride-periodic plaintexts and shared rotations serve all streams.
    const int stream_stride = batch_size / args.streams;
    if (args.streams > 1 && stream_stride < dims_payload.proj_dim) {
      throw std::runtime_error(
          "streams do not fit: batch/streams must be >= proj_dim (" +
          std::to_string(stream_stride) + " < " + std::to_string(dims_payload.proj_dim) + ")");
    }
    // Input-replicated BSGS shapes (replicas == 1 -> legacy path).
    const int forced_replicas =
        args.bsgs_replicas == "auto" ? 0
        : args.bsgs_replicas == "1"  ? -1
                                     : std::stoi(args.bsgs_replicas);
    const auto resolve_projection_shape = [&](int output_dim, int input_dim) {
      if (forced_replicas < 0) {
        return ReplicatedShape{};
      }
      return args.interleaved_replicated_projection
                 ? resolve_interleaved_replicated_shape(
                       output_dim, input_dim, batch_size, forced_replicas)
                 : resolve_replicated_shape(output_dim, input_dim, batch_size,
                                            forced_replicas);
    };
    auto rep_in =
        resolve_projection_shape(dims_payload.proj_dim, dims_payload.d_model);
    auto rep_out =
        resolve_projection_shape(dims_payload.d_model, dims_payload.d_inner);
    if (args.replicated_true_bsgs) {
      if (rep_in.replicas > 1) {
        rep_in.baby_step = std::max(
            1, static_cast<int>(std::sqrt(static_cast<double>(rep_in.per_replica))));
      }
      if (rep_out.replicas > 1) {
        rep_out.baby_step = std::max(
            1, static_cast<int>(std::sqrt(static_cast<double>(rep_out.per_replica))));
      }
    }
    log_phase("bsgs layout mode=" + args.bsgs_replicas +
              " in_proj r=" + std::to_string(rep_in.replicas) +
              " window=" + std::to_string(rep_in.window) +
              " diagonals=" + std::to_string(rep_in.replicas > 1 ? rep_in.per_replica : 0) +
              " baby=" + std::to_string(rep_in.baby_step) +
              " out_proj r=" + std::to_string(rep_out.replicas) +
              " window=" + std::to_string(rep_out.window) +
              " diagonals=" +
              std::to_string(rep_out.replicas > 1 ? rep_out.per_replica : 0) +
              " baby=" + std::to_string(rep_out.baby_step));
    // Rotation indices depend only on dims/packing (asserted equal across
    // layers) plus the BSGS layout choice; the required set is the source of
    // truth (logged to JSON for reconciliation with the Python planner).
    const auto rotation_indices = required_rotations(
        dims_payload, packing, rep_in, rep_out, args.replicated_state_blocks);
    verify_naf(rotation_indices);
    auto rotation_freqs = rotation_frequencies(
        dims_payload, packing, layers_loaded, args.streams, stream_stride,
        rep_in, rep_out, args.replicated_state_blocks);
    {
      // Guard against generator/frequency drift: every counted index must be
      // required, and every required index gets an entry (frequency 0 if a
      // family only exists for another geometry, e.g. full-batch sums when
      // streams > 1).
      for (const auto& [index, frequency] : rotation_freqs) {
        if (std::find(rotation_indices.begin(), rotation_indices.end(), index) ==
            rotation_indices.end()) {
          throw std::runtime_error("rotation frequency index " + std::to_string(index) +
                                   " is not in the required set");
        }
        (void)frequency;
      }
      for (const int32_t index : rotation_indices) {
        rotation_freqs.try_emplace(index, 0.0);
      }
    }
    // Two-tier key plan: base = the signed powers of two covering every NAF
    // decomposition actually used; balanced adds the hottest non-base keys by
    // frequency x (NAF weight - 1) under the byte budget.
    const double rotation_per_key_gib =
        rotation_key_gib_estimate(args.ring_dim, args.multiplicative_depth);
    std::set<int32_t> rotation_base_keys;
    for (const int32_t index : rotation_indices) {
      for (const int step : naf_steps(index)) {
        rotation_base_keys.insert(static_cast<int32_t>(step));
      }
    }
    std::set<int32_t> rotation_key_set;
    if (args.rotation_keys == "full") {
      rotation_key_set.insert(rotation_indices.begin(), rotation_indices.end());
    } else {
      rotation_key_set = rotation_base_keys;
      if (args.rotation_keys == "balanced") {
        std::vector<std::pair<double, int32_t>> candidates;
        for (const auto& [index, frequency] : rotation_freqs) {
          if (rotation_base_keys.count(index) > 0) {
            continue;
          }
          const auto weight = static_cast<double>(naf_steps(index).size());
          const double savings = frequency * (weight - 1.0);
          if (savings > 0.0) {
            candidates.emplace_back(savings, index);
          }
        }
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& lhs, const auto& rhs) {
                    if (lhs.first != rhs.first) {
                      return lhs.first > rhs.first;
                    }
                    return std::abs(lhs.second) < std::abs(rhs.second);
                  });
        const auto affordable = static_cast<long long>(
            args.rotation_key_gib / rotation_per_key_gib) -
            static_cast<long long>(rotation_key_set.size());
        long long taken = 0;
        for (const auto& [savings, index] : candidates) {
          if (taken >= affordable) {
            break;
          }
          rotation_key_set.insert(index);
          ++taken;
          (void)savings;
        }
      }
    }
    // Planned composite applications per token (estimate for JSON).
    double planned_composite_apps = 0.0;
    for (const auto& [index, frequency] : rotation_freqs) {
      if (rotation_key_set.count(index) == 0) {
        planned_composite_apps +=
            frequency * static_cast<double>(naf_steps(index).size());
      }
    }
    const std::vector<int32_t> rotation_keygen_indices(rotation_key_set.begin(),
                                                       rotation_key_set.end());
    log_phase("rotation key plan mode=" + args.rotation_keys +
              " required=" + std::to_string(rotation_indices.size()) +
              " keys=" + std::to_string(rotation_keygen_indices.size()) +
              " base=" + std::to_string(rotation_base_keys.size()) +
              " est_gib=" +
              std::to_string(rotation_keygen_indices.size() * rotation_per_key_gib) +
              " planned_composite_apps_per_token=" +
              std::to_string(planned_composite_apps));
    // Depth estimate per layer (degrees are frozen across layers, but the
    // decay range-reduction squarings differ); merge to the worst layer for
    // the segment map and geometry warning.
    auto depth_estimate = estimate_levels(
        dims_payload, args.tokens, args.bootstrap_before_token,
        args.debug_client_reencrypt_before_token,
        args.refresh_recurrent_state_post ||
            args.refresh_recurrent_state_post_layers.count(0) > 0,
        args.state_refresh_interval,
        args.replicated_state_blocks,
        args.streams);
    for (std::size_t index = 1; index < layer_payloads.size(); ++index) {
      const auto candidate = estimate_levels(
          layer_payloads[index], args.tokens, args.bootstrap_before_token,
          args.debug_client_reencrypt_before_token,
          args.refresh_recurrent_state_post ||
              args.refresh_recurrent_state_post_layers.count(static_cast<int>(index)) > 0,
          args.state_refresh_interval,
          args.replicated_state_blocks,
          args.streams);
      depth_estimate.required_depth =
          std::max(depth_estimate.required_depth, candidate.required_depth);
      depth_estimate.req_dt = std::max(depth_estimate.req_dt, candidate.req_dt);
      depth_estimate.max_segment =
          std::max(depth_estimate.max_segment, candidate.max_segment);
    }
    const int final_norm_requirement = 12;
    {
      std::ostringstream estimate_text;
      estimate_text << "no-bootstrap ledger per token output:";
      for (const int level : depth_estimate.token_output_levels) {
        estimate_text << " " << level;
      }
      estimate_text << "; segment map: residual->" << depth_estimate.req_residual
                    << " proj->" << depth_estimate.req_proj
                    << " fifo->" << depth_estimate.req_fifo
                    << " conv->" << depth_estimate.req_conv
                    << " dt->" << depth_estimate.req_dt
                    << " decay->" << depth_estimate.req_decay
                    << " state_pre->" << depth_estimate.req_state_pre
                    << " state_tail->" << depth_estimate.req_state_tail
                    << " y->" << depth_estimate.req_y
                    << " newton->" << kNewtonSegmentEstimate
                    << " (max segment " << depth_estimate.max_segment
                    << " + bootstrap output " << kAssumedBootstrapOutputLevel
                    << " vs depth " << args.multiplicative_depth << ")";
      log_phase(estimate_text.str());
      if (kAssumedBootstrapOutputLevel + depth_estimate.max_segment >
          args.multiplicative_depth) {
        log_phase("WARNING: max circuit segment does not fit between bootstrap refreshes "
                  "at the configured depth; expect level exhaustion");
      }
    }
    log_phase(
        "payload loaded d_model=" + std::to_string(dims_payload.d_model) +
        " proj_dim=" + std::to_string(dims_payload.proj_dim) +
        " groups=" + std::to_string(packing.group_count) +
        " layers=" + std::to_string(layers_loaded) +
        (chain_mode ? "/" + std::to_string(chain.n_layers) : std::string()) +
        " rotation_keys=" + std::to_string(rotation_indices.size()));

    // -----------------------------------------------------------------------
    // Context setup (ordering per stage1_rank_gate_fideslib.cpp).
    // -----------------------------------------------------------------------
    // Mid-circuit bootstrap checkpoints are mandatory under the FIDESlib
    // MAXP=64 depth ceiling, so bootstrapping is always provisioned.
    const bool bootstrap_available = true;
    const auto setup_start = now();
    log_phase("context setup begin");
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(resolve_secret_key_dist(args.secret_key_dist));
    parameters.SetSecurityLevel(resolve_security(args.security));
    parameters.SetRingDim(static_cast<uint32_t>(args.ring_dim));
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetFirstModSize(static_cast<uint32_t>(args.first_mod_size));
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetMultiplicativeDepth(static_cast<uint32_t>(args.multiplicative_depth));
    parameters.SetScalingModSize(static_cast<uint32_t>(args.scaling_mod_size));
    parameters.SetBatchSize(static_cast<uint32_t>(batch_size));
    parameters.SetDevices({0});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    if (args.secret_key_dist == "sparse-ternary" ||
        args.secret_key_dist == "sparse-encapsulated") {
      parameters.SetNumLargeDigits(3);
    }

    CryptoContext<DCRTPoly> cc;
    PublicKey<DCRTPoly> public_key;
    PrivateKey<DCRTPoly> secret_key;
    double rotate_keygen_seconds = 0.0;
    double bootstrap_precompute_seconds = 0.0;
    if (args.process_role == "server-eval") {
      const auto paths = handoff_paths(args.handoff_dir);
      const auto& server_dir = paths.server;
      require_server_has_no_secret_files(paths);
      require_serialized(
          fideslib::Serial::DeserializeFromFile(
              (server_dir / "context.bin").string(), cc, SerType::BINARY),
          "failed to deserialize server context");
      require_serialized(
          fideslib::Serial::DeserializeFromFile(
              (server_dir / "public-key.bin").string(), public_key,
              SerType::BINARY),
          "failed to deserialize server public key");
      {
        std::ifstream stream(server_dir / "eval-mult.bin",
                             std::ios::binary);
        require_serialized(
            cc->DeserializeEvalMultKey(stream, SerType::BINARY),
            "failed to deserialize multiplication keys");
      }
      {
        std::ifstream stream(server_dir / "eval-rotation.bin",
                             std::ios::binary);
        require_serialized(
            cc->DeserializeEvalAutomorphismKey(stream, SerType::BINARY),
            "failed to deserialize rotation/bootstrap keys");
      }
      log_phase("server public/evaluation key load done secret_key_loaded=false");
    } else {
      cc = GenCryptoContext(parameters);
      cc->Enable(PKE);
      cc->Enable(KEYSWITCH);
      cc->Enable(LEVELEDSHE);
      if (bootstrap_available) {
        cc->Enable(ADVANCEDSHE);
        cc->Enable(FHE);
      }
      auto generated_keys = cc->KeyGen();
      public_key = generated_keys.publicKey;
      secret_key = generated_keys.secretKey;
      cc->EvalMultKeyGen(secret_key);
      log_phase("context setup done");
      const auto keygen_start = now();
      log_phase("rotation keygen begin mode=" + args.rotation_keys +
                " count=" + std::to_string(rotation_keygen_indices.size()));
      cc->EvalRotateKeyGen(secret_key, rotation_keygen_indices);
      rotate_keygen_seconds = seconds_since(keygen_start);
      log_phase("rotation keygen done");
      if (bootstrap_available) {
        const auto bootstrap_precompute_start = now();
        log_phase("bootstrap setup/keygen begin");
        const std::vector<uint32_t> level_budget = {
            static_cast<uint32_t>(args.bootstrap_level_budget_cts),
            static_cast<uint32_t>(args.bootstrap_level_budget_stc),
        };
        const std::vector<uint32_t> bsgs_dim = {
            static_cast<uint32_t>(args.bootstrap_bsgs_dim_cts),
            static_cast<uint32_t>(args.bootstrap_bsgs_dim_stc),
        };
        cc->EvalBootstrapSetup(level_budget, bsgs_dim,
                               static_cast<uint32_t>(batch_size), 0);
        cc->EvalBootstrapKeyGen(secret_key,
                                static_cast<uint32_t>(batch_size));
        bootstrap_precompute_seconds =
            seconds_since(bootstrap_precompute_start);
        log_phase("bootstrap setup/keygen done");
      }
    }
    const auto load_start = now();
    log_phase("load context begin");
    cc->LoadContext(public_key);
    const double load_context_seconds = seconds_since(load_start);
    log_phase("load context done");
    const double setup_seconds = seconds_since(setup_start);

    // -----------------------------------------------------------------------
    // Counters, phase timing, elementary helpers.
    // -----------------------------------------------------------------------
    int rotations = 0;
    int rotations_direct = 0;
    int rotations_composite_steps = 0;
    int ct_pt_muls = 0;
    int ct_ct_muls = 0;
    int adds = 0;
    int unity_multiplies = 0;
    int direct_level_drops = 0;
    int projection_late_level_drops = 0;
    int bootstraps = 0;
    int state_bootstraps = 0;  // refreshes of carried-forward (state/FIFO) cts
    int meta_bts_applied = 0;  // refreshes that ran the double-BTS path
    int debug_client_reencrypt_ciphertexts = 0;
    double bootstrap_eval_seconds = 0.0;
    double debug_client_reencrypt_seconds = 0.0;
    struct BootstrapEvent {
      std::string checkpoint;
      int level_before = 0;
      int level_after = 0;
      int requirement = 0;
      int policy_headroom = 0;
      int physical_bootstraps = 0;
      bool carried = false;
      bool meta_bts = false;
      double bound = 0.0;
      double seconds = 0.0;
    };
    std::vector<BootstrapEvent> bootstrap_events;
    std::map<std::string, double> phase_timings;
    std::map<std::string, OperationCounts> phase_operation_counts;
    std::map<std::string, int> ckks_levels;

    auto current_operation_counts = [&]() {
      return OperationCounts{
          .rotations = rotations,
          .rotations_direct = rotations_direct,
          .rotations_composite_steps = rotations_composite_steps,
          .ct_pt_mul = ct_pt_muls,
          .ct_ct_mul = ct_ct_muls,
          .adds = adds,
          .unity_level_align_muls = unity_multiplies,
          .direct_level_align_drops = direct_level_drops,
          .bootstraps = bootstraps,
      };
    };
    auto record_phase = [&](
                            const std::string& name,
                            std::chrono::steady_clock::time_point phase_start,
                            const OperationCounts& before) {
      phase_timings[name] += seconds_since(phase_start);
      phase_operation_counts[name] = add_counts(
          phase_operation_counts[name],
          subtract_counts(current_operation_counts(), before));
    };
    auto time_phase = [&](const std::string& name, auto&& work)
        -> std::invoke_result_t<decltype(work)&> {
      using Result = std::invoke_result_t<decltype(work)&>;
      const auto before = current_operation_counts();
      const auto phase_start = now();
      if constexpr (std::is_void_v<Result>) {
        work();
        record_phase(name, phase_start, before);
      } else {
        Result result = work();
        record_phase(name, phase_start, before);
        return result;
      }
    };

    auto make_plain = [&](const std::vector<double>& values) {
      auto plain = cc->MakeCKKSPackedPlaintext(values);
      plain->SetLength(static_cast<size_t>(batch_size));
      return plain;
    };
    auto make_plain_at_level = [&](const std::vector<double>& values,
                                   uint32_t level) {
      if (level == 0) {
        return make_plain(values);
      }
      return cc->MakeCKKSPackedPlaintext(
          values, 1, level, nullptr, static_cast<uint32_t>(batch_size));
    };
    auto encrypt_values = [&](const std::vector<double>& values) {
      auto plain = make_plain(values);
      return cc->Encrypt(public_key, plain);
    };
    if (args.process_role == "client-init") {
      const auto paths = handoff_paths(args.handoff_dir);
      prepare_client_handoff(paths);
      serialize_context(paths.client / "context.bin", cc);
      fs::copy_file(paths.client / "context.bin", paths.server / "context.bin",
                    fs::copy_options::overwrite_existing);
      copy_context_device_metadata(paths.client / "context.bin",
                                   paths.server / "context.bin");
      require_serialized(
          fideslib::Serial::SerializeToFile(
              (paths.server / "public-key.bin").string(), public_key,
              SerType::BINARY),
          "failed to serialize public key");
      require_serialized(
          fideslib::Serial::SerializeToFile(
              (paths.client / "secret-key.bin").string(), secret_key,
              SerType::BINARY),
          "failed to serialize secret key");
      fs::permissions(paths.client / "secret-key.bin",
                      fs::perms::owner_read | fs::perms::owner_write,
                      fs::perm_options::replace);
      {
        std::ofstream stream(paths.server / "eval-mult.bin",
                             std::ios::binary);
        require_serialized(cc->SerializeEvalMultKey(stream, SerType::BINARY),
                           "failed to serialize multiplication keys");
      }
      {
        std::ofstream stream(paths.server / "eval-rotation.bin",
                             std::ios::binary);
        require_serialized(
            cc->SerializeEvalAutomorphismKey(stream, SerType::BINARY),
            "failed to serialize rotation/bootstrap keys");
      }
      const std::vector<double>& client_inputs =
          chain_mode ? chain.input_embeddings
                     : layer_payloads.front().tensors.at("test_layer_input");
      for (int token = 0; token < args.tokens; ++token) {
        std::vector<double> slots(static_cast<std::size_t>(batch_size), 0.0);
        for (int stream = 0; stream < args.streams; ++stream) {
          const auto base = static_cast<std::size_t>(stream * stream_stride);
          for (int slot = 0; slot < dims_payload.d_model; ++slot) {
            slots[base + static_cast<std::size_t>(slot)] =
                client_inputs[static_cast<std::size_t>(token) *
                                  dims_payload.d_model +
                              slot];
          }
        }
        auto input = encrypt_values(slots);
        serialize_ciphertext(
            paths.exchange /
                ("input_t" + std::to_string(token) + ".ct"),
            cc, public_key, input);
      }
      std::ostringstream result;
      result << "{";
      write_artifact_prefix(result, args);
      result << "\"status\":\"passed\",\"passed\":true,";
      result << "\"parameters\":{\"process_role\":\"client-init\",";
      result << "\"tokens\":" << args.tokens << ",\"layers\":"
             << layers_loaded << "},";
      result << "\"measurements\":{\"encrypted_inputs_written\":"
             << args.tokens << "},";
      result << "\"measurement_scope\":{";
      result << "\"client_server_process_separation\":true,";
      result << "\"server_secret_key_loaded\":false,";
      result << "\"full_model_correctness_claimed\":false,";
      result << "\"claim\":\"Client key generation and encrypted input "
                "serialization for the process-separated Mamba kernel; no "
                "server evaluation is claimed by this phase.\"}";
      result << "}";
      write_payload(args.output_json, result.str());
      return EXIT_SUCCESS;
    }
    auto ones_ct = encrypt_values(std::vector<double>(static_cast<size_t>(batch_size), 1.0));

    // -----------------------------------------------------------------------
    // Token-invariant plaintext cache. dgx profiling: the per-token host NTT
    // re-encode of token-invariant masks/diagonals (MakeCKKSPackedPlaintext,
    // ~100 ms each on GB10) is 89% of M1 wall time; the encode-once tables
    // remove it for cached entries. Cache stores mutable Plaintexts so the
    // existing non-const Plaintext& call pattern is unchanged. Entries are
    // registered with a per-token reuse count and a builder, then encoded
    // greedily (reuse desc, registration order asc) until the byte budget is
    // exhausted; the rest falls back to per-use encoding. Registration and
    // encoding happen after the layer plans are built (see below).
    // -----------------------------------------------------------------------
    // "auto" resolves to "full" in BOTH modes: dgx chain telemetry (masks
    // mode, hits=2588 misses=0) showed BSGS diagonals bypassing cache and
    // pool entirely (1203->1086 s instead of ~5x). Chain mode admits hot
    // diagonals frequency-greedy across ALL layers within the byte budget
    // (uses are uniform, so the budget fills layer 0 upward in registration
    // order) instead of a rolling current+next-layer window: a window only
    // helps tokens >= 2 and swapping a layer's table in/out per token costs
    // exactly the per-diagonal encodes the cache exists to avoid; the
    // parallel-encode pool covers the uncached remainder either way.
    const std::string pt_cache_mode =
        args.pt_cache == "auto" ? std::string("full") : args.pt_cache;
    // 1-arg MakeCKKSPackedPlaintext encodes at full level: (depth+1) towers of
    // ring_dim 8-byte words per entry. --pt-cache-level shrinks entries via
    // the 5-arg overload when the caller guarantees the consumption level.
    const auto plain_bytes_at_level = [&](int level) {
      return static_cast<double>(args.ring_dim) *
             static_cast<double>(args.multiplicative_depth + 1 - level) * 8.0;
    };
    const double pt_plain_bytes = plain_bytes_at_level(args.pt_cache_level);
    const double pt_weight_plain_bytes =
        plain_bytes_at_level(args.pt_cache_weight_level);
    const double pt_cache_budget_bytes =
        pt_cache_mode == "off"
            ? 0.0
            : args.pt_cache_gib * 1024.0 * 1024.0 * 1024.0;
    struct PlainCacheEntry {
      int uses_per_token = 0;
      int order = 0;
      int encode_level = 0;
      double estimated_bytes = 0.0;
      std::function<std::vector<double>()> build;
      Plaintext plain;  // non-null once selected and encoded
    };
    std::map<std::string, PlainCacheEntry> plain_cache;
    long long pt_cache_hits = 0;
    long long pt_cache_misses = 0;
    long long pt_consumption_count = 0;
    long long pt_consumption_level_sum = 0;
    int pt_consumption_level_min = args.multiplicative_depth;
    int pt_consumption_level_max = 0;
    int pt_cache_hit_consumption_level_min = args.multiplicative_depth;
    long long pt_cache_level_bypasses = 0;
    long long pt_miss_consumption_level_encodes = 0;
    long long pt_miss_consumption_level_sum = 0;
    int pt_miss_consumption_level_min = args.multiplicative_depth;
    int pt_miss_consumption_level_max = 0;
    int pt_cache_order = 0;
    auto register_plain = [&](const std::string& key, int uses_per_token,
                              std::function<std::vector<double>()> build,
                              int encode_level = -1) {
      if (pt_cache_mode == "off") {
        return;
      }
      if (encode_level < 0) {
        encode_level = args.pt_cache_level;
      }
      auto [entry, inserted] = plain_cache.try_emplace(key);
      if (inserted) {
        entry->second.order = pt_cache_order++;
        entry->second.encode_level = encode_level;
        entry->second.estimated_bytes = plain_bytes_at_level(encode_level);
        entry->second.build = std::move(build);
      } else if (encode_level < entry->second.encode_level) {
        // A shared key must be valid at its earliest consumption site.
        entry->second.encode_level = encode_level;
        entry->second.estimated_bytes = plain_bytes_at_level(encode_level);
      }
      entry->second.uses_per_token += uses_per_token;
    };
    auto encode_cache_plain = [&](const std::vector<double>& values, int level) {
      if (level == 0) {
        return make_plain(values);
      }
      // Proven overload (used by the forensic path): scaleDeg 1 at a target
      // level; slots passed explicitly instead of SetLength.
      return cc->MakeCKKSPackedPlaintext(values, 1,
                                         static_cast<uint32_t>(level), nullptr,
                                         static_cast<uint32_t>(batch_size));
    };
    // Lookup for the small keyed vectors; values are only built/encoded on a
    // miss, so caching cannot change the math, only where encoding happens.
    auto cached_plain = [&](const std::string& key,
                            const std::vector<double>& values,
                            uint32_t consumption_level) -> Plaintext {
      ++pt_consumption_count;
      pt_consumption_level_sum += consumption_level;
      pt_consumption_level_min =
          std::min(pt_consumption_level_min, static_cast<int>(consumption_level));
      pt_consumption_level_max =
          std::max(pt_consumption_level_max, static_cast<int>(consumption_level));
      auto entry = plain_cache.find(key);
      if (entry != plain_cache.end() && entry->second.plain &&
          entry->second.encode_level <= static_cast<int>(consumption_level)) {
        ++pt_cache_hits;
        pt_cache_hit_consumption_level_min =
            std::min(pt_cache_hit_consumption_level_min,
                     static_cast<int>(consumption_level));
        return entry->second.plain;
      }
      if (entry != plain_cache.end() && entry->second.plain) {
        ++pt_cache_level_bypasses;
      }
      ++pt_cache_misses;
      if (!args.pt_miss_consumption_level || consumption_level == 0) {
        return make_plain(values);
      }
      ++pt_miss_consumption_level_encodes;
      pt_miss_consumption_level_sum += consumption_level;
      pt_miss_consumption_level_min =
          std::min(pt_miss_consumption_level_min,
                   static_cast<int>(consumption_level));
      pt_miss_consumption_level_max =
          std::max(pt_miss_consumption_level_max,
                   static_cast<int>(consumption_level));
      return make_plain_at_level(values, consumption_level);
    };

    // ct * scalar via Clone + EvalMultInPlace (only API forms proven in the
    // rank/gate kernel are used). Consumes one level.
    auto scaled_clone = [&](const Ciphertext<DCRTPoly>& ciphertext, double scalar) {
      auto output = ciphertext->Clone();
      cc->EvalMultInPlace(output, scalar);
      ++ct_pt_muls;
      return output;
    };
    // Mutating aligned add: both handles may be level-boosted in place.
    auto add_aligned = [&](Ciphertext<DCRTPoly> lhs, Ciphertext<DCRTPoly> rhs) {
      align_levels(
          cc, lhs, rhs, args.level_align_mode, unity_multiplies, direct_level_drops);
      ++adds;
      return cc->EvalAdd(lhs, rhs);
    };
    auto mul_aligned = [&](Ciphertext<DCRTPoly> lhs, Ciphertext<DCRTPoly> rhs) {
      align_levels(
          cc, lhs, rhs, args.level_align_mode, unity_multiplies, direct_level_drops);
      ++ct_ct_muls;
      return cc->EvalMult(lhs, rhs);
    };
    auto sub_aligned = [&](Ciphertext<DCRTPoly> lhs, Ciphertext<DCRTPoly> rhs,
                           const std::string& mode) {
      align_levels(cc, lhs, rhs, mode, unity_multiplies, direct_level_drops);
      ++adds;
      return cc->EvalSub(lhs, rhs);
    };
    auto add_scalar = [&](const Ciphertext<DCRTPoly>& ciphertext, double scalar) {
      return add_aligned(ciphertext, scaled_clone(ones_ct, scalar));
    };
    auto add_const_vector = [&](const Ciphertext<DCRTPoly>& ciphertext,
                                const std::string& key, const std::vector<double>& values) {
      auto plain = cached_plain(
          key, values, static_cast<uint32_t>(ciphertext->GetLevel()));
      return add_aligned(ciphertext, cc->Encrypt(public_key, plain));
    };
    auto mul_mask = [&](const Ciphertext<DCRTPoly>& ciphertext, const std::string& key,
                        const std::vector<double>& mask) {
      ++ct_pt_muls;
      // FIDESlib EvalMult takes Plaintext by non-const reference; a
      // MakeCKKSPackedPlaintext temporary cannot bind to it (the cache stores
      // mutable Plaintexts for the same reason).
      auto plain = cached_plain(
          key, mask, static_cast<uint32_t>(ciphertext->GetLevel()));
      return cc->EvalMult(ciphertext, plain);
    };
    // Rotation choke point: EVERY EvalRotate goes through here. "rotations"
    // counts logical rotations (mode-independent, so full mode stays
    // bit-identical); direct/composite applications are tracked separately.
    // EvalRotate is keyswitch-only (no rescale, never level-aligned in the
    // proven kernels), so NAF composition consumes zero levels.
    auto rotate = [&](const Ciphertext<DCRTPoly>& ciphertext, int amount)
        -> Ciphertext<DCRTPoly> {
      if (amount == 0) {
        return ciphertext;
      }
      ++rotations;
      if (rotation_key_set.count(static_cast<int32_t>(amount)) > 0) {
        ++rotations_direct;
        return cc->EvalRotate(ciphertext, amount);
      }
      auto output = ciphertext;
      for (const int step : naf_steps(amount)) {
        if (rotation_key_set.count(static_cast<int32_t>(step)) == 0) {
          throw std::runtime_error("missing rotation key for NAF step " +
                                   std::to_string(step) + " of index " +
                                   std::to_string(amount));
        }
        output = cc->EvalRotate(output, step);
        ++rotations_composite_steps;
      }
      return output;
    };
    // ct += rot(ct, -stride) doubling ladder.
    auto doubling_fill = [&](Ciphertext<DCRTPoly> ciphertext, int base_stride, int steps) {
      for (int k = 0; k < steps; ++k) {
        ciphertext = add_aligned(ciphertext, rotate(ciphertext, -(base_stride << k)));
      }
      return ciphertext;
    };
    auto full_batch_sum = [&](Ciphertext<DCRTPoly> ciphertext) {
      for (int k = 0; k < int_log2(batch_size); ++k) {
        ciphertext = add_aligned(ciphertext, rotate(ciphertext, 1 << k));
      }
      return ciphertext;
    };
    // Per-stream RMS variance reduction. streams == 1 keeps the exact
    // single-stream path (full cyclic sum, no mask, no extra level). For
    // streams > 1 the in-stride window sum is only correct at each stream
    // base slot (the squared vector is zero elsewhere in the stride), so the
    // bases are masked out (one tiled plaintext, +1 level) and re-broadcast
    // across the stride with existing power-of-two rotations.
    std::vector<double> stream_base_mask(static_cast<std::size_t>(batch_size), 0.0);
    for (int stream = 0; stream < args.streams; ++stream) {
      stream_base_mask[static_cast<std::size_t>(stream * stream_stride)] = 1.0;
    }
    auto norm_variance_sum = [&](Ciphertext<DCRTPoly> ciphertext) {
      if (args.streams == 1) {
        return full_batch_sum(std::move(ciphertext));
      }
      for (int k = 0; k < int_log2(stream_stride); ++k) {
        ciphertext = add_aligned(ciphertext, rotate(ciphertext, 1 << k));
      }
      ciphertext = mul_mask(ciphertext, "mask.streambase", stream_base_mask);
      return doubling_fill(ciphertext, 1, int_log2(stream_stride));
    };
    // Stream-base shifts (s * stride) decomposed into power-of-two rotations
    // so the rotation-key set is unchanged by multi-stream packing.
    auto rotate_composite = [&](Ciphertext<DCRTPoly> ciphertext, int amount) {
      const int sign = amount < 0 ? -1 : 1;
      const int magnitude = std::abs(amount);
      for (int bit = 0; (1 << bit) <= magnitude; ++bit) {
        if ((magnitude & (1 << bit)) != 0) {
          ciphertext = rotate(ciphertext, sign * (1 << bit));
        }
      }
      return ciphertext;
    };

    // -----------------------------------------------------------------------
    // Chebyshev PS ciphertext evaluator. Input u must already be normalized to
    // [-1, 1] (callers fold the affine domain map into the producing op).
    // -----------------------------------------------------------------------
    auto eval_chebyshev = [&](const Ciphertext<DCRTPoly>& u,
                              const std::vector<double>& coeffs) -> Ciphertext<DCRTPoly> {
      const int degree = static_cast<int>(coeffs.size()) - 1;
      if (degree < 1) {
        return scaled_clone(ones_ct, coeffs.empty() ? 0.0 : coeffs[0]);
      }
      const int m = cheb_baby_size(degree);
      std::map<int, Ciphertext<DCRTPoly>> t_cache;
      t_cache[1] = u;
      auto neg_u = scaled_clone(u, -1.0);
      std::function<Ciphertext<DCRTPoly>(int)> get_t = [&](int i) -> Ciphertext<DCRTPoly> {
        auto found = t_cache.find(i);
        if (found != t_cache.end()) {
          return found->second;
        }
        Ciphertext<DCRTPoly> value;
        if (i % 2 == 0) {
          auto half = get_t(i / 2);
          auto square = mul_aligned(half->Clone(), half->Clone());
          ++adds;
          auto doubled = cc->EvalAdd(square, square);
          value = add_aligned(doubled, scaled_clone(ones_ct, -1.0));
        } else {
          auto high = get_t((i + 1) / 2);
          auto low = get_t(i / 2);
          auto product = mul_aligned(high->Clone(), low->Clone());
          ++adds;
          auto doubled = cc->EvalAdd(product, product);
          value = add_aligned(doubled, neg_u->Clone());
        }
        t_cache[i] = value;
        return value;
      };
      std::function<Ciphertext<DCRTPoly>(std::vector<double>)> rec =
          [&](std::vector<double> c) -> Ciphertext<DCRTPoly> {
        const int n = static_cast<int>(c.size()) - 1;
        if (n < m) {
          Ciphertext<DCRTPoly> accumulator;
          bool has_accumulator = false;
          for (int i = 1; i <= n; ++i) {
            if (std::abs(c[static_cast<std::size_t>(i)]) < kChebCoefficientFloor) {
              continue;
            }
            auto term = scaled_clone(get_t(i), c[static_cast<std::size_t>(i)]);
            if (!has_accumulator) {
              accumulator = term;
              has_accumulator = true;
            } else {
              accumulator = add_aligned(accumulator, term);
            }
          }
          if (!has_accumulator) {
            return scaled_clone(ones_ct, c[0]);
          }
          if (std::abs(c[0]) >= kChebCoefficientFloor) {
            accumulator = add_scalar(accumulator, c[0]);
          }
          return accumulator;
        }
        int k = m;
        while (2 * k - 1 < n) {
          k *= 2;
        }
        std::vector<double> btil(static_cast<std::size_t>(n - k + 1), 0.0);
        for (int j = 0; j <= n - k; ++j) {
          btil[static_cast<std::size_t>(j)] = 2.0 * c[static_cast<std::size_t>(k + j)];
        }
        btil[0] = c[static_cast<std::size_t>(k)];
        std::vector<double> aprime(c.begin(), c.begin() + k);
        for (int i = k + 1; i <= n; ++i) {
          aprime[static_cast<std::size_t>(2 * k - i)] -= c[static_cast<std::size_t>(i)];
        }
        auto giant_term = mul_aligned(get_t(k)->Clone(), rec(btil));
        return add_aligned(rec(aprime), giant_term);
      };
      return rec(coeffs);
    };

    // -----------------------------------------------------------------------
    // Debug value telemetry (--debug-decrypt): decrypt and log max |value|
    // and non-finite slot counts at phase/bootstrap checkpoints to localize
    // numeric blowups. No-op unless the flag is set.
    // -----------------------------------------------------------------------
    auto debug_value_stats = [&](const Ciphertext<DCRTPoly>& ciphertext,
                                 const std::string& tag) {
      if (!args.debug_decrypt) {
        return;
      }
      Plaintext plaintext;
      auto ciphertext_handle = ciphertext->Clone();
      cc->Decrypt(secret_key, ciphertext_handle, &plaintext);
      plaintext->SetLength(static_cast<std::size_t>(batch_size));
      const auto complex_slots = plaintext->GetCKKSPackedValue();
      double max_real = 0.0;
      double max_imag = 0.0;  // CKKS bootstrap assumes real inputs; large
                              // imaginary mass is a blowup suspect on its own.
      int non_finite = 0;
      for (const auto& value : complex_slots) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
          ++non_finite;
          continue;
        }
        max_real = std::max(max_real, std::abs(value.real()));
        max_imag = std::max(max_imag, std::abs(value.imag()));
      }
      log_phase("DEBUG " + tag + " level=" + std::to_string(ciphertext->GetLevel()) +
                " deg=" + std::to_string(ciphertext->GetNoiseScaleDeg()) +
                " max_abs=" + std::to_string(max_real) +
                " max_imag=" + std::to_string(max_imag) +
                " non_finite=" + std::to_string(non_finite));
      // Lineage refresh probe: test-bootstrap a normalized clone to find the
      // earliest op whose output poisons EvalBootstrap (values are known
      // clean here; a NaN probe means the ciphertext state is the trigger).
      if (args.debug_refresh_probes && non_finite == 0 && std::isfinite(max_real)) {
        // Repeat the probe: identical inputs sometimes refresh clean and
        // sometimes NaN, so count outcomes to expose nondeterminism.
        int poisoned_runs = 0;
        constexpr int kProbeRepeats = 4;
        for (int repeat = 0; repeat < kProbeRepeats; ++repeat) {
          try {
            auto probe = ciphertext->Clone();
            cc->EvalMultInPlace(probe, 1.0 / std::max(1.0, 1.5 * max_real));
            while (probe->GetNoiseScaleDeg() > 1) {
              cc->RescaleInPlace(probe);
            }
            if (cudaDeviceSynchronize() != 0) {
              throw std::runtime_error("CUDA synchronization failed before debug bootstrap");
            }
            auto refreshed = cc->EvalBootstrap(probe);
            if (cudaDeviceSynchronize() != 0) {
              throw std::runtime_error("CUDA synchronization failed after debug bootstrap");
            }
            Plaintext probe_plain;
            cc->Decrypt(secret_key, refreshed, &probe_plain);
            probe_plain->SetLength(static_cast<std::size_t>(batch_size));
            auto probe_values = probe_plain->GetRealPackedValue();
            for (const double value : probe_values) {
              if (!std::isfinite(value)) {
                ++poisoned_runs;
                break;
              }
            }
          } catch (const std::exception& exc) {
            log_phase("DEBUG refresh_probe." + tag + " EXCEPTION " + exc.what());
          }
        }
        log_phase("DEBUG refresh_probe." + tag + " poisoned_runs=" +
                  std::to_string(poisoned_runs) + "/" + std::to_string(kProbeRepeats));
      }
    };

    // -----------------------------------------------------------------------
    // Auto-bootstrap policy: refresh a ciphertext when its remaining levels
    // cannot cover the estimated downstream requirement plus headroom. Uses
    // runtime GetLevel(); the assumed-bootstrap-output constant plus the
    // minimum-gain floor guard against refreshing near-fresh ciphertexts.
    // -----------------------------------------------------------------------
    // Per-layer measured carried-lineage bounds, set at run_layer entry from
    // the active layer's payload (< 0 -> no measurement -> generic fallback).
    double active_state_bound = -1.0;
    std::vector<double> active_state_group_bounds;
    double active_fifo_bound = -1.0;
    std::map<std::string, double> active_checkpoint_bounds;
    bool carried_bound_fallback_warned = false;
    auto eval_bootstrap_synced = [&](const Ciphertext<DCRTPoly>& input) {
      if (cudaDeviceSynchronize() != 0) {
        throw std::runtime_error("CUDA synchronization failed before EvalBootstrap");
      }
      auto output = cc->EvalBootstrap(input);
      if (cudaDeviceSynchronize() != 0) {
        throw std::runtime_error("CUDA synchronization failed after EvalBootstrap");
      }
      return output;
    };
    auto maybe_bootstrap = [&](Ciphertext<DCRTPoly>& ciphertext, int requirement,
                               const std::string& what, int& counter) {
      if (!bootstrap_available) {
        return;
      }
      const int level = static_cast<int>(ciphertext->GetLevel());
      const bool carried = is_carried_checkpoint(what);
      const bool residual_checkpoint = what.find("residual") != std::string::npos;
      // Persistent normalized state is already stored at calibration scale,
      // so undoing a standard bootstrap only amplifies its error by ~1.  The
      // second Meta-BTS pass was needed for large carried-state bounds; on
      // normalized state it doubles refresh cost without that justification.
      const bool normalized_state =
          carried && what.find("state") != std::string::npos &&
          args.normalized_recurrent_state;
      const bool use_meta_bts =
          args.meta_bts && !normalized_state &&
          (carried || what.find("gated_poly_input") != std::string::npos);
      // Meta-BTS residual amplification needs one live level after the
      // normalize/rescale, so eligible checkpoints trigger one level earlier.
      const int meta_headroom = use_meta_bts ? 1 : 0;
      int policy_headroom = args.auto_bootstrap_headroom;
      if (residual_checkpoint) {
        policy_headroom = args.residual_bootstrap_headroom;
      }
      if (carried) {
        policy_headroom = args.carried_bootstrap_headroom;
      }
      if (args.multiplicative_depth - level >=
          requirement + meta_headroom + policy_headroom) {
        return;
      }
      if (level < kAssumedBootstrapOutputLevel + kMinBootstrapGain) {
        return;  // bootstrapping cannot usefully improve a near-fresh ciphertext
      }
      debug_value_stats(ciphertext, "pre_bootstrap." + what);
      // FIDESlib EvalBootstrap needs noiseScaleDeg-1 inputs with |m| <~ 1:
      // deg-2 refresh error grows polynomially with magnitude (all-NaN at the
      // measured |m|~24 y checkpoint). Normalize by the per-checkpoint bound,
      // force rescale to deg 1, refresh, then undo the normalization. The
      // undo-multiply scales the refresh noise floor by the bound, so carried
      // lineages (state, FIFO) use the tighter state margin to inject less
      // noise per token; transient activations keep the looser margin.
      double base_bound = checkpoint_bound(what);
      if (!carried) {
        const bool scaled_y = what.find("y_scaled") != std::string::npos;
        for (const auto* name : {"gated_variance", "gated_newton", "residual", "output",
                                 "proj", "conv_silu", "dt"}) {
          const auto found = active_checkpoint_bounds.find(name);
          if (found != active_checkpoint_bounds.end() &&
              what.find(name) != std::string::npos) {
            base_bound = found->second;
            break;
          }
        }
        if (scaled_y) {
          base_bound = 1.0;
        } else {
          const auto y_bound = active_checkpoint_bounds.find("y");
          if (y_bound != active_checkpoint_bounds.end() &&
              what.find(".y") != std::string::npos) {
            base_bound = y_bound->second;
          }
        }
      }
      if (carried) {
        ++state_bootstraps;
        // Export-time carried bounds replace the generic 24/6 values. State
        // checkpoints use the maximum for the heads packed in that state
        // ciphertext when the payload supplies head-wise calibration; older
        // independent-calibration payloads retain their per-layer scalar.
        const bool is_fifo = what.find("fifo") != std::string::npos;
        double measured = is_fifo ? active_fifo_bound : active_state_bound;
        if (!is_fifo && !active_state_group_bounds.empty()) {
          std::size_t digit = what.size();
          while (digit > 0 && what[digit - 1] >= '0' && what[digit - 1] <= '9') {
            --digit;
          }
          if (digit < what.size()) {
            const auto state_slot = static_cast<std::size_t>(std::stoul(what.substr(digit)));
            measured = active_state_group_bounds[
                state_slot % active_state_group_bounds.size()];
          }
        }
        if (measured >= 0.0) {
          base_bound = measured;
        } else if (!carried_bound_fallback_warned) {
          carried_bound_fallback_warned = true;
          log_phase("WARNING: payload lacks independent calibration carried_bounds; "
                    "falling back to the generic 24/6 state/FIFO bounds (regenerate the "
                    "payload to tighten refresh noise)");
        }
      }
      const double bound = base_bound *
                           (carried ? args.state_bootstrap_margin : args.bootstrap_norm_margin);
      const auto bootstrap_start = now();
      time_phase("bootstrap", [&]() {
        cc->EvalMultInPlace(ciphertext, 1.0 / bound);
        ++ct_pt_muls;
        while (ciphertext->GetNoiseScaleDeg() > 1) {
          cc->RescaleInPlace(ciphertext);
        }
        debug_value_stats(ciphertext, "norm_bootstrap_in." + what);
        if (args.debug_decrypt) {
          // Clip guard: the normalized input must stay within [-1, 1] or the
          // bound clipped a real value (bound < true max |m|). Cheap decrypt,
          // debug-only; logs a warning so a too-tight margin is caught in sim
          // before it silently corrupts a run.
          Plaintext clip_plain;
          auto clip_handle = ciphertext->Clone();
          cc->Decrypt(secret_key, clip_handle, &clip_plain);
          clip_plain->SetLength(static_cast<std::size_t>(batch_size));
          double clip_max = 0.0;
          for (const double value : clip_plain->GetRealPackedValue()) {
            if (std::isfinite(value)) {
              clip_max = std::max(clip_max, std::abs(value));
            }
          }
          if (clip_max > 1.0 + 1e-6) {
            log_phase("WARNING: bootstrap norm clip at " + what +
                      " normalized_max=" + std::to_string(clip_max) +
                      " (bound too tight; raise the margin)");
          }
        }
        if (args.debug_refresh_probes) {
          // Forensic split (debug only): bootstrap a fresh re-encryption of
          // the exact same values at the same level. If the fresh copy
          // refreshes cleanly while the original NaNs, the failure lives in
          // the ciphertext state (noise / GPU-side metadata), not the values.
          Plaintext forensic_plain;
          auto forensic_handle = ciphertext->Clone();
          cc->Decrypt(secret_key, forensic_handle, &forensic_plain);
          forensic_plain->SetLength(static_cast<std::size_t>(batch_size));
          auto forensic_values = forensic_plain->GetRealPackedValue();
          forensic_values.resize(static_cast<std::size_t>(batch_size));
          Plaintext reencoded = cc->MakeCKKSPackedPlaintext(
              forensic_values, 1, ciphertext->GetLevel(), nullptr,
              static_cast<uint32_t>(batch_size));
          auto fresh = cc->Encrypt(public_key, reencoded);
          auto fresh_refreshed = eval_bootstrap_synced(fresh);
          debug_value_stats(fresh_refreshed, "forensic_fresh_bootstrap." + what);
        }
        if (use_meta_bts) {
          // Meta-BTS: x_n is the normalized deg-1 input at this point.
          const int meta_alpha =
              carried && args.state_meta_bts_alpha >= 0
                  ? args.state_meta_bts_alpha
                  : args.meta_bts_alpha;
          const double amplify = std::pow(2.0, meta_alpha);
          auto x_n = ciphertext->Clone();
          auto y1 = eval_bootstrap_synced(ciphertext);
          ++bootstraps;
          // r = x_n - y1 = -e1, formed at x_n's (deep) level: negate a fresh
          // clone of y1. This residual is itself only bootstrap error, so it
          // has a separate alignment knob from the rest of the circuit.
          auto residual = sub_aligned(
              x_n, y1->Clone(), args.meta_bts_residual_align_mode);
          cc->EvalMultInPlace(residual, amplify);  // the reserved live level
          ++ct_pt_muls;
          while (residual->GetNoiseScaleDeg() > 1) {
            cc->RescaleInPlace(residual);
          }
          debug_value_stats(residual, "meta_bts_residual." + what);
          auto y2 = eval_bootstrap_synced(residual);  // -e1*2^alpha + e2
          ++bootstraps;
          ++meta_bts_applied;
          auto correction = scaled_clone(y2, 1.0 / amplify);
          ciphertext = add_aligned(y1, correction);  // x_n + e2*2^-alpha
          ++counter;
          cc->EvalMultInPlace(ciphertext, bound);
          ++ct_pt_muls;
        } else {
          ciphertext = eval_bootstrap_synced(ciphertext);
          ++bootstraps;
          ++counter;
          cc->EvalMultInPlace(ciphertext, bound);
          ++ct_pt_muls;
        }
      });
      debug_value_stats(ciphertext, "post_bootstrap." + what);
      const double event_seconds = seconds_since(bootstrap_start);
      bootstrap_eval_seconds += event_seconds;
      bootstrap_events.push_back(BootstrapEvent{
          .checkpoint = what,
          .level_before = level,
          .level_after = static_cast<int>(ciphertext->GetLevel()),
          .requirement = requirement,
          .policy_headroom = policy_headroom,
          .physical_bootstraps = use_meta_bts ? 2 : 1,
          .carried = carried,
          .meta_bts = use_meta_bts,
          .bound = bound,
          .seconds = event_seconds,
      });
      log_phase("auto bootstrap " + what + " level_before=" + std::to_string(level) +
                " level_after=" + std::to_string(ciphertext->GetLevel()));
    };

    // Newton inverse-sqrt refinement: y <- 1.5*y + (-0.5*v*y)*(y*y). Each
    // iteration consumes 2 levels; under the MAXP=64 depth ceiling the
    // iterate is refreshed mid-run whenever it cannot cover the next
    // iteration plus headroom (v_neg_half stays at its low input level).
    auto newton_refine = [&](Ciphertext<DCRTPoly> y,
                             const Ciphertext<DCRTPoly>& v_neg_half,
                             int iterations, const std::string& tag, int& counter) {
      for (int iteration = 0; iteration < iterations; ++iteration) {
        maybe_bootstrap(y, 2, tag + ".iter" + std::to_string(iteration), counter);
        ++ct_ct_muls;
        auto y_squared = cc->EvalMult(y, y);
        auto vy = mul_aligned(v_neg_half->Clone(), y->Clone());
        auto product = mul_aligned(vy, y_squared);
        auto y_scaled = scaled_clone(y, 1.5);
        y = add_aligned(y_scaled, product);
      }
      return y;
    };

    // -----------------------------------------------------------------------
    // Host-side plaintext constants (folds validated by simulation).
    // -----------------------------------------------------------------------
    const int d_model = dims_payload.d_model;
    const int d_inner = dims_payload.d_inner;
    const int heads = dims_payload.num_heads;
    const int head_dim = dims_payload.head_dim;
    const int state_size = dims_payload.state_size;
    const int conv_dim = dims_payload.conv_dim;
    const int conv_kernel = dims_payload.conv_kernel;
    const int proj_dim = dims_payload.proj_dim;
    const int group_count = packing.group_count;
    const int group_heads = packing.group_heads;
    const int group_block = packing.group_block;
    const int xbc0 = packing.xbc0;
    const int dt0 = packing.dt0;
    const std::size_t batch = static_cast<std::size_t>(batch_size);

    auto affine = [](double lo, double hi) {
      return std::pair<double, double>{2.0 / (hi - lo), -(lo + hi) / (hi - lo)};
    };
    // Norms run on V = sum(x^2) + width*eps; rsqrt(V) = rsqrt(var+eps)/sqrt(w).
    auto make_layer_plan = [&](const M1Payload& payload) {
      LayerPlan plan;
      plan.eps_block = payload.eps_block;
      plan.state_abs_max = payload.state_abs_max;
      if (payload.state_head_abs_max.size() == static_cast<std::size_t>(heads)) {
        plan.state_group_abs_max.reserve(static_cast<std::size_t>(group_count));
        for (int group = 0; group < group_count; ++group) {
          const auto begin = payload.state_head_abs_max.begin() + group * group_heads;
          plan.state_group_abs_max.push_back(
              *std::max_element(begin, begin + group_heads));
        }
      }
      if (args.normalized_recurrent_state) {
        if (plan.state_group_abs_max.size() != static_cast<std::size_t>(group_count)) {
          throw std::runtime_error(
              "normalized recurrent state requires one calibrated bound per head group");
        }
        plan.state_group_scales.reserve(static_cast<std::size_t>(group_count));
        plan.normalized_state_x_masks.reserve(static_cast<std::size_t>(group_count));
        plan.normalized_state_readout_masks.reserve(
            static_cast<std::size_t>(group_count));
        for (int group = 0; group < group_count; ++group) {
          const double measured =
              plan.state_group_abs_max[static_cast<std::size_t>(group)];
          if (!std::isfinite(measured) || measured < 0.0) {
            throw std::runtime_error(
                "normalized recurrent state received an invalid calibrated bound");
          }
          const double scale = std::max(measured, 1.0e-6);
          plan.state_group_scales.push_back(scale);
          std::vector<double> x_mask(batch, 0.0);
          for (int slot = 0; slot < group_block; ++slot) {
            x_mask[static_cast<std::size_t>(group * group_block + slot)] =
                1.0 / scale;
          }
          plan.normalized_state_x_masks.push_back(std::move(x_mask));
          std::vector<double> readout_mask(batch, 0.0);
          for (int slot = 0; slot < group_block; ++slot) {
            readout_mask[static_cast<std::size_t>(slot)] = scale;
          }
          plan.normalized_state_readout_masks.push_back(
              std::move(readout_mask));
        }
        // maybe_bootstrap observes the stored coordinate system. Each state
        // ciphertext is now bounded by one; its original scale is folded
        // into the update/readout masks above.
        plan.state_abs_max = 1.0;
        std::fill(plan.state_group_abs_max.begin(),
                  plan.state_group_abs_max.end(), 1.0);
      }
      plan.fifo_abs_max = payload.fifo_abs_max;
      plan.checkpoint_abs_max = payload.checkpoint_abs_max;
      if (const auto y_bound = plan.checkpoint_abs_max.find("y");
          y_bound != plan.checkpoint_abs_max.end()) {
        plan.y_scale = std::max(1.0, y_bound->second);
      }
      plan.eps_gated = payload.eps_gated;
      const auto& p_conv = payload.polys.at("conv_silu");
      const auto& p_gate = payload.polys.at("gate_silu");
      const auto& p_dt = payload.polys.at("dt_softplus");
      const auto& p_exp = payload.polys.at("decay_exp");
      const auto& p_rms = payload.polys.at("rms_invsqrt");
      const auto& p_gated = payload.polys.at("gated_rms_invsqrt");
      const auto [a_conv, b_conv] = affine(p_conv.lo, p_conv.hi);
      const auto [a_gate, b_gate] = affine(p_gate.lo, p_gate.hi);
      const auto [a_dt, b_dt] = affine(p_dt.lo, p_dt.hi);
      const auto [a_exp_raw, b_exp] = affine(p_exp.lo, p_exp.hi);
      const double a_exp = a_exp_raw / std::pow(2.0, p_exp.squarings);
      const auto [a_rms, b_rms] = affine(p_rms.lo, p_rms.hi);
      plan.b_gate = b_gate;
      plan.b_exp = b_exp;
      plan.b_rms = b_rms;
      plan.a_rms_v = a_rms / static_cast<double>(d_model);
      plan.rms_iterations = p_rms.iterations;
      plan.gated_iterations = p_gated.iterations;
      plan.exp_squarings = p_exp.squarings;
      plan.conv_coeffs = p_conv.coeffs;
      plan.gate_coeffs = p_gate.coeffs;
      if (p_gated.kind == "sq-poly-newton") {
        for (double& coefficient : plan.gate_coeffs) {
          coefficient /= plan.y_scale;
        }
      }
      plan.dt_coeffs = p_dt.coeffs;
      plan.exp_coeffs = p_exp.coeffs;
      plan.rms_coeffs = p_rms.coeffs;
      for (double& coefficient : plan.rms_coeffs) {
        coefficient *= p_rms.damping / std::sqrt(static_cast<double>(d_model));
      }
      if (p_gated.kind == "sq-poly-newton") {
        const auto [a_gated, b_gated] = affine(p_gated.lo, p_gated.hi);
        plan.a_gated_v = a_gated / static_cast<double>(d_inner);
        plan.b_gated = b_gated;
        plan.gated_coeffs = p_gated.coeffs;
        plan.gated_damping_mean = p_gated.damping;
      } else {
        plan.gated_guess_v =
            p_gated.guess / std::sqrt(static_cast<double>(d_inner));
      }

      const auto& in_proj_w = payload.tensors.at("in_proj_w");
      const auto& block_norm_w = payload.tensors.at("block_norm_w");
      plan.in_w_folded.assign(in_proj_w.size(), 0.0);
      for (int output = 0; output < proj_dim; ++output) {
        for (int input = 0; input < d_model; ++input) {
          const auto index = static_cast<std::size_t>(output) * d_model + input;
          plan.in_w_folded[index] =
              in_proj_w[index] * block_norm_w[static_cast<std::size_t>(input)] *
              std::sqrt(static_cast<double>(d_model));
        }
      }
      const auto& out_proj_w = payload.tensors.at("out_proj_w");
      const auto& gated_norm_w = payload.tensors.at("gated_norm_w");
      const double gated_weight_fold =
          p_gated.kind == "sq-poly-newton"
              ? plan.y_scale
              : std::sqrt(static_cast<double>(d_inner));
      plan.out_w_folded.assign(out_proj_w.size(), 0.0);
      for (int output = 0; output < d_model; ++output) {
        for (int input = 0; input < d_inner; ++input) {
          const auto index = static_cast<std::size_t>(output) * d_inner + input;
          plan.out_w_folded[index] =
              out_proj_w[index] * gated_norm_w[static_cast<std::size_t>(input)] *
              gated_weight_fold;
        }
      }

      const auto& conv_w = payload.tensors.at("conv_w");  // (conv_dim, conv_kernel)
      const auto& conv_b = payload.tensors.at("conv_b");
      // All packed-layout vectors are tiled once per stream stride (a single
      // copy when streams == 1).
      plan.conv_tap_masks.assign(static_cast<std::size_t>(conv_kernel), {});
      for (int tap = 0; tap < conv_kernel; ++tap) {
        std::vector<double> mask(batch, 0.0);
        for (int stream = 0; stream < args.streams; ++stream) {
          const auto base = static_cast<std::size_t>(stream * stream_stride);
          for (int channel = 0; channel < conv_dim; ++channel) {
            mask[base + static_cast<std::size_t>(xbc0 + channel)] =
                a_conv * conv_w[static_cast<std::size_t>(channel) * conv_kernel + tap];
          }
        }
        plan.conv_tap_masks[static_cast<std::size_t>(tap)] = std::move(mask);
      }
      plan.conv_const.assign(batch, b_conv);
      for (int stream = 0; stream < args.streams; ++stream) {
        const auto base = static_cast<std::size_t>(stream * stream_stride);
        for (int channel = 0; channel < conv_dim; ++channel) {
          plan.conv_const[base + static_cast<std::size_t>(xbc0 + channel)] =
              a_conv * conv_b[static_cast<std::size_t>(channel)] + b_conv;
        }
      }

      plan.gate_mask.assign(batch, 0.0);
      const auto& dt_bias = payload.tensors.at("dt_bias");
      plan.dt_mask.assign(batch, 0.0);
      plan.dt_const.assign(batch, 0.0);
      const auto& a_log = payload.tensors.at("a_log");
      plan.a_vec.assign(batch, 0.0);
      const auto& d_skip = payload.tensors.at("d_skip");
      plan.d_vec.assign(batch, 0.0);
      for (int stream = 0; stream < args.streams; ++stream) {
        const auto base = static_cast<std::size_t>(stream * stream_stride);
        for (int slot = 0; slot < d_inner; ++slot) {
          plan.gate_mask[base + static_cast<std::size_t>(slot)] = a_gate;
        }
        for (int head = 0; head < heads; ++head) {
          plan.dt_mask[base + static_cast<std::size_t>(dt0 + head)] = a_dt;
          plan.dt_const[base + static_cast<std::size_t>(dt0 + head)] =
              a_dt * dt_bias[static_cast<std::size_t>(head)] + b_dt;
          plan.a_vec[base + static_cast<std::size_t>(dt0 + head)] =
              a_exp * (-std::exp(a_log[static_cast<std::size_t>(head)]));
          for (int position = 0; position < head_dim; ++position) {
            plan.d_vec[base + static_cast<std::size_t>(head * head_dim + position)] =
                d_skip[static_cast<std::size_t>(head)];
          }
        }
      }
      plan.test_layer_output = payload.tensors.at("test_layer_output");

      // Segment requirements for the mid-circuit bootstrap checkpoints (same
      // formulas as estimate_levels, using this layer's fits).
      const int rms_depth = cheb_ps_depth(static_cast<int>(p_rms.coeffs.size()) - 1);
      const int conv_depth = cheb_ps_depth(static_cast<int>(p_conv.coeffs.size()) - 1);
      const int gate_depth = cheb_ps_depth(static_cast<int>(p_gate.coeffs.size()) - 1);
      const int dt_depth = cheb_ps_depth(static_cast<int>(p_dt.coeffs.size()) - 1);
      const int exp_depth = cheb_ps_depth(static_cast<int>(p_exp.coeffs.size()) - 1);
      const int norm_extra = args.streams > 1 ? 1 : 0;
      const int inv1 = 2 + norm_extra + rms_depth + 2 * p_rms.iterations;
      plan.req_residual = std::max(inv1, 1) + 2;
      plan.req_proj = 1 + std::max(conv_depth, std::max(gate_depth + 2, dt_depth + 1));
      plan.req_fifo = 2 + conv_depth;
      plan.req_conv = args.replicated_state_blocks ? 7 : 6;
      plan.req_dt = 2 + exp_depth + p_exp.squarings;
      plan.req_decay = 3;
      plan.req_state_pre = 5;
      plan.req_state_tail = 4;
      plan.req_y = 4 + norm_extra;
      plan.req_out = 2;
      return plan;
    };
    std::vector<LayerPlan> layer_plans;
    layer_plans.reserve(layer_payloads.size());
    for (const auto& layer_payload : layer_payloads) {
      layer_plans.push_back(make_layer_plan(layer_payload));
      std::string layer_text = std::to_string(layer_plans.size() - 1);
      if (layer_text.size() < 2) {
        layer_text.insert(layer_text.begin(), '0');
      }
      layer_plans.back().cache_prefix = "L" + layer_text + ".";
    }

    // Final RMSNorm constants (full chain only). No projection follows norm_f,
    // so the inverse is applied directly and final_norm_w (with the sqrt(w)
    // variance fold) is a plaintext multiply. The inverse-sqrt uses the last
    // layer's rms_invsqrt fit: its calibrated variance domain is the closest
    // available to the final hidden state (no dedicated fit is exported).
    std::vector<double> final_rms_coeffs;
    double final_a_rms_v = 0.0;
    double final_b_rms = 0.0;
    double final_norm_scale = 1.0;
    int final_rms_iterations = 0;
    std::vector<double> final_w_vec;
    if (full_chain) {
      const auto& p_rms_final = layer_payloads.back().polys.at("rms_invsqrt");
      const auto [a_rms_f, b_rms_f] = affine(p_rms_final.lo, p_rms_final.hi);
      if (const auto output_bound = layer_plans.back().checkpoint_abs_max.find("output");
          output_bound != layer_plans.back().checkpoint_abs_max.end()) {
        final_norm_scale = std::max(1.0, output_bound->second * args.bootstrap_norm_margin);
      }
      // If h' = h/B, then mean(h^2) = B^2*mean(h'^2), while the inverse
      // required by RMSNorm is 1/sqrt(sum(h'^2)) = B/sqrt(sum(h^2)). Reuse
      // the certified original-domain polynomial by scaling its affine input
      // by B^2 and its output coefficients by B.
      final_a_rms_v =
          a_rms_f * final_norm_scale * final_norm_scale /
          static_cast<double>(d_model);
      final_b_rms = b_rms_f;
      final_rms_iterations = p_rms_final.iterations;
      final_rms_coeffs = p_rms_final.coeffs;
      for (double& coefficient : final_rms_coeffs) {
        coefficient *= p_rms_final.damping * final_norm_scale /
                       std::sqrt(static_cast<double>(d_model));
      }
      final_w_vec.assign(batch, 0.0);
      for (int slot = 0; slot < d_model; ++slot) {
        final_w_vec[static_cast<std::size_t>(slot)] =
            chain.final_norm_w[static_cast<std::size_t>(slot)] *
            std::sqrt(static_cast<double>(d_model));
      }
    }

    std::vector<double> group_mask(batch, 0.0);
    for (int slot = 0; slot < group_block; ++slot) {
      group_mask[static_cast<std::size_t>(slot)] = 1.0;
    }
    // Replicated out_proj leaves fold partials beyond slot d_model (unlike
    // the legacy path, whose masks land every term in [0, m)); the residual
    // handoff must stay clean for the next layer's variance sum, so the
    // folded result is masked to [0, d_model). Costs one ct-pt level on the
    // out lineage only when replicas > 1 (absorbed by the runtime policy).
    std::vector<double> out_clean_mask(batch, 0.0);
    for (int slot = 0; slot < d_model; ++slot) {
      out_clean_mask[static_cast<std::size_t>(slot)] = 1.0;
    }
    auto unit_mask = [&](int slot) {
      std::vector<double> mask(batch, 0.0);
      mask[static_cast<std::size_t>(slot)] = 1.0;
      return mask;
    };
    auto block_mask = [&](int start, int length) {
      std::vector<double> mask(batch, 0.0);
      for (int slot = 0; slot < length; ++slot) {
        mask[static_cast<std::size_t>(start + slot)] = 1.0;
      }
      return mask;
    };
    const auto bc_source_mask = block_mask(packing.b_base, 2 * state_size);
    std::vector<double> state_seed_mask(batch, 0.0);
    for (int state = 0; state < state_size; ++state) {
      state_seed_mask[static_cast<std::size_t>(state * group_block)] = 1.0;
    }

    // -----------------------------------------------------------------------
    // Encode thread-safety self test. FIDESlib/OpenFHE MakeCKKSPackedPlaintext
    // on a shared CryptoContext is expected to be read-only but is unproven:
    // encode the same diagonals serially and in parallel and compare decoded
    // values exactly. Any mismatch or exception falls back to serial encoding
    // with a logged warning (crash-type failures cannot be intercepted; the
    // default --encode-threads 1 never runs this test). Runs before the cache
    // build so the one-time build can use the pool too.
    // -----------------------------------------------------------------------
    int effective_encode_threads = 1;
    std::string encode_selftest_result = "skipped";
    if (args.encode_threads > 1) {
      const auto& test_weights = layer_plans.front().in_w_folded;
      const auto test_giants = slot_bsgs_giant_with_zero(d_model, proj_dim, kBabyStepIn);
      std::vector<std::pair<int, int>> test_diagonals;  // (giant, offset)
      for (std::size_t giant_index = 0;
           giant_index < test_giants.size() &&
           test_diagonals.size() < static_cast<std::size_t>(kEncodeSelfTestDiagonals);
           ++giant_index) {
        for (int baby = 0;
             baby < kBabyStepIn &&
             test_diagonals.size() < static_cast<std::size_t>(kEncodeSelfTestDiagonals);
             ++baby) {
          const int giant = test_giants[giant_index];
          auto mask = slot_bsgs_pre_mask(test_weights, d_model, proj_dim, batch_size,
                                         stream_stride, giant, giant + baby);
          if (std::all_of(mask.begin(), mask.end(),
                          [](double value) { return value == 0.0; })) {
            continue;
          }
          test_diagonals.emplace_back(giant, giant + baby);
        }
      }
      try {
        auto decoded_values = [&](const Plaintext& plain) {
          auto handle = plain;
          handle->SetLength(static_cast<std::size_t>(batch_size));
          auto values = handle->GetRealPackedValue();
          values.resize(static_cast<std::size_t>(batch_size));
          return values;
        };
        std::vector<std::vector<double>> serial_values;
        serial_values.reserve(test_diagonals.size());
        for (const auto& [giant, offset] : test_diagonals) {
          serial_values.push_back(decoded_values(make_plain(slot_bsgs_pre_mask(
              test_weights, d_model, proj_dim, batch_size, stream_stride, giant, offset))));
        }
        std::vector<Plaintext> parallel_plains(test_diagonals.size());
        const auto worker_count = std::min<std::size_t>(
            static_cast<std::size_t>(args.encode_threads), test_diagonals.size());
        std::vector<std::exception_ptr> worker_errors(worker_count);
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
          workers.emplace_back([&, worker]() {
            try {
              for (std::size_t job = worker; job < test_diagonals.size();
                   job += worker_count) {
                auto mask = slot_bsgs_pre_mask(test_weights, d_model, proj_dim, batch_size,
                                               stream_stride, test_diagonals[job].first,
                                               test_diagonals[job].second);
                auto plain = cc->MakeCKKSPackedPlaintext(mask);
                plain->SetLength(static_cast<size_t>(batch_size));
                parallel_plains[job] = plain;
              }
            } catch (...) {
              worker_errors[worker] = std::current_exception();
            }
          });
        }
        for (auto& worker : workers) {
          worker.join();
        }
        for (const auto& error : worker_errors) {
          if (error) {
            std::rethrow_exception(error);
          }
        }
        bool matches = true;
        for (std::size_t job = 0; job < test_diagonals.size() && matches; ++job) {
          if (!parallel_plains[job]) {
            matches = false;
            break;
          }
          const auto parallel_decoded = decoded_values(parallel_plains[job]);
          matches = parallel_decoded == serial_values[job];
        }
        if (matches) {
          effective_encode_threads = args.encode_threads;
          encode_selftest_result = "passed";
          log_phase("encode self test passed threads=" +
                    std::to_string(effective_encode_threads) + " diagonals=" +
                    std::to_string(test_diagonals.size()));
        } else {
          encode_selftest_result = "failed-mismatch";
          log_phase("WARNING: parallel encode self test MISMATCH; falling back to serial "
                    "encoding (FIDESlib encode is not thread-safe on this build)");
        }
      } catch (const std::exception& exc) {
        encode_selftest_result = std::string("failed-exception: ") + exc.what();
        log_phase(std::string("WARNING: parallel encode self test threw (") + exc.what() +
                  "); falling back to serial encoding");
      }
    }

    // -----------------------------------------------------------------------
    // Plaintext cache: register every token-invariant plaintext with its
    // per-token reuse count, then encode greedily until the byte budget is
    // spent. BSGS diagonal tables (mode "full") are indexed giant-major/
    // baby-minor over the exact enumeration the evaluation walks.
    // -----------------------------------------------------------------------
    std::vector<std::vector<Plaintext>> in_proj_tables(layer_plans.size());
    std::vector<std::vector<Plaintext>> out_proj_tables(layer_plans.size());
    // key -> (layer, 0=in/1=out, table index) links resolved after selection.
    std::vector<std::tuple<std::string, std::size_t, int, std::size_t>> bsgs_links;
    double pt_cache_encode_seconds = 0.0;
    std::size_t pt_cache_entries_cached = 0;
    double pt_cache_bytes_cached = 0.0;
    if (pt_cache_mode != "off") {
      const int layers_count = static_cast<int>(layer_plans.size());
      // Shared select/broadcast masks (layer-independent slot patterns).
      if (args.replicated_state_blocks) {
        register_plain("mask.bc_source", args.streams * layers_count,
                       [&bc_source_mask]() { return bc_source_mask; });
        register_plain("mask.state_seed", 2 * args.streams * layers_count,
                       [&state_seed_mask]() { return state_seed_mask; });
      } else {
        for (int a = 0; a < state_size / group_heads; ++a) {
          for (int b = 0; b < group_heads; ++b) {
            const int slot = group_heads * a + group_block * b;
            register_plain("mask.unit." + std::to_string(slot), 2 * layers_count,
                           [unit_mask, slot]() { return unit_mask(slot); });
          }
        }
      }
      for (int h_local = 0; h_local < group_heads; ++h_local) {
        register_plain("mask.unit." + std::to_string(h_local),
                       2 * group_count * layers_count,
                       [unit_mask, h_local]() { return unit_mask(h_local); });
      }
      if (args.normalized_recurrent_state) {
        for (const auto& plan : layer_plans) {
          for (int group = 0; group < group_count; ++group) {
            register_plain(
                plan.cache_prefix + "normalized_state_x." +
                    std::to_string(group),
                1,
                [&plan, group]() {
                  return plan.normalized_state_x_masks[
                      static_cast<std::size_t>(group)];
                },
                args.pt_cache_weight_level);
            register_plain(
                plan.cache_prefix + "normalized_state_readout." +
                    std::to_string(group),
                1,
                [&plan, group]() {
                  return plan.normalized_state_readout_masks[
                      static_cast<std::size_t>(group)];
                },
                args.pt_cache_weight_level);
          }
        }
      } else {
        for (int group = 0; group < group_count; ++group) {
          register_plain("mask.xblock." + std::to_string(group), layers_count,
                         [block_mask, group_block, group]() {
                           return block_mask(group_block * group, group_block);
                         });
        }
        register_plain("mask.group", group_count * layers_count,
                       [&group_mask]() { return group_mask; });
      }
      if (args.streams > 1) {
        register_plain("mask.streambase", 2 * layers_count,
                       [&stream_base_mask]() { return stream_base_mask; });
      }
      if (full_chain) {
        register_plain("final.norm_w", 1, [&final_w_vec]() { return final_w_vec; });
      }
      // Per-layer token-invariant vectors.
      for (const auto& plan : layer_plans) {
        for (int tap = 0; tap < conv_kernel; ++tap) {
          register_plain(plan.cache_prefix + "tap" + std::to_string(tap), 1,
                         [&plan, tap]() {
                           return plan.conv_tap_masks[static_cast<std::size_t>(tap)];
                         });
        }
        register_plain(plan.cache_prefix + "conv_const", 1,
                       [&plan]() { return plan.conv_const; });
        register_plain(plan.cache_prefix + "gate_mask", 1,
                       [&plan]() { return plan.gate_mask; });
        register_plain(plan.cache_prefix + "dt_mask", 1, [&plan]() { return plan.dt_mask; });
        register_plain(plan.cache_prefix + "dt_const", 1,
                       [&plan]() { return plan.dt_const; });
        register_plain(plan.cache_prefix + "a_vec", 1, [&plan]() { return plan.a_vec; });
        register_plain(plan.cache_prefix + "d_vec", 1, [&plan]() { return plan.d_vec; });
      }
      // Replicated-BSGS combined masks (few, high value: register in every
      // caching mode; they replace the legacy diagonal tables).
      for (std::size_t layer = 0; layer < layer_plans.size(); ++layer) {
        const auto& plan = layer_plans[layer];
        if (rep_in.replicas > 1) {
          for (int k = 0; k < rep_in.per_replica; ++k) {
            register_plain(plan.cache_prefix + "bsgsrep_in." + std::to_string(k), 1,
                           [&plan, k, rep_in, proj_dim, d_model, batch_size]() {
                             return replicated_bsgs_pre_mask(
                                 plan.in_w_folded, proj_dim, d_model, k, rep_in,
                                 batch_size);
                           },
                           layer == 0 ? args.pt_cache_level
                                      : args.pt_cache_weight_level);
          }
        }
        if (rep_out.replicas > 1) {
          for (int k = 0; k < rep_out.per_replica; ++k) {
            register_plain(plan.cache_prefix + "bsgsrep_out." + std::to_string(k), 1,
                           [&plan, k, rep_out, d_model, d_inner, batch_size]() {
                             return replicated_bsgs_pre_mask(
                                 plan.out_w_folded, d_model, d_inner, k, rep_out,
                                 batch_size);
                           },
                           args.pt_cache_weight_level);
          }
        }
      }
      if (rep_out.replicas > 1) {
        register_plain("mask.out_clean", static_cast<int>(layer_plans.size()),
                       [&out_clean_mask]() { return out_clean_mask; });
      }
      // Legacy BSGS diagonal tables (the dominant cost; mode "full" only).
      if (pt_cache_mode == "full") {
        for (std::size_t layer = 0; layer < layer_plans.size(); ++layer) {
          const auto& plan = layer_plans[layer];
          if (rep_in.replicas > 1 && rep_out.replicas > 1) {
            break;  // both matmuls replicated: no legacy tables needed
          }
          const auto register_bsgs = [&](int which, const std::vector<double>& weights,
                                         int input_dim, int output_dim, int baby_step,
                                         std::vector<Plaintext>& table) {
            const auto giants = slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step);
            table.assign(giants.size() * static_cast<std::size_t>(baby_step), Plaintext());
            for (std::size_t giant_index = 0; giant_index < giants.size(); ++giant_index) {
              const int giant = giants[giant_index];
              for (int baby = 0; baby < baby_step; ++baby) {
                const int offset = giant + baby;
                auto mask = slot_bsgs_pre_mask(weights, input_dim, output_dim, batch_size,
                                               stream_stride, giant, offset);
                if (std::all_of(mask.begin(), mask.end(),
                                [](double value) { return value == 0.0; })) {
                  continue;
                }
                const auto table_index =
                    giant_index * static_cast<std::size_t>(baby_step) +
                    static_cast<std::size_t>(baby);
                const std::string key = plan.cache_prefix +
                                        (which == 0 ? "bsgs_in." : "bsgs_out.") +
                                        std::to_string(table_index);
                register_plain(
                    key, 1,
                    [&weights, input_dim, output_dim, giant, offset,
                     this_batch = batch_size, this_stride = stream_stride]() {
                      return slot_bsgs_pre_mask(weights, input_dim, output_dim,
                                                this_batch, this_stride, giant,
                                                offset);
                    });
                bsgs_links.emplace_back(key, layer, which, table_index);
              }
            }
          };
          if (rep_in.replicas <= 1) {
            register_bsgs(0, plan.in_w_folded, d_model, proj_dim, kBabyStepIn,
                          in_proj_tables[layer]);
          }
          if (rep_out.replicas <= 1) {
            register_bsgs(1, plan.out_w_folded, d_inner, d_model, kBabyStepOut,
                          out_proj_tables[layer]);
          }
        }
      }
      // Greedy selection: reuse count desc, registration order asc.
      std::vector<std::map<std::string, PlainCacheEntry>::iterator> selection;
      selection.reserve(plain_cache.size());
      for (auto entry = plain_cache.begin(); entry != plain_cache.end(); ++entry) {
        selection.push_back(entry);
      }
      std::stable_sort(selection.begin(), selection.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs->second.uses_per_token != rhs->second.uses_per_token) {
          return lhs->second.uses_per_token > rhs->second.uses_per_token;
        }
        return lhs->second.order < rhs->second.order;
      });
      std::vector<std::map<std::string, PlainCacheEntry>::iterator> selected;
      selected.reserve(selection.size());
      double selected_bytes = 0.0;
      for (const auto& entry : selection) {
        if (selected_bytes + entry->second.estimated_bytes > pt_cache_budget_bytes) {
          continue;
        }
        selected.push_back(entry);
        selected_bytes += entry->second.estimated_bytes;
      }
      const auto encode_start = now();
      const auto to_encode = selected.size();
      log_phase("pt cache encode begin mode=" + pt_cache_mode +
                " registered=" + std::to_string(plain_cache.size()) +
                " selected=" + std::to_string(to_encode) +
                " selected_gib=" +
                std::to_string(selected_bytes / (1024.0 * 1024.0 * 1024.0)) +
                " threads=" + std::to_string(effective_encode_threads));
      if (effective_encode_threads > 1 && to_encode > 1) {
        // One-time build through the (self-tested) worker pool: entries are
        // independent; each worker fills disjoint striped slots.
        const auto worker_count = std::min<std::size_t>(
            static_cast<std::size_t>(effective_encode_threads), to_encode);
        std::vector<std::exception_ptr> worker_errors(worker_count);
        std::vector<std::thread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
          workers.emplace_back([&, worker]() {
            try {
              for (std::size_t job = worker; job < to_encode; job += worker_count) {
                selected[job]->second.plain = encode_cache_plain(
                    selected[job]->second.build(), selected[job]->second.encode_level);
              }
            } catch (...) {
              worker_errors[worker] = std::current_exception();
            }
          });
        }
        for (auto& worker : workers) {
          worker.join();
        }
        for (const auto& error : worker_errors) {
          if (error) {
            std::rethrow_exception(error);
          }
        }
        pt_cache_entries_cached = to_encode;
        pt_cache_bytes_cached = selected_bytes;
      } else {
        for (const auto& entry : selected) {
          entry->second.plain = encode_cache_plain(entry->second.build(),
                                                   entry->second.encode_level);
          ++pt_cache_entries_cached;
          pt_cache_bytes_cached += entry->second.estimated_bytes;
          if (pt_cache_entries_cached % 1000 == 0) {
            log_phase("pt cache encode progress " +
                      std::to_string(pt_cache_entries_cached) + "/" +
                      std::to_string(to_encode));
          }
        }
      }
      // Wire the selected BSGS entries into the per-layer tables.
      for (const auto& [key, layer, which, table_index] : bsgs_links) {
        const auto entry = plain_cache.find(key);
        if (entry == plain_cache.end() || !entry->second.plain) {
          continue;
        }
        auto& table = which == 0 ? in_proj_tables[layer] : out_proj_tables[layer];
        table[table_index] = entry->second.plain;
      }
      pt_cache_encode_seconds = seconds_since(encode_start);
      log_phase("pt cache encode done cached=" + std::to_string(pt_cache_entries_cached) +
                "/" + std::to_string(plain_cache.size()) + " bytes_gib=" +
                std::to_string(pt_cache_bytes_cached /
                               (1024.0 * 1024.0 * 1024.0)));
    }
    // Wire tables in every mode except "off": empty tables route the BSGS
    // eval through the miss counter and the parallel-encode pool ("masks"
    // mode previously bypassed both — the dgx chain-mode gap).
    for (std::size_t layer = 0; layer < layer_plans.size(); ++layer) {
      if (pt_cache_mode != "off") {
        layer_plans[layer].in_proj_table = &in_proj_tables[layer];
        layer_plans[layer].out_proj_table = &out_proj_tables[layer];
      }
    }

    // -----------------------------------------------------------------------
    // Shared-plaintext reuse check: cached Plaintexts are handed to EvalMult
    // repeatedly through the non-const Plaintext& interface. dgx decode
    // errors already matched with caching on; this cheap assertion pins the
    // "EvalMult does not mutate the plaintext" contract anyway: multiply a
    // reduced-level ciphertext by a cached plaintext and verify the decoded
    // plaintext is unchanged. On mismatch the cache is dropped (all entries
    // fall back to per-use encoding) with a logged warning.
    // -----------------------------------------------------------------------
    std::string pt_reuse_check_result = "skipped";
    if (pt_cache_entries_cached > 0) {
      try {
        Plaintext probe_plain;
        for (auto& [key, entry] : plain_cache) {
          if (entry.plain && entry.encode_level == 0) {
            probe_plain = entry.plain;
            break;
          }
        }
        if (probe_plain) {
          auto decoded_before = [&]() {
            auto handle = probe_plain;
            handle->SetLength(static_cast<std::size_t>(batch_size));
            auto values = handle->GetRealPackedValue();
            values.resize(static_cast<std::size_t>(batch_size));
            return values;
          }();
          // Reduced-level ciphertext: the risky case is the library adjusting
          // the plaintext's towers in place to match a consumed ciphertext.
          // Self-check ops, intentionally outside the telemetry counters.
          auto probe_ct = ones_ct->Clone();
          for (int reduction = 0; reduction < 3; ++reduction) {
            cc->EvalMultInPlace(probe_ct, 1.0);
          }
          auto product = cc->EvalMult(probe_ct, probe_plain);
          (void)product;
          auto decoded_after = [&]() {
            auto handle = probe_plain;
            handle->SetLength(static_cast<std::size_t>(batch_size));
            auto values = handle->GetRealPackedValue();
            values.resize(static_cast<std::size_t>(batch_size));
            return values;
          }();
          if (decoded_before == decoded_after) {
            pt_reuse_check_result = "passed";
            log_phase("pt reuse check passed (EvalMult left the shared plaintext intact)");
          } else {
            pt_reuse_check_result = "failed-mutation";
            log_phase("WARNING: EvalMult MUTATED a shared plaintext; dropping the "
                      "plaintext cache (per-use encoding only)");
            for (auto& [key, entry] : plain_cache) {
              entry.plain = Plaintext();
            }
            for (auto& table : in_proj_tables) {
              std::fill(table.begin(), table.end(), Plaintext());
            }
            for (auto& table : out_proj_tables) {
              std::fill(table.begin(), table.end(), Plaintext());
            }
            pt_cache_entries_cached = 0;
            pt_cache_bytes_cached = 0.0;
          }
        }
      } catch (const std::exception& exc) {
        pt_reuse_check_result = std::string("failed-exception: ") + exc.what();
        log_phase(std::string("WARNING: pt reuse check threw (") + exc.what() + ")");
      }
    }

    // -----------------------------------------------------------------------
    // Packing primitives.
    // -----------------------------------------------------------------------
    // Slot (base + n) -> slot group_block*n for n in [0, state), then fill the
    // 512 block by doubling. Rotations are decomposed as baby (b = n mod 8)
    // plus giant (a = n div 8) so the shared key set stays small; the slot map
    // is identical to 128 single mask+rotate pairs with indices
    // base - (group_block-1)*n.
    auto place_state_blocks = [&](const Ciphertext<DCRTPoly>& conv_packed, int base) {
      const int stride = group_block - 1;
      const int giant_stride = group_heads * stride;
      std::vector<Ciphertext<DCRTPoly>> baby_rotations;
      baby_rotations.reserve(static_cast<std::size_t>(group_heads));
      for (int b = 0; b < group_heads; ++b) {
        baby_rotations.push_back(rotate(conv_packed, base - stride * b));
      }
      Ciphertext<DCRTPoly> accumulator;
      bool has_accumulator = false;
      for (int a = 0; a < state_size / group_heads; ++a) {
        Ciphertext<DCRTPoly> inner;
        bool has_inner = false;
        for (int b = 0; b < group_heads; ++b) {
          const int slot = group_heads * a + group_block * b;
          auto term = mul_mask(baby_rotations[static_cast<std::size_t>(b)],
                               "mask.unit." + std::to_string(slot), unit_mask(slot));
          if (!has_inner) {
            inner = term;
            has_inner = true;
          } else {
            ++adds;
            inner = cc->EvalAdd(inner, term);
          }
        }
        inner = rotate(inner, -giant_stride * a);
        if (!has_accumulator) {
          accumulator = inner;
          has_accumulator = true;
        } else {
          ++adds;
          accumulator = cc->EvalAdd(accumulator, inner);
        }
      }
      return doubling_fill(accumulator, 1, int_log2(group_block));
    };

    // The contiguous B/C source has already been selected. Copy the branch at
    // offsets (group_block-1)*2^k: source element n then lands at the first
    // slot of state block n. One seed mask removes every other copied value.
    auto place_state_blocks_replicated =
        [&](const Ciphertext<DCRTPoly>& bc_source, int base) {
          const int stride = group_block - 1;
          auto replicated = rotate(bc_source, base);
          for (int step = 1; step < state_size; step *= 2) {
            auto copy = rotate(replicated, -stride * step);
            ++adds;
            replicated = cc->EvalAdd(replicated, copy);
          }
          auto seeds = mul_mask(replicated, "mask.state_seed", state_seed_mask);
          return doubling_fill(seeds, 1, int_log2(group_block));
        };

    // Head value of group g at proj slot dt0 + group_heads*g + h_local ->
    // all slots n*group_block + h_local*head_dim + p.
    auto place_heads = [&](const Ciphertext<DCRTPoly>& source, int group) {
      auto shifted = rotate(source, dt0 + group_heads * group);
      Ciphertext<DCRTPoly> accumulator;
      bool has_accumulator = false;
      for (int h_local = 0; h_local < group_heads; ++h_local) {
        auto term = mul_mask(shifted, "mask.unit." + std::to_string(h_local),
                             unit_mask(h_local));
        term = rotate(term, -(head_dim - 1) * h_local);
        if (!has_accumulator) {
          accumulator = term;
          has_accumulator = true;
        } else {
          ++adds;
          accumulator = cc->EvalAdd(accumulator, term);
        }
      }
      accumulator = doubling_fill(accumulator, 1, int_log2(head_dim));
      return doubling_fill(accumulator, group_block, int_log2(state_size));
    };

    auto expand_x = [&](const Ciphertext<DCRTPoly>& conv_packed, int group,
                        const LayerPlan& plan) {
      std::vector<double> legacy_mask;
      const std::vector<double>* mask = nullptr;
      if (args.normalized_recurrent_state) {
        mask = &plan.normalized_state_x_masks[static_cast<std::size_t>(group)];
      } else {
        legacy_mask = block_mask(group_block * group, group_block);
        mask = &legacy_mask;
      }
      const std::string key =
          args.normalized_recurrent_state
              ? plan.cache_prefix + "normalized_state_x." +
                    std::to_string(group)
              : "mask.xblock." + std::to_string(group);
      auto masked = mul_mask(conv_packed, key, *mask);
      masked = rotate(masked, group_block * group);
      return doubling_fill(masked, group_block, int_log2(state_size));
    };

    // -----------------------------------------------------------------------
    // Input-replicated BSGS matmul: schedule verified bitwise against the
    // spec simulator (fhemamba/src/fhemamba/bsgs_layout.py). Extension, fill,
    // and fold are rotate/add only (0 levels); the matmul is one ct-pt level,
    // so the level ledger matches the legacy path. Requires the input clean
    // outside [0, input_dim) (established invariants for hidden and y).
    // -----------------------------------------------------------------------
    auto replicated_bsgs = [&](const Ciphertext<DCRTPoly>& input_ct,
                               const std::vector<double>& weights, int output_dim,
                               int input_dim, const ReplicatedShape& shape,
                               const std::string& key_prefix) -> Ciphertext<DCRTPoly> {
      // In-window cyclic self-extension of the period-n input tile.
      auto extended = input_ct;
      for (int t = 1; t < shape.reps; ++t) {
        extended = add_aligned(extended, rotate(input_ct, -t * input_dim));
      }
      // Identical replica fill at window stride.
      auto replicated = extended;
      for (int j = 1; j < shape.replicas + shape.guard_windows; ++j) {
        replicated = add_aligned(replicated, rotate(extended, -j * shape.window));
      }
      // The measured path uses one roll per diagonal group. The opt-in true
      // BSGS path reuses baby rotations and pre-rotates each plaintext mask
      // before one rotation per giant group.
      Ciphertext<DCRTPoly> accumulator;
      bool has_accumulator = false;
      if (shape.baby_step <= 1) {
        for (int k = 0; k < shape.per_replica; ++k) {
          auto mask = replicated_bsgs_pre_mask(weights, output_dim, input_dim, k,
                                               shape, batch_size);
          if (std::all_of(mask.begin(), mask.end(),
                          [](double value) { return value == 0.0; })) {
            continue;
          }
          auto rolled = k == 0 ? replicated : rotate(replicated, k * shape.replicas);
          auto term = mul_mask(rolled, key_prefix + std::to_string(k), mask);
          if (!has_accumulator) {
            accumulator = term;
            has_accumulator = true;
          } else {
            accumulator = add_aligned(accumulator, term);
          }
        }
      } else {
        std::vector<Ciphertext<DCRTPoly>> babies;
        babies.reserve(static_cast<std::size_t>(shape.baby_step));
        babies.push_back(replicated);
        for (int baby = 1; baby < shape.baby_step; ++baby) {
          babies.push_back(rotate(replicated, baby * shape.replicas));
        }
        const int giant_count =
            (shape.per_replica + shape.baby_step - 1) / shape.baby_step;
        for (int giant = 0; giant < giant_count; ++giant) {
          Ciphertext<DCRTPoly> inner;
          bool has_inner = false;
          for (int baby = 0; baby < shape.baby_step; ++baby) {
            const int k = giant * shape.baby_step + baby;
            if (k >= shape.per_replica) {
              break;
            }
            auto mask = replicated_bsgs_pre_mask(
                weights, output_dim, input_dim, k, shape, batch_size);
            if (std::all_of(mask.begin(), mask.end(),
                            [](double value) { return value == 0.0; })) {
              continue;
            }
            auto term = mul_mask(babies[static_cast<std::size_t>(baby)],
                                 key_prefix + std::to_string(k), mask);
            if (!has_inner) {
              inner = term;
              has_inner = true;
            } else {
              inner = add_aligned(inner, term);
            }
          }
          if (!has_inner) {
            continue;
          }
          if (giant != 0) {
            inner = rotate(inner, giant * shape.baby_step * shape.replicas);
          }
          if (!has_accumulator) {
            accumulator = inner;
            has_accumulator = true;
          } else {
            accumulator = add_aligned(accumulator, inner);
          }
        }
      }
      if (!has_accumulator) {
        throw std::runtime_error("replicated BSGS produced no terms");
      }
      // Fold windows into window 0 at stride window+1 (mirrors the spec
      // simulator's two fold branches).
      // Fold rotates LEFT (window j lands at window 0): positive indices.
      auto folded = accumulator;
      if ((shape.replicas & (shape.replicas - 1)) == 0) {
        for (int step = shape.window + 1; step < shape.replicas * (shape.window + 1);
             step *= 2) {
          folded = add_aligned(folded, rotate(folded, step));
        }
      } else {
        for (int j = 1; j < shape.replicas; ++j) {
          folded = add_aligned(folded, rotate(accumulator, j * (shape.window + 1)));
        }
      }
      return folded;
    };

    auto late_level_projection_input = [&](const Ciphertext<DCRTPoly>& input,
                                           const Ciphertext<DCRTPoly>& inverse,
                                           int linear_levels) {
      if (!args.projection_late_level || args.level_align_mode != "drop") {
        return input;
      }
      const int inverse_level = static_cast<int>(inverse->GetLevel());
      const int input_level = static_cast<int>(input->GetLevel());
      const int target_level = inverse_level - linear_levels;
      if (target_level <= input_level || target_level < 0) {
        return input;
      }
      auto dropped = input->Clone();
      dropped->SetLevel(static_cast<uint32_t>(target_level));
      ++direct_level_drops;
      ++projection_late_level_drops;
      return dropped;
    };

    // -----------------------------------------------------------------------
    // One full layer circuit (block norm ... residual add). State and conv
    // FIFO live in the per-layer runtime and stay ciphertext across tokens.
    // Mid-circuit bootstrap checkpoints (after in_proj, after conv_silu,
    // after dt, after decay, after the state update, before the gated norm,
    // before out_proj, plus the per-iteration Newton check) keep every
    // ciphertext lineage within the depth-44 MAXP=64 geometry.
    // -----------------------------------------------------------------------
    auto run_layer = [&](const LayerPlan& plan, LayerRuntime& runtime, int token_index,
                         int layer_index,
                         const Ciphertext<DCRTPoly>& hidden_ct, const std::string& tag,
                         int& layer_bootstraps) -> Ciphertext<DCRTPoly> {
      // This layer's measured carried bounds drive every state/FIFO refresh
      // inside the call (single-threaded, sequential -> a shared active value
      // is safe and avoids threading it through every maybe_bootstrap site).
      active_state_bound = plan.state_abs_max;
      active_state_group_bounds = plan.state_group_abs_max;
      active_fifo_bound = plan.fifo_abs_max;
      active_checkpoint_bounds = plan.checkpoint_abs_max;
      // Block RMSNorm inverse sqrt on V = sum(x^2) + d_model*eps; the sqrt(w)
      // and norm weights are folded into the in_proj plaintexts, so proj =
      // BSGS(h) * inv (inv is a uniform broadcast and commutes with matmul).
      auto inv_block = time_phase("block_norm", [&]() {
        ++ct_ct_muls;
        auto squared = cc->EvalMult(hidden_ct, hidden_ct);
        auto variance = norm_variance_sum(squared);
        variance = add_scalar(variance, d_model * plan.eps_block);
        auto u = add_scalar(scaled_clone(variance, plan.a_rms_v), plan.b_rms);
        auto guess = eval_chebyshev(u, plan.rms_coeffs);
        auto v_neg_half = scaled_clone(variance, -0.5);
        return newton_refine(guess, v_neg_half, plan.rms_iterations,
                             tag + "rms_newton", layer_bootstraps);
      });
      ckks_levels[tag + "block_inv"] = static_cast<int>(inv_block->GetLevel());
      debug_value_stats(inv_block, tag + "block_inv");

      auto proj_ct = time_phase("in_proj_bsgs", [&]() {
        const auto linear_input = late_level_projection_input(
            hidden_ct, inv_block, 1);
        Ciphertext<DCRTPoly> linear;
        if (rep_in.replicas > 1) {
          linear = replicated_bsgs(linear_input, plan.in_w_folded, proj_dim, d_model, rep_in,
                                   plan.cache_prefix + "bsgsrep_in.");
        } else {
          auto babies = slot_bsgs_precompute_baby_rotations(
              rotate, linear_input, kBabyStepIn);
          linear = slot_bsgs_linear_block0_from_babies(
              cc, rotate, babies, plan.in_w_folded, d_model, proj_dim, kBabyStepIn,
              batch_size, ct_pt_muls, adds, plan.in_proj_table, &pt_cache_hits,
              &pt_cache_misses, effective_encode_threads, stream_stride);
        }
        return mul_aligned(linear, inv_block);
      });
      // Checkpoint: proj feeds the conv FIFO and the conv/gate/dt branches.
      maybe_bootstrap(proj_ct, plan.req_proj, tag + "proj", layer_bootstraps);
      ckks_levels[tag + "proj"] = static_cast<int>(proj_ct->GetLevel());
      debug_value_stats(proj_ct, tag + "proj");

      // Conv FIFO: taps multiply xBC slots of the raw proj ciphertexts, the
      // silu-domain affine map is folded into taps/bias, and the poly output
      // is rotated from proj coordinates into the packed x|B|C layout.
      runtime.conv_fifo.push_back(proj_ct);
      if (static_cast<int>(runtime.conv_fifo.size()) > conv_kernel) {
        runtime.conv_fifo.erase(runtime.conv_fifo.begin());
      }
      for (std::size_t position = 0; position < runtime.conv_fifo.size(); ++position) {
        maybe_bootstrap(runtime.conv_fifo[position], plan.req_fifo,
                        tag + "fifo" + std::to_string(position), layer_bootstraps);
      }
      auto conv_u = time_phase("conv_fifo", [&]() {
        Ciphertext<DCRTPoly> accumulator;
        bool has_accumulator = false;
        const int fifo_size = static_cast<int>(runtime.conv_fifo.size());
        for (int position = 0; position < fifo_size; ++position) {
          const int tap = conv_kernel - fifo_size + position;
          auto term = mul_mask(runtime.conv_fifo[static_cast<std::size_t>(position)],
                               plan.cache_prefix + "tap" + std::to_string(tap),
                               plan.conv_tap_masks[static_cast<std::size_t>(tap)]);
          if (!has_accumulator) {
            accumulator = term;
            has_accumulator = true;
          } else {
            // FIFO entries can sit at different levels after a bootstrap
            // refresh, so align before adding.
            accumulator = add_aligned(accumulator, term);
          }
        }
        return add_const_vector(accumulator, plan.cache_prefix + "conv_const",
                                plan.conv_const);
      });
      auto conv_packed = time_phase("conv_silu_poly", [&]() {
        auto activated = eval_chebyshev(conv_u, plan.conv_coeffs);
        return rotate(activated, xbc0);
      });
      // Checkpoint: conv_silu output feeds x/B/C expands, updates, readout.
      maybe_bootstrap(conv_packed, plan.req_conv, tag + "conv_silu", layer_bootstraps);
      ckks_levels[tag + "conv_silu"] = static_cast<int>(conv_packed->GetLevel());
      debug_value_stats(conv_packed, tag + "conv_silu");

      auto gate_ct = time_phase("gate_silu_poly", [&]() {
        auto u = add_scalar(mul_mask(proj_ct, plan.cache_prefix + "gate_mask", plan.gate_mask),
                            plan.b_gate);
        return eval_chebyshev(u, plan.gate_coeffs);
      });
      ckks_levels[tag + "gate_silu"] = static_cast<int>(gate_ct->GetLevel());
      debug_value_stats(gate_ct, tag + "gate_silu");

      auto dt_ct = time_phase("dt_softplus_poly", [&]() {
        auto u = add_const_vector(mul_mask(proj_ct, plan.cache_prefix + "dt_mask", plan.dt_mask),
                                  plan.cache_prefix + "dt_const", plan.dt_const);
        auto root = eval_chebyshev(u, plan.dt_coeffs);
        ++ct_ct_muls;
        return cc->EvalMult(root, root);  // cheb-squared: softplus >= 0
      });
      // Checkpoint: dt feeds the decay polynomial (whose depth includes the
      // per-layer range-reduction squarings) and the dt expand.
      maybe_bootstrap(dt_ct, plan.req_dt, tag + "dt", layer_bootstraps);
      ckks_levels[tag + "dt"] = static_cast<int>(dt_ct->GetLevel());
      debug_value_stats(dt_ct, tag + "dt");

      auto decay_ct = time_phase("decay_exp_poly", [&]() {
        auto u = add_scalar(mul_mask(dt_ct, plan.cache_prefix + "a_vec", plan.a_vec),
                            plan.b_exp);
        auto value = eval_chebyshev(u, plan.exp_coeffs);
        for (int squaring = 0; squaring < plan.exp_squarings; ++squaring) {
          // Some layers need 12-15 range-reduction squarings because a few
          // heads have very negative A*dt. Do not wait until the final decay
          // ciphertext is at level ~40: refresh the bounded [0, 1] partial
          // exponential while it can still be bootstrapped reliably.
          const int remaining_squarings = plan.exp_squarings - squaring;
          maybe_bootstrap(value, remaining_squarings + plan.req_decay,
                          tag + "decay_sq" + std::to_string(squaring),
                          layer_bootstraps);
          ++ct_ct_muls;
          value = cc->EvalMult(value, value);
        }
        return value;
      });
      // Checkpoint: decay feeds the per-group expand and the state multiply.
      maybe_bootstrap(decay_ct, plan.req_decay, tag + "decay", layer_bootstraps);
      ckks_levels[tag + "decay"] = static_cast<int>(decay_ct->GetLevel());
      debug_value_stats(decay_ct, tag + "decay");

      // SSM mid-section. Each stream owns a full-batch state layout, so the
      // packed multi-stream conv/dt/decay ciphertexts are shifted to stream
      // base 0 (composite power-of-two rotations, no new keys), the existing
      // per-group expansions/update/readout run per stream on the stream's
      // replicated state ciphertexts, and each stream's packed y is shifted
      // back to its stride. streams == 1 executes exactly the previous code.
      Ciphertext<DCRTPoly> y_ssm;
      bool has_y = false;
      for (int stream = 0; stream < args.streams; ++stream) {
        const auto conv_stream =
            stream == 0 ? conv_packed
                        : rotate_composite(conv_packed, stream * stream_stride);
        const auto dt_stream =
            stream == 0 ? dt_ct : rotate_composite(dt_ct, stream * stream_stride);
        const auto decay_stream =
            stream == 0 ? decay_ct : rotate_composite(decay_ct, stream * stream_stride);
        auto bc_expanded = time_phase("bc_expand", [&]() {
          if (!args.replicated_state_blocks) {
            return std::make_pair(
                place_state_blocks(conv_stream, packing.b_base),
                place_state_blocks(conv_stream, packing.c_base));
          }
          auto bc_source =
              mul_mask(conv_stream, "mask.bc_source", bc_source_mask);
          return std::make_pair(
              place_state_blocks_replicated(bc_source, packing.b_base),
              place_state_blocks_replicated(bc_source, packing.c_base));
        });
        auto& b_expanded = bc_expanded.first;
        auto& c_expanded = bc_expanded.second;
        if (stream == 0) {
          ckks_levels[tag + "bc_expand"] = static_cast<int>(b_expanded->GetLevel());
          debug_value_stats(b_expanded, tag + "bc_expand");
        }
        Ciphertext<DCRTPoly> y_stream;
        bool has_y_stream = false;
        for (int group = 0; group < group_count; ++group) {
          const auto state_slot =
              static_cast<std::size_t>(stream * group_count + group);
          const std::string state_name =
              "state" + std::to_string(stream * group_count + group);
          auto x_group = time_phase("x_expand", [&]() {
            return expand_x(conv_stream, group, plan);
          });
          auto dt_group =
              time_phase("dt_expand", [&]() { return place_heads(dt_stream, group); });
          auto decay_group =
              time_phase("decay_expand", [&]() { return place_heads(decay_stream, group); });
          if (runtime.has_state) {
            maybe_bootstrap(runtime.state_cts[state_slot], plan.req_state_pre,
                            tag + state_name, layer_bootstraps);
          }
          time_phase("state_update", [&]() {
            auto dtx = mul_aligned(x_group, dt_group);
            auto update = mul_aligned(dtx, b_expanded->Clone());
            if (!runtime.has_state) {
              runtime.state_cts[state_slot] = update;
            } else {
              auto decayed = mul_aligned(decay_group, runtime.state_cts[state_slot]);
              runtime.state_cts[state_slot] = add_aligned(decayed, update);
            }
          });
          // The updated state feeds this token's readout and is the carried
          // lineage for the next token. Refresh after the recurrent multiply
          // so both consumers start from a usable level.
          const bool periodic_state_refresh =
              runtime.has_state && args.state_refresh_interval > 0 &&
              token_index % args.state_refresh_interval == 0;
          const int state_tail_requirement =
              runtime.has_state &&
                      (args.refresh_recurrent_state_post ||
                       args.refresh_recurrent_state_post_layers.count(layer_index) > 0 ||
                       periodic_state_refresh)
                  ? args.multiplicative_depth
                  : plan.req_state_tail;
          maybe_bootstrap(runtime.state_cts[state_slot], state_tail_requirement,
                          tag + "state_post" + std::to_string(stream * group_count + group),
                          layer_bootstraps);
          ckks_levels[tag + state_name] =
              static_cast<int>(runtime.state_cts[state_slot]->GetLevel());
          debug_value_stats(runtime.state_cts[state_slot], tag + state_name);
          auto y_group = time_phase("readout", [&]() {
            auto readout =
                mul_aligned(runtime.state_cts[state_slot]->Clone(), c_expanded->Clone());
            for (int k = 0; k < int_log2(state_size); ++k) {
              readout = add_aligned(readout, rotate(readout, group_block << k));
            }
            const auto& readout_mask =
                args.normalized_recurrent_state
                    ? plan.normalized_state_readout_masks[
                          static_cast<std::size_t>(group)]
                    : group_mask;
            const std::string readout_key =
                args.normalized_recurrent_state
                    ? plan.cache_prefix + "normalized_state_readout." +
                          std::to_string(group)
                    : "mask.group";
            auto masked = mul_mask(readout, readout_key, readout_mask);
            return rotate(masked, -group_block * group);
          });
          if (!has_y_stream) {
            y_stream = y_group;
            has_y_stream = true;
          } else {
            y_stream = add_aligned(y_stream, y_group);
          }
        }
        if (stream != 0) {
          y_stream = rotate_composite(y_stream, -stream * stream_stride);
        }
        if (!has_y) {
          y_ssm = y_stream;
          has_y = true;
        } else {
          y_ssm = add_aligned(y_ssm, y_stream);
        }
      }
      runtime.has_state = true;

      auto y_ct = time_phase("skip_gate", [&]() {
        auto skip = mul_mask(conv_packed, plan.cache_prefix + "d_vec", plan.d_vec);
        auto combined = add_aligned(y_ssm, skip);
        return mul_aligned(combined, gate_ct);
      });
      // Keep squared-polynomial payloads in a statically normalized y
      // coordinate. Its inverse scale is folded into out_proj, so the
      // RMSNorm result is unchanged while any y refresh stays near unit
      // magnitude.
      const double y_normalization =
          plan.gated_coeffs.empty() ? 1.0 : plan.y_scale;
      maybe_bootstrap(y_ct, plan.req_y,
                      tag + (plan.gated_coeffs.empty() ? "y" : "y_scaled"),
                      layer_bootstraps);
      ckks_levels[tag + "y"] = static_cast<int>(y_ct->GetLevel());
      debug_value_stats(y_ct, tag + "y");

      // Gated RMSNorm on V2 = sum(y^2) + d_inner*eps. New payloads evaluate
      // a polynomial approximation of v^(-1/4), square it, and apply four
      // Newton refinements. Legacy payloads retain the normalized
      // constant-guess path. The gated norm weights (and, for legacy sum
      // variance payloads, sqrt(w)) are folded into out_proj; the uniform
      // inverse is applied after the BSGS matmul.
      auto inv_gated = time_phase("gated_norm", [&]() {
        ++ct_ct_muls;
        auto squared = cc->EvalMult(y_ct, y_ct);
        auto variance = norm_variance_sum(squared);
        const double y_normalization_squared = y_normalization * y_normalization;
        variance = add_scalar(
            variance,
            d_inner * plan.eps_gated / y_normalization_squared);
        if (!plan.gated_coeffs.empty()) {
          auto u = add_scalar(
              scaled_clone(variance,
                           plan.a_gated_v * y_normalization_squared),
              plan.b_gated);
          const int gated_requirement =
              cheb_ps_depth(static_cast<int>(plan.gated_coeffs.size()) - 1) +
              1 + 2 * plan.gated_iterations;
          maybe_bootstrap(u, gated_requirement,
                          tag + "gated_poly_input", layer_bootstraps);
          auto quarter_root = eval_chebyshev(u, plan.gated_coeffs);
          ++ct_ct_muls;
          auto guess = cc->EvalMult(quarter_root, quarter_root);
          guess = scaled_clone(guess, plan.gated_damping_mean);
          // Refine against the mean variance, not its d_inner-wide sum.
          // Reconstructing the sum from a bootstrapped affine coordinate
          // amplifies its refresh error by d_inner before every Newton step.
          auto mean_neg_half = scaled_clone(
              add_scalar(u->Clone(), -plan.b_gated),
              -0.5 / (plan.a_gated_v * static_cast<double>(d_inner)));
          return newton_refine(guess, mean_neg_half, plan.gated_iterations,
                               tag + "gated_newton", layer_bootstraps);
        }
        const double c0 = plan.gated_guess_v;
        auto u_neg_half = scaled_clone(variance, -0.5 * c0 * c0);
        maybe_bootstrap(u_neg_half, kNewtonSegmentEstimate,
                        tag + "gated_variance", layer_bootstraps);
        // First iteration from z0=1 is affine in U: z1=1.5-0.5U.
        auto z_first = add_scalar(u_neg_half->Clone(), 1.5);
        auto z = newton_refine(z_first, u_neg_half, plan.gated_iterations - 1,
                               tag + "gated_newton", layer_bootstraps);
        return scaled_clone(z, c0);
      });
      ckks_levels[tag + "gated_inv"] = static_cast<int>(inv_gated->GetLevel());
      debug_value_stats(inv_gated, tag + "gated_inv");

      // Checkpoint: y again before out_proj (usually a no-op; fires only if
      // the gated-norm segment left it deeper than the BSGS entry allows).
      maybe_bootstrap(y_ct, plan.req_out, tag + "y_out", layer_bootstraps);
      auto out_ct = time_phase("out_proj_bsgs", [&]() {
        const int linear_levels = rep_out.replicas > 1 ? 2 : 1;
        const auto linear_input = late_level_projection_input(
            y_ct, inv_gated, linear_levels);
        Ciphertext<DCRTPoly> linear;
        if (rep_out.replicas > 1) {
          linear = replicated_bsgs(linear_input, plan.out_w_folded, d_model, d_inner, rep_out,
                                   plan.cache_prefix + "bsgsrep_out.");
          linear = mul_mask(linear, "mask.out_clean", out_clean_mask);
        } else {
          auto babies = slot_bsgs_precompute_baby_rotations(
              rotate, linear_input, kBabyStepOut);
          linear = slot_bsgs_linear_block0_from_babies(
              cc, rotate, babies, plan.out_w_folded, d_inner, d_model, kBabyStepOut,
              batch_size, ct_pt_muls, adds, plan.out_proj_table, &pt_cache_hits,
              &pt_cache_misses, effective_encode_threads, stream_stride);
        }
        return mul_aligned(linear, inv_gated);
      });
      auto layer_output = time_phase("residual_add", [&]() {
        return add_aligned(out_ct, hidden_ct->Clone());
      });
      ckks_levels[tag + "output"] = static_cast<int>(layer_output->GetLevel());
      debug_value_stats(layer_output, tag + "output");
      return layer_output;
    };

    // -----------------------------------------------------------------------
    // Token loop: per-layer state + conv FIFO stay ciphertext, residual is a
    // ciphertext handoff between layers, no intermediate decrypts.
    // -----------------------------------------------------------------------
    const auto eval_start = now();
    log_phase("token loop begin tokens=" + std::to_string(args.tokens) +
              " layers=" + std::to_string(layers_loaded));
    std::vector<LayerRuntime> layer_runtimes(layer_payloads.size());
    for (auto& runtime : layer_runtimes) {
      runtime.state_cts.resize(static_cast<std::size_t>(group_count * args.streams));
    }
    std::vector<Ciphertext<DCRTPoly>> token_outputs;
    std::vector<int> token_bootstrap_counts;
    std::map<std::string, int> summary_level_in;
    std::map<std::string, int> summary_level_out;
    std::map<std::string, double> debug_boundary_errors;
    std::map<std::string, int> summary_bootstraps;
    std::vector<int> autoregressive_selected_ids;
    std::vector<double> autoregressive_next_embedding;
    std::vector<std::vector<double>> autoregressive_decrypted_outputs(
        static_cast<std::size_t>(args.tokens));
    double autoregressive_client_seconds = 0.0;
    auto layer_key = [](int token, int layer) {
      std::string layer_text = std::to_string(layer);
      if (layer_text.size() < 2) {
        layer_text.insert(layer_text.begin(), '0');
      }
      return "t" + std::to_string(token) + ".L" + layer_text;
    };
    const std::vector<double>& input_vectors =
        args.autoregressive_client_loop ? chain.autoregressive_embeddings
        : chain_mode ? chain.input_embeddings
                   : layer_payloads.front().tensors.at("test_layer_input");

    for (int token = 0; token < args.tokens; ++token) {
      int token_bootstraps_total = 0;
      if (args.debug_client_reencrypt_before_token.count(token) > 0) {
        const auto reencrypt_start = now();
        for (std::size_t layer = 0; layer < layer_runtimes.size(); ++layer) {
          auto& runtime = layer_runtimes[layer];
          if (!runtime.has_state) {
            continue;
          }
          const std::string tag =
              layer_key(token, static_cast<int>(layer)) + ".client_reencrypt.";
          const auto reencrypt = [&](Ciphertext<DCRTPoly>& ciphertext,
                                     const std::string& name) {
            auto slots = decrypt_slots(cc, secret_key, ciphertext,
                                       static_cast<std::size_t>(batch_size));
            ciphertext = encrypt_values(slots);
            ++debug_client_reencrypt_ciphertexts;
            ckks_levels[tag + name] = static_cast<int>(ciphertext->GetLevel());
          };
          for (std::size_t state = 0; state < runtime.state_cts.size(); ++state) {
            reencrypt(runtime.state_cts[state], "state" + std::to_string(state));
          }
          for (std::size_t fifo = 0; fifo < runtime.conv_fifo.size(); ++fifo) {
            reencrypt(runtime.conv_fifo[fifo], "fifo" + std::to_string(fifo));
          }
        }
        debug_client_reencrypt_seconds += seconds_since(reencrypt_start);
        log_phase("DEBUG client re-encrypt before token " + std::to_string(token) +
                  " done ciphertexts=" +
                  std::to_string(debug_client_reencrypt_ciphertexts));
      }
      if (args.bootstrap_before_token.count(token) > 0) {
        for (std::size_t layer = 0; layer < layer_runtimes.size(); ++layer) {
          auto& runtime = layer_runtimes[layer];
          if (!runtime.has_state) {
            continue;
          }
          const auto& plan = layer_plans[layer];
          active_state_bound = plan.state_abs_max;
          active_state_group_bounds = plan.state_group_abs_max;
          active_fifo_bound = plan.fifo_abs_max;
          active_checkpoint_bounds = plan.checkpoint_abs_max;
          const std::string tag = layer_key(token, static_cast<int>(layer)) + ".scheduled.";
          for (std::size_t state = 0; state < runtime.state_cts.size(); ++state) {
            maybe_bootstrap(runtime.state_cts[state], args.multiplicative_depth,
                            tag + "state" + std::to_string(state),
                            token_bootstraps_total);
          }
          for (std::size_t fifo = 0; fifo < runtime.conv_fifo.size(); ++fifo) {
            maybe_bootstrap(runtime.conv_fifo[fifo], args.multiplicative_depth,
                            tag + "fifo" + std::to_string(fifo),
                            token_bootstraps_total);
          }
        }
        log_phase("bootstrap before token " + std::to_string(token) + " done");
      }

      // Encrypt this token's layer input / embedding exactly once.
      auto hidden_ct = time_phase("encrypt_input", [&]() {
        if (args.process_role == "server-eval") {
          return deserialize_ciphertext(
              fs::path(args.handoff_dir) / "exchange" /
                  ("input_t" + std::to_string(token) + ".ct"),
              cc);
        }
        std::vector<double> hidden(batch, 0.0);
        const bool generated_input =
            args.autoregressive_client_loop &&
            token >= chain.autoregressive_prompt_tokens;
        if (generated_input &&
            autoregressive_next_embedding.size() !=
                static_cast<std::size_t>(d_model)) {
          throw std::runtime_error("missing client-selected autoregressive embedding");
        }
        for (int stream = 0; stream < args.streams; ++stream) {
          const auto base = static_cast<std::size_t>(stream * stream_stride);
          for (int slot = 0; slot < d_model; ++slot) {
            hidden[base + static_cast<std::size_t>(slot)] =
                generated_input
                    ? autoregressive_next_embedding[static_cast<std::size_t>(slot)]
                    : input_vectors[static_cast<std::size_t>(token) * d_model + slot];
          }
        }
        return encrypt_values(hidden);
      });

      for (int layer = 0; layer < layers_loaded; ++layer) {
        const std::string key = layer_key(token, layer);
        const std::string tag = key + ".";
        int layer_bootstraps = 0;
        const auto& plan = layer_plans[static_cast<std::size_t>(layer)];
        // The residual checkpoint is outside run_layer, so select this
        // layer's calibrated bounds before refreshing the handoff.
        active_state_bound = plan.state_abs_max;
        active_state_group_bounds = plan.state_group_abs_max;
        active_fifo_bound = plan.fifo_abs_max;
        active_checkpoint_bounds = plan.checkpoint_abs_max;
        maybe_bootstrap(hidden_ct, plan.req_residual, tag + "residual",
                        layer_bootstraps);
        summary_level_in[key] = static_cast<int>(hidden_ct->GetLevel());
        hidden_ct = run_layer(plan,
                              layer_runtimes[static_cast<std::size_t>(layer)], token, layer,
                              hidden_ct, tag, layer_bootstraps);
        summary_level_out[key] = static_cast<int>(hidden_ct->GetLevel());
        summary_bootstraps[key] = layer_bootstraps;
        token_bootstraps_total += layer_bootstraps;
        if (args.debug_layer_errors) {
          const auto& layer_payload = layer_payloads[static_cast<std::size_t>(layer)];
          const auto poly_boundary = layer_payload.tensors.find("test_layer_output_poly");
          const auto& boundary =
              poly_boundary != layer_payload.tensors.end()
                  ? poly_boundary->second
                  : layer_plans[static_cast<std::size_t>(layer)].test_layer_output;
          double boundary_error = 0.0;
          int non_finite = 0;
          try {
            const auto slots =
                decrypt_slots(cc, secret_key, hidden_ct,
                              static_cast<std::size_t>(d_model));
            for (int slot = 0; slot < d_model; ++slot) {
              const double diff = std::abs(
                  slots[static_cast<std::size_t>(slot)] -
                  boundary[static_cast<std::size_t>(token) * d_model + slot]);
              if (!std::isfinite(diff)) {
                ++non_finite;
                continue;
              }
              boundary_error = std::max(boundary_error, diff);
            }
            if (non_finite > 0) {
              boundary_error = 1.0e308;
            }
          } catch (const std::exception& exc) {
            boundary_error = 1.0e308;
            non_finite = d_model;
            log_phase("DEBUG boundary " + key + " decrypt_failed error=" + exc.what());
          }
          debug_boundary_errors[key] = boundary_error;
          log_phase("DEBUG boundary " + key + " error=" +
                    std::to_string(boundary_error) +
                    " non_finite=" + std::to_string(non_finite));
        }
        log_phase(key + " done output_level=" + std::to_string(hidden_ct->GetLevel()) +
                  " bootstraps=" + std::to_string(layer_bootstraps));
      }

      if (full_chain) {
        int final_bootstraps = 0;
        // Keep the large final residual in normalized coordinates. Undoing
        // this scale after BTS would amplify its refresh noise by O(10^3),
        // and RMSNorm cancels the scale analytically anyway.
        hidden_ct = scaled_clone(hidden_ct, 1.0 / final_norm_scale);
        maybe_bootstrap(hidden_ct, final_norm_requirement,
                        "t" + std::to_string(token) + ".final_norm_scaled",
                        final_bootstraps);
        hidden_ct = time_phase("final_norm", [&]() {
          ++ct_ct_muls;
          auto squared = cc->EvalMult(hidden_ct, hidden_ct);
          auto variance = norm_variance_sum(squared);
          variance = add_scalar(
              variance,
              d_model * chain.final_norm_eps /
                  (final_norm_scale * final_norm_scale));
          auto u = add_scalar(scaled_clone(variance, final_a_rms_v), final_b_rms);
          auto guess = eval_chebyshev(u, final_rms_coeffs);
          auto v_neg_half = scaled_clone(variance, -0.5);
          auto inv = newton_refine(guess, v_neg_half, final_rms_iterations,
                                   "t" + std::to_string(token) + ".final_newton",
                                   final_bootstraps);
          auto normed = mul_aligned(hidden_ct, inv);
          return mul_mask(normed, "final.norm_w", final_w_vec);
        });
        token_bootstraps_total += final_bootstraps;
        ckks_levels["t" + std::to_string(token) + ".final_norm"] =
            static_cast<int>(hidden_ct->GetLevel());
      } else {
        // A truncated chain has no next-layer residual checkpoint. Refresh
        // only when the final ciphertext has fewer than twelve levels left,
        // using the independently calibrated layer-output bound.
        constexpr int kDecryptHeadroom = 12;
        int final_output_bootstraps = 0;
        const int output_requirement =
            std::max(0, kDecryptHeadroom - args.auto_bootstrap_headroom);
        maybe_bootstrap(hidden_ct, output_requirement,
                        "t" + std::to_string(token) + ".output",
                        final_output_bootstraps);
        token_bootstraps_total += final_output_bootstraps;
        ckks_levels["t" + std::to_string(token) + ".output"] =
            static_cast<int>(hidden_ct->GetLevel());
      }
      if (args.autoregressive_client_loop &&
          token >= chain.autoregressive_prompt_tokens - 1) {
        const auto client_start = now();
        auto client_hidden = decrypt_slots(
            cc, secret_key, hidden_ct, static_cast<std::size_t>(d_model));
        const auto& lm_head = chain.client_lm_head_w.empty()
                                  ? chain.client_embedding_w
                                  : chain.client_lm_head_w;
        int selected_id = -1;
        double selected_logit = -std::numeric_limits<double>::infinity();
        for (int vocab = 0; vocab < chain.autoregressive_vocab_size; ++vocab) {
          double logit = chain.client_lm_head_b.empty()
                             ? 0.0
                             : chain.client_lm_head_b[static_cast<std::size_t>(vocab)];
          const auto row = static_cast<std::size_t>(vocab) * d_model;
          for (int slot = 0; slot < d_model; ++slot) {
            logit += lm_head[row + static_cast<std::size_t>(slot)] *
                     client_hidden[static_cast<std::size_t>(slot)];
          }
          if (!std::isfinite(logit)) {
            throw std::runtime_error("non-finite autoregressive client logit");
          }
          if (logit > selected_logit) {
            selected_logit = logit;
            selected_id = vocab;
          }
        }
        if (selected_id < 0) {
          throw std::runtime_error("autoregressive client argmax failed");
        }
        autoregressive_selected_ids.push_back(selected_id);
        autoregressive_decrypted_outputs[static_cast<std::size_t>(token)] =
            std::move(client_hidden);
        if (token + 1 < args.tokens) {
          const auto row = static_cast<std::size_t>(selected_id) * d_model;
          autoregressive_next_embedding.assign(
              chain.client_embedding_w.begin() + static_cast<std::ptrdiff_t>(row),
              chain.client_embedding_w.begin() +
                  static_cast<std::ptrdiff_t>(row + d_model));
        }
        autoregressive_client_seconds += seconds_since(client_start);
        log_phase("autoregressive client round trip after token " +
                  std::to_string(token) + " selected_id=" +
                  std::to_string(selected_id));
      }
      token_outputs.push_back(hidden_ct);
      if (args.autoregressive_client_loop &&
          !autoregressive_decrypted_outputs[static_cast<std::size_t>(token)].empty()) {
        // The client already consumed this output; retain only its plaintext
        // measurement instead of pinning one ciphertext per generated token.
        token_outputs.back() = nullptr;
      }
      token_bootstrap_counts.push_back(token_bootstraps_total);
      log_phase("token " + std::to_string(token) + " done output_level=" +
                std::to_string(hidden_ct->GetLevel()) +
                " bootstraps=" + std::to_string(token_bootstraps_total));
    }
    const double eval_seconds = seconds_since(eval_start);
    log_phase("token loop done rotations=" + std::to_string(rotations) +
              " ct_pt=" + std::to_string(ct_pt_muls) + " ct_ct=" + std::to_string(ct_ct_muls) +
              " bootstraps=" + std::to_string(bootstraps));

    if (args.process_role == "server-eval") {
      const auto paths = handoff_paths(args.handoff_dir);
      require_server_has_no_secret_files(paths);
      const int server_secret_key_files = count_server_secret_files(paths);
      for (int token = 0; token < args.tokens; ++token) {
        auto& output = token_outputs[static_cast<std::size_t>(token)];
        serialize_ciphertext(
            paths.exchange /
                ("output_t" + std::to_string(token) + ".ct"),
            cc, public_key, output);
      }
      std::ostringstream result;
      result << "{";
      write_artifact_prefix(result, args);
      result << "\"status\":\"passed\",\"passed\":true,";
      result << "\"parameters\":{\"process_role\":\"server-eval\",";
      result << "\"tokens\":" << args.tokens << ",\"layers\":"
             << layers_loaded << "},";
      result << "\"measurements\":{\"encrypted_outputs_written\":"
             << args.tokens << ",\"server_secret_key_files\":"
             << server_secret_key_files << ",\"eval_seconds\":"
             << eval_seconds << "},";
      result << "\"operation_counts\":{\"rotations\":" << rotations
             << ",\"ct_pt_mul\":" << ct_pt_muls
             << ",\"ct_ct_mul\":" << ct_ct_muls << ",\"adds\":"
             << adds << ",\"bootstraps\":" << bootstraps << "},";
      result << "\"measurement_scope\":{";
      result << "\"client_server_process_separation\":true,";
      result << "\"server_secret_key_loaded\":false,";
      result << "\"encrypted_full_layer_chain_evaluated\":"
             << (full_chain ? "true" : "false") << ",";
      result << "\"full_model_correctness_claimed\":false,";
      result << "\"claim\":\"Secret-key-free server evaluation and "
                "encrypted output serialization; correctness is established "
                "only by the separate client-decrypt phase.\"}";
      result << "}";
      write_payload(args.output_json, result.str());
      return EXIT_SUCCESS;
    }

    // -----------------------------------------------------------------------
    // Decrypt per-token outputs (only after the loop) and compare.
    // -----------------------------------------------------------------------
    const auto decrypt_start = now();
    log_phase("decrypt begin");
    // Cryptographic correctness compares against the identical polynomial
    // circuit. Exact-op vectors remain a separate approximation-quality
    // measurement. Older payloads have only the exact vectors and retain the
    // legacy behavior.
    const std::vector<double>& exact_test_output =
        args.autoregressive_client_loop
            ? chain.autoregressive_expected_exact_final
        : chain_mode ? (full_chain ? chain.expected_final
                                 : layer_payloads.back().tensors.at("test_layer_output"))
                   : layer_payloads.front().tensors.at("test_layer_output");
    const bool has_poly_reference =
        args.autoregressive_client_loop || (chain_mode &&
        (full_chain ? !chain.expected_poly_final.empty()
                    : layer_payloads.back().tensors.count("test_layer_output_poly") > 0));
    const std::vector<double>& test_output =
        args.autoregressive_client_loop
            ? chain.autoregressive_expected_poly_final
        : has_poly_reference
            ? (full_chain
                   ? chain.expected_poly_final
                   : layer_payloads.back().tensors.at("test_layer_output_poly"))
            : exact_test_output;
    std::vector<double> per_token_errors;
    std::vector<double> per_token_exact_errors;
    std::vector<int> per_token_decrypt_ok;
    std::vector<double> token0_decrypted_sample;
    std::vector<double> token0_expected_sample;
    std::vector<double> token0_exact_sample;
    double max_error = 0.0;
    double max_exact_error = 0.0;
    double max_interstream_deviation = 0.0;
    for (int token = 0; token < args.tokens; ++token) {
      // Decrypt through the last stream's stride; every stream's output is
      // compared against the same reference (v1 streams carry identical data
      // through circuit-independent lineages, so agreement IS the check).
      const auto decrypt_length = static_cast<std::size_t>(
          (args.streams - 1) * stream_stride + d_model);
      std::vector<double> slots;
      try {
        const auto& cached =
            autoregressive_decrypted_outputs[static_cast<std::size_t>(token)];
        slots = cached.empty()
                    ? decrypt_slots(
                          cc, secret_key,
                          token_outputs[static_cast<std::size_t>(token)],
                          decrypt_length)
                    : cached;
        per_token_decrypt_ok.push_back(1);
      } catch (const std::exception& exc) {
        per_token_decrypt_ok.push_back(0);
        per_token_errors.push_back(1.0e308);
        per_token_exact_errors.push_back(1.0e308);
        max_error = 1.0e308;
        max_exact_error = 1.0e308;
        log_phase("token " + std::to_string(token) +
                  " decrypt_failed error=" + exc.what());
        continue;
      }
      if (token == 0) {
        for (int slot = 0; slot < std::min(8, d_model); ++slot) {
          token0_decrypted_sample.push_back(slots[static_cast<std::size_t>(slot)]);
          token0_expected_sample.push_back(test_output[static_cast<std::size_t>(slot)]);
          token0_exact_sample.push_back(exact_test_output[static_cast<std::size_t>(slot)]);
        }
      }
      double token_error = 0.0;
      double token_exact_error = 0.0;
      int non_finite_slots = 0;
      for (int stream = 0; stream < args.streams; ++stream) {
        const auto base = static_cast<std::size_t>(stream * stream_stride);
        for (int slot = 0; slot < d_model; ++slot) {
          const double decrypted = slots[base + static_cast<std::size_t>(slot)];
          const double reference =
              test_output[static_cast<std::size_t>(token) * d_model + slot];
          const double exact_reference =
              exact_test_output[static_cast<std::size_t>(token) * d_model + slot];
          const double diff = std::abs(decrypted - reference);
          const double exact_diff = std::abs(decrypted - exact_reference);
          if (!std::isfinite(diff)) {
            // NaN/Inf must fail loudly: std::max(0.0, NaN) silently returns
            // 0.0 and would report a broken decrypt as a perfect one.
            ++non_finite_slots;
            continue;
          }
          token_error = std::max(token_error, diff);
          token_exact_error = std::max(token_exact_error, exact_diff);
          if (stream > 0) {
            const double stream0_value = slots[static_cast<std::size_t>(slot)];
            const double deviation = std::abs(decrypted - stream0_value);
            if (std::isfinite(deviation)) {
              max_interstream_deviation = std::max(max_interstream_deviation, deviation);
            }
          }
        }
      }
      if (non_finite_slots > 0) {
        // JSON-safe sentinel (prints as a finite double, always > tolerance).
        token_error = 1.0e308;
        token_exact_error = 1.0e308;
      }
      per_token_errors.push_back(token_error);
      per_token_exact_errors.push_back(token_exact_error);
      max_error = std::max(max_error, token_error);
      max_exact_error = std::max(max_exact_error, token_exact_error);
      log_phase("token " + std::to_string(token) + " max_abs_error=" +
                std::to_string(token_error) +
                " max_abs_error_vs_exact=" + std::to_string(token_exact_error) +
                " non_finite_slots=" + std::to_string(non_finite_slots) +
                " slot0=" + std::to_string(slots[0]) +
                " ref0=" + std::to_string(test_output[static_cast<std::size_t>(token) * d_model]));
    }
    const double decrypt_seconds = seconds_since(decrypt_start);
    log_phase("decrypt done");
    const bool autoregressive_tokens_match =
        !args.autoregressive_client_loop ||
        autoregressive_selected_ids ==
            chain.autoregressive_expected_generated_ids;
    const bool passed = max_error <= args.tolerance && autoregressive_tokens_match;

    std::ostringstream out;
    out << "{";
    write_artifact_prefix(out, args);
    out << "\"status\":\"" << (passed ? "passed" : "failed") << "\",";
    out << "\"passed\":" << (passed ? "true" : "false") << ",";
    out << "\"parameters\":{";
    out << "\"d_model\":" << d_model << ",";
    out << "\"d_inner\":" << d_inner << ",";
    out << "\"num_heads\":" << heads << ",";
    out << "\"head_dim\":" << head_dim << ",";
    out << "\"state_size\":" << state_size << ",";
    out << "\"n_groups\":" << dims_payload.n_groups << ",";
    out << "\"conv_kernel\":" << conv_kernel << ",";
    out << "\"conv_dim\":" << conv_dim << ",";
    out << "\"proj_dim\":" << proj_dim << ",";
    out << "\"state_group_count\":" << group_count << ",";
    out << "\"batch_size\":" << batch_size << ",";
    out << "\"ring_dimension\":" << args.ring_dim << ",";
    out << "\"multiplicative_depth\":" << args.multiplicative_depth << ",";
    out << "\"scaling_mod_size\":" << args.scaling_mod_size << ",";
    out << "\"first_mod_size\":" << args.first_mod_size << ",";
    out << "\"security\":\"" << json_escape(args.security) << "\",";
    out << "\"secret_key_dist\":\"" << json_escape(args.secret_key_dist) << "\",";
    out << "\"tokens\":" << args.tokens << ",";
    out << "\"tolerance\":" << args.tolerance << ",";
    out << "\"baby_step_in\":" << kBabyStepIn << ",";
    out << "\"baby_step_out\":" << kBabyStepOut << ",";
    out << "\"chain_mode\":" << (chain_mode ? "true" : "false") << ",";
    out << "\"process_role\":\"" << json_escape(args.process_role) << "\",";
    out << "\"autoregressive_client_loop\":"
        << (args.autoregressive_client_loop ? "true" : "false") << ",";
    out << "\"autoregressive_prompt_tokens\":"
        << (args.autoregressive_client_loop
                ? chain.autoregressive_prompt_tokens
                : 0)
        << ",";
    out << "\"autoregressive_generate_tokens\":"
        << (args.autoregressive_client_loop
                ? chain.autoregressive_generate_tokens
                : 0)
        << ",";
    out << "\"n_layers_total\":" << (chain_mode ? chain.n_layers : 1) << ",";
    out << "\"n_layers_loaded\":" << layers_loaded << ",";
    out << "\"max_layers\":" << args.max_layers << ",";
    out << "\"final_norm_applied\":" << (full_chain ? "true" : "false") << ",";
    out << "\"auto_bootstrap_headroom\":" << args.auto_bootstrap_headroom << ",";
    out << "\"residual_bootstrap_headroom\":"
        << args.residual_bootstrap_headroom << ",";
    out << "\"carried_bootstrap_headroom\":" << args.carried_bootstrap_headroom << ",";
    out << "\"streams\":" << args.streams << ",";
    out << "\"stream_stride\":" << stream_stride << ",";
    out << "\"level_align_mode\":\"" << json_escape(args.level_align_mode) << "\",";
    out << "\"projection_late_level\":"
        << (args.projection_late_level ? "true" : "false") << ",";
    out << "\"bsgs_layout\":{";
    out << "\"mode\":\"" << json_escape(args.bsgs_replicas) << "\",";
    out << "\"true_bsgs\":" << (args.replicated_true_bsgs ? "true" : "false") << ",";
    out << "\"interleaved_window\":"
        << (args.interleaved_replicated_projection ? "true" : "false") << ",";
    out << "\"in_replicas\":" << rep_in.replicas << ",";
    out << "\"in_window\":" << rep_in.window << ",";
    out << "\"in_diagonals\":" << (rep_in.replicas > 1 ? rep_in.per_replica : 0) << ",";
    out << "\"in_baby_step\":" << rep_in.baby_step << ",";
    out << "\"in_guard_windows\":" << rep_in.guard_windows << ",";
    out << "\"out_replicas\":" << rep_out.replicas << ",";
    out << "\"out_window\":" << rep_out.window << ",";
    out << "\"out_diagonals\":" << (rep_out.replicas > 1 ? rep_out.per_replica : 0) << ",";
    out << "\"out_baby_step\":" << rep_out.baby_step << ",";
    out << "\"out_guard_windows\":" << rep_out.guard_windows;
    out << "},";
    out << "\"bootstrap_norm_margin\":" << args.bootstrap_norm_margin << ",";
    out << "\"state_bootstrap_margin\":" << args.state_bootstrap_margin << ",";
    out << "\"meta_bts\":" << (args.meta_bts ? "true" : "false") << ",";
    out << "\"meta_bts_alpha\":" << args.meta_bts_alpha << ",";
    out << "\"state_meta_bts_alpha\":" << args.state_meta_bts_alpha << ",";
    out << "\"meta_bts_residual_align_mode\":\""
        << json_escape(args.meta_bts_residual_align_mode) << "\",";
    std::vector<int> gated_init_degrees;
    std::vector<int> gated_newton_iterations;
    gated_init_degrees.reserve(layer_plans.size());
    gated_newton_iterations.reserve(layer_plans.size());
    for (const auto& plan : layer_plans) {
      gated_init_degrees.push_back(
          plan.gated_coeffs.empty() ? 0 : static_cast<int>(plan.gated_coeffs.size()) - 1);
      gated_newton_iterations.push_back(plan.gated_iterations);
    }
    out << "\"gated_init_degrees\":";
    write_int_vector_json(out, gated_init_degrees);
    out << ",\"gated_newton_iterations\":";
    write_int_vector_json(out, gated_newton_iterations);
    out << ",";
    out << "\"final_norm_scale\":" << final_norm_scale << ",";
    out << "\"carried_bounds_source\":\""
        << (all_carried_bounds_calibrated ? "calibration-text" : "generic-fallback")
        << "\",";
    out << "\"carried_state_bounds_granularity\":\""
        << (all_state_head_bounds_calibrated ? "head-group" : "layer") << "\",";
    out << "\"transient_bounds_source\":\""
        << (all_checkpoint_bounds_calibrated ? "calibration-text" : "generic-fallback")
        << "\",";
    out << "\"carried_state_abs_max\":" << dims_payload.state_abs_max << ",";
    out << "\"carried_fifo_abs_max\":" << dims_payload.fifo_abs_max << ",";
    out << "\"bootstrap_before_token\":";
    write_int_set_json(out, args.bootstrap_before_token);
    out << ",\"debug_client_reencrypt_before_token\":";
    write_int_set_json(out, args.debug_client_reencrypt_before_token);
    out << ",\"refresh_recurrent_state_post\":"
        << (args.refresh_recurrent_state_post ? "true" : "false");
    out << ",\"refresh_recurrent_state_post_layers\":";
    write_int_set_json(out, args.refresh_recurrent_state_post_layers);
    out << ",\"state_refresh_interval\":" << args.state_refresh_interval;
    out << ",\"normalized_recurrent_state\":"
        << (args.normalized_recurrent_state ? "true" : "false");
    out << ",\"normalized_state_group_scales\":[";
    for (std::size_t layer = 0; layer < layer_plans.size(); ++layer) {
      if (layer > 0) {
        out << ",";
      }
      write_double_vector_json(out, layer_plans[layer].state_group_scales);
    }
    out << "]";
    out << ",\"replicated_state_blocks\":"
        << (args.replicated_state_blocks ? "true" : "false");
    out << "},";
    out << "\"measurements\":{";
    out.precision(12);  // show real CKKS noise digits in errors and samples
    out << "\"correctness_reference\":\""
        << (has_poly_reference ? "polynomial-circuit" : "exact-legacy") << "\",";
    out << "\"max_abs_error\":" << max_error << ",";
    out << "\"per_token_max_abs_error\":";
    write_double_vector_json(out, per_token_errors);
    out << ",";
    out << "\"max_abs_error_vs_exact\":" << max_exact_error << ",";
    out << "\"per_token_max_abs_error_vs_exact\":";
    write_double_vector_json(out, per_token_exact_errors);
    out << ",";
    out << "\"per_token_decrypt_ok\":";
    write_int_vector_json(out, per_token_decrypt_ok);
    out << ",";
    out << "\"autoregressive_selected_ids\":";
    write_int_vector_json(out, autoregressive_selected_ids);
    out << ",";
    out << "\"autoregressive_expected_ids\":";
    write_int_vector_json(
        out, args.autoregressive_client_loop
                 ? chain.autoregressive_expected_generated_ids
                 : std::vector<int>{});
    out << ",";
    out << "\"autoregressive_tokens_match\":"
        << (autoregressive_tokens_match ? "true" : "false") << ",";
    out << "\"token0_decrypted_sample\":";
    write_double_vector_json(out, token0_decrypted_sample);
    out << ",";
    out << "\"token0_expected_sample\":";
    write_double_vector_json(out, token0_expected_sample);
    out << ",";
    out << "\"token0_exact_sample\":";
    write_double_vector_json(out, token0_exact_sample);
    out << ",";
    out << "\"max_interstream_deviation\":" << max_interstream_deviation << ",";
    out << "\"per_token_bootstrap_count\":";
    write_int_vector_json(out, token_bootstrap_counts);
    out << ",";
    out << "\"required_application_rotation_key_count\":" << rotation_indices.size() << ",";
    out << "\"estimated_token_output_levels\":";
    write_int_vector_json(out, depth_estimate.token_output_levels);
    out << ",";
    out << "\"estimated_required_depth_without_bootstrap\":" << depth_estimate.required_depth
        << ",";
    out << "\"segment_requirements\":{";
    out << "\"residual\":" << depth_estimate.req_residual << ",";
    out << "\"proj\":" << depth_estimate.req_proj << ",";
    out << "\"fifo\":" << depth_estimate.req_fifo << ",";
    out << "\"conv\":" << depth_estimate.req_conv << ",";
    out << "\"dt_decay\":" << depth_estimate.req_dt << ",";
    out << "\"decay\":" << depth_estimate.req_decay << ",";
    out << "\"state_pre\":" << depth_estimate.req_state_pre << ",";
    out << "\"state_tail\":" << depth_estimate.req_state_tail << ",";
    out << "\"y_gated\":" << depth_estimate.req_y << ",";
    out << "\"y_out\":" << depth_estimate.req_out << ",";
    out << "\"newton_iteration\":2,";
    out << "\"max_segment\":" << depth_estimate.max_segment << ",";
    out << "\"assumed_bootstrap_output_level\":" << kAssumedBootstrapOutputLevel;
    out << "},";
    out << "\"rotation_keys\":{";
    out << "\"mode\":\"" << json_escape(args.rotation_keys) << "\",";
    out << "\"budget_gib\":" << args.rotation_key_gib << ",";
    out << "\"keys_total\":" << rotation_keygen_indices.size() << ",";
    out << "\"keys_gib_est\":"
        << (rotation_keygen_indices.size() * rotation_per_key_gib) << ",";
    out << "\"per_key_gib_est\":" << rotation_per_key_gib << ",";
    out << "\"required_indices_count\":" << rotation_indices.size() << ",";
    out << "\"base\":" << rotation_base_keys.size() << ",";
    out << "\"direct\":"
        << (rotation_keygen_indices.size() >= rotation_base_keys.size() &&
                    args.rotation_keys != "full"
                ? rotation_keygen_indices.size() - rotation_base_keys.size()
                : rotation_keygen_indices.size())
        << ",";
    out << "\"composite_apps_per_token\":" << planned_composite_apps;
    out << "},";
    out << "\"required_rotation_indices\":";
    {
      std::vector<int> required_plain(rotation_indices.begin(), rotation_indices.end());
      write_int_vector_json(out, required_plain);
    }
    out << ",";
    out << "\"plaintext_coefficient_floor\":" << kPlaintextCoefficientFloor << ",";
    out << "\"pt_cache\":{";
    out << "\"mode\":\"" << json_escape(pt_cache_mode) << "\",";
    out << "\"budget_gib\":" << args.pt_cache_gib << ",";
    out << "\"encode_level\":" << args.pt_cache_level << ",";
    out << "\"weight_encode_level\":" << args.pt_cache_weight_level << ",";
    out << "\"miss_consumption_level\":"
        << (args.pt_miss_consumption_level ? "true" : "false") << ",";
    out << "\"miss_consumption_level_encodes\":"
        << pt_miss_consumption_level_encodes << ",";
    out << "\"miss_consumption_level_min\":"
        << (pt_miss_consumption_level_encodes > 0
                ? pt_miss_consumption_level_min
                : 0)
        << ",";
    out << "\"miss_consumption_level_max\":"
        << (pt_miss_consumption_level_encodes > 0
                ? pt_miss_consumption_level_max
                : 0)
        << ",";
    out << "\"miss_consumption_level_mean\":"
        << (pt_miss_consumption_level_encodes > 0
                ? static_cast<double>(pt_miss_consumption_level_sum) /
                      static_cast<double>(pt_miss_consumption_level_encodes)
                : 0.0)
        << ",";
    out << "\"consumption_count\":" << pt_consumption_count << ",";
    out << "\"consumption_level_min\":"
        << (pt_consumption_count > 0 ? pt_consumption_level_min : 0) << ",";
    out << "\"consumption_level_max\":"
        << (pt_consumption_count > 0 ? pt_consumption_level_max : 0) << ",";
    out << "\"consumption_level_mean\":"
        << (pt_consumption_count > 0
                ? static_cast<double>(pt_consumption_level_sum) /
                      static_cast<double>(pt_consumption_count)
                : 0.0)
        << ",";
    out << "\"cache_hit_consumption_level_min\":"
        << (pt_cache_hits > 0 ? pt_cache_hit_consumption_level_min : 0) << ",";
    out << "\"cache_level_bypasses\":" << pt_cache_level_bypasses << ",";
    out << "\"bytes_per_plaintext\":" << pt_plain_bytes << ",";
    out << "\"weight_bytes_per_plaintext\":" << pt_weight_plain_bytes << ",";
    out << "\"entries_registered\":" << plain_cache.size() << ",";
    out << "\"entries_cached\":" << pt_cache_entries_cached << ",";
    out << "\"bytes_cached_gib\":"
        << (pt_cache_bytes_cached / (1024.0 * 1024.0 * 1024.0)) << ",";
    out << "\"hits\":" << pt_cache_hits << ",";
    out << "\"misses\":" << pt_cache_misses << ",";
    out << "\"encode_seconds\":" << pt_cache_encode_seconds << ",";
    out << "\"encode_threads_requested\":" << args.encode_threads << ",";
    out << "\"encode_threads_effective\":" << effective_encode_threads << ",";
    out << "\"encode_selftest\":\"" << json_escape(encode_selftest_result) << "\",";
    out << "\"pt_reuse_check\":\"" << json_escape(pt_reuse_check_result) << "\"";
    out << "},";
    out << "\"executed_bootstrap_count\":" << bootstraps << ",";
    out << "\"state_bootstrap_count\":" << state_bootstraps << ",";
    out << "\"debug_client_reencrypt_ciphertext_count\":"
        << debug_client_reencrypt_ciphertexts << ",";
    out << "\"meta_bts\":{";
    out << "\"enabled\":" << (args.meta_bts ? "true" : "false") << ",";
    out << "\"alpha\":" << args.meta_bts_alpha << ",";
    out << "\"state_alpha\":"
        << (args.state_meta_bts_alpha >= 0 ? args.state_meta_bts_alpha
                                          : args.meta_bts_alpha)
        << ",";
    out << "\"residual_align_mode\":\""
        << json_escape(args.meta_bts_residual_align_mode) << "\",";
    out << "\"applied_count\":" << meta_bts_applied;
    out << "},";
    out << "\"bootstrap_events\":[";
    for (std::size_t index = 0; index < bootstrap_events.size(); ++index) {
      if (index != 0) {
        out << ",";
      }
      const auto& event = bootstrap_events[index];
      out << "{";
      out << "\"checkpoint\":\"" << json_escape(event.checkpoint) << "\",";
      out << "\"level_before\":" << event.level_before << ",";
      out << "\"level_after\":" << event.level_after << ",";
      out << "\"requirement\":" << event.requirement << ",";
      out << "\"policy_headroom\":" << event.policy_headroom << ",";
      out << "\"physical_bootstraps\":" << event.physical_bootstraps << ",";
      out << "\"carried\":" << (event.carried ? "true" : "false") << ",";
      out << "\"meta_bts\":" << (event.meta_bts ? "true" : "false") << ",";
      out << "\"bound\":" << event.bound << ",";
      out << "\"seconds\":" << event.seconds;
      out << "}";
    }
    out << "],";
    out << "\"peak_rss_gib\":" << peak_rss_gib() << ",";
    out << "\"rss_gib\":" << rss_gib();
    out << "},";
    out << "\"layer_token_summary\":{";
    {
      bool first_summary = true;
      for (const auto& [key, level_in] : summary_level_in) {
        if (!first_summary) {
          out << ",";
        }
        first_summary = false;
        out << "\"" << json_escape(key) << "\":{";
        out << "\"level_in\":" << level_in << ",";
        out << "\"level_out\":" << summary_level_out[key] << ",";
        out << "\"bootstraps\":" << summary_bootstraps[key];
        if (args.debug_layer_errors && debug_boundary_errors.count(key) > 0) {
          out << ",\"debug_boundary_error\":" << debug_boundary_errors[key];
        }
        out << "}";
      }
    }
    out << "},";
    out << "\"ckks_levels\":";
    write_int_map_json(out, ckks_levels);
    out << ",";
    out << "\"timing\":{";
    out << "\"setup_seconds\":" << setup_seconds << ",";
    out << "\"pt_cache_encode_seconds\":" << pt_cache_encode_seconds << ",";
    out << "\"rotate_keygen_seconds\":" << rotate_keygen_seconds << ",";
    out << "\"bootstrap_precompute_seconds\":" << bootstrap_precompute_seconds << ",";
    out << "\"bootstrap_eval_seconds\":" << bootstrap_eval_seconds << ",";
    out << "\"debug_client_reencrypt_seconds\":"
        << debug_client_reencrypt_seconds << ",";
    out << "\"autoregressive_client_seconds\":"
        << autoregressive_client_seconds << ",";
    out << "\"load_context_seconds\":" << load_context_seconds << ",";
    out << "\"eval_seconds\":" << eval_seconds << ",";
    out << "\"decrypt_seconds\":" << decrypt_seconds;
    out << "},";
    out << "\"phase_timings\":";
    write_double_map_json(out, phase_timings);
    out << ",";
    out << "\"phase_operation_counts\":";
    write_phase_operation_counts_json(out, phase_operation_counts);
    out << ",";
    out << "\"operation_counts\":{";
    out << "\"rotations\":" << rotations << ",";
    out << "\"rotations_direct\":" << rotations_direct << ",";
    out << "\"rotations_composite_steps\":" << rotations_composite_steps << ",";
    out << "\"ct_pt_mul\":" << ct_pt_muls << ",";
    out << "\"ct_ct_mul\":" << ct_ct_muls << ",";
    out << "\"adds\":" << adds << ",";
    out << "\"unity_level_align_muls\":" << unity_multiplies << ",";
    out << "\"direct_level_align_drops\":" << direct_level_drops << ",";
    out << "\"projection_late_level_drops\":"
        << projection_late_level_drops << ",";
    out << "\"bootstraps\":" << bootstraps;
    out << "},";
    out << "\"measurement_scope\":{";
    out << "\"mamba2_full_width_layer_circuit\":true,";
    out << "\"multi_layer_ciphertext_residual_handoff\":" << (chain_mode ? "true" : "false")
        << ",";
    out << "\"layers_loaded\":" << layers_loaded << ",";
    out << "\"full_layer_chain\":" << (full_chain ? "true" : "false") << ",";
    out << "\"final_norm_applied\":" << (full_chain ? "true" : "false") << ",";
    out << "\"multi_token_ciphertext_state_carry\":true,";
    out << "\"ciphertext_conv_fifo\":true,";
    out << "\"zero_intermediate_decrypts\":"
        << ((args.debug_decrypt || args.debug_refresh_probes ||
             args.debug_layer_errors ||
             !args.debug_client_reencrypt_before_token.empty() ||
             args.autoregressive_client_loop)
                ? "false"
                : "true")
        << ",";
    out << "\"autoregressive_client_loop_simulation\":"
        << (args.autoregressive_client_loop ? "true" : "false") << ",";
    out << "\"client_server_process_separated\":false,";
    out << "\"debug_client_reencrypt_simulation\":"
        << (args.debug_client_reencrypt_before_token.empty() ? "false" : "true") << ",";
    out << "\"recurrent_state_post_refresh\":"
        << ((args.refresh_recurrent_state_post ||
             !args.refresh_recurrent_state_post_layers.empty())
                ? "true"
                : "false")
        << ",";
    out << "\"recurrent_state_post_refresh_layers\":";
    write_int_set_json(out, args.refresh_recurrent_state_post_layers);
    out << ",";
    out << "\"recurrent_state_refresh_interval\":"
        << args.state_refresh_interval << ",";
    out << "\"persistent_normalized_recurrent_state\":"
        << (args.normalized_recurrent_state ? "true" : "false") << ",";
    out << "\"per_token_fresh_embedding_encryption\":true,";
    out << "\"multi_stream_slot_packing\":" << (args.streams > 1 ? "true" : "false") << ",";
    out << "\"streams\":" << args.streams << ",";
    out << "\"streams_carry_identical_data\":" << (args.streams > 1 ? "true" : "false")
        << ",";
    out << "\"tokens\":" << args.tokens << ",";
    out << "\"auto_bootstrap_policy\":" << (bootstrap_available ? "true" : "false") << ",";
    out << "\"scheduled_bootstrap\":"
        << (args.bootstrap_before_token.empty() ? "false" : "true") << ",";
    out << "\"bootstrap_before_token\":";
    write_int_set_json(out, args.bootstrap_before_token);
    out << ",";
    out << "\"debug_client_reencrypt_before_token\":";
    write_int_set_json(out, args.debug_client_reencrypt_before_token);
    out << ",";
    out << "\"executed_bootstrap_count\":" << bootstraps << ",";
    out << "\"ckks_level_telemetry\":true,";
    out << "\"fideslib_encrypted_execution\":true,";
    out << "\"full_model_correctness_claimed\":false,";
    out << "\"claim\":\"Native FIDESlib encrypted full-width Mamba-2 decode over one or "
           "more checkpoint layers with ciphertext residual handoff, per-layer ciphertext "
           "state and conv FIFO carry, and mid-circuit auto-bootstrap refresh. In "
        << (args.autoregressive_client_loop
                ? "autoregressive mode, one-process client simulation decrypts final_norm, "
                  "runs lm_head/greedy argmax, and freshly encrypts the selected next-token "
                  "embedding; it does not claim client/server process separation."
                : "fixed-vector mode, embeddings are encrypted fresh from payload test "
                  "vectors; it does not claim lm_head decoding.")
        << "\"";
    out << "},";
    out << "\"notes\":[";
    if (args.streams > 1) {
      out << "\"multi-stream v1: all streams carry IDENTICAL data through "
             "circuit-independent ciphertext lineages (replicated state, shared "
             "stride-periodic plaintexts, shared rotations); per-stream agreement with "
             "the reference is the correctness check, not independent throughput "
             "content.\",";
    }
    out << "\"dgx runbook: FIDESlib v2.1.0 MAXP=64 caps towers (max depth 50 at scale 40, "
           "44 at scale 59); EvalBootstrap needs scaling-mod >= 54 (dead at 40/50/52) and "
           "outputs GetLevel=18 at depth 44/scale 59, warm cost 12-14 ms; run with "
           "--scaling-mod-size 59 --multiplicative-depth 44, mid-circuit checkpoints keep "
           "every lineage within the 26-level segment budget; ring 65536 fits GB10 "
           "memory, ring 131072 keygen+LoadContext needs > 119 GB.\"";
    out << "]";
    out << "}";
    write_payload(args.output_json, out.str());
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exc) {
    if (args_available && !payload_file_exists(args.output_json)) {
      try {
        write_runtime_failure_payload(args, "runtime", exc.what());
      } catch (const std::exception& write_exc) {
        std::cerr << "failed to write runtime failure artifact: " << write_exc.what()
                  << std::endl;
      }
    }
    std::cerr << "stage1_mamba2_decode_fideslib failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
}
