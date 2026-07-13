#pragma once

#include <set>
#include <string>
#include <string_view>

namespace fhemamba::stage1 {

struct Config {
  std::string input;
  std::string input_chain;
  std::string output_json;
  std::string artifact_version = "0.0.0+unknown";
  std::string repo_commit = "unknown";
  std::string binary_sha256 = "unknown";
  // Process-separated fixed-vector protocol. inline is the legacy one-process
  // benchmark. client-init generates/serializes keys and encrypted inputs;
  // server-eval loads only public/evaluation material and writes encrypted
  // outputs; client-decrypt loads only the client secret and verifies them.
  std::string process_role = "inline";
  std::string handoff_dir;
  // FIDESlib v2.1.0 MAXP=64 tower cap: empirical max depth 50 at scale 40 and
  // 44 at scale 59; EvalBootstrap is numerically dead below scale 54 (dgx
  // findings). Defaults follow the working dgx geometry. Ring 65536 is also
  // supported (batch halves; packing regroups to 6 state ciphertexts).
  int ring_dim = 131072;
  int multiplicative_depth = 44;
  int scaling_mod_size = 59;
  int first_mod_size = 60;
  int tokens = 4;
  int max_layers = 0;  // chain mode: 0 = all layers
  int auto_bootstrap_headroom = 12;
  // A layer output at level 26 has exactly the 18 levels required by the
  // next block-norm-to-projection segment. Extra headroom here refreshes a
  // potentially large residual at every layer and scales the refresh noise
  // back up by that residual's calibration bound.
  int residual_bootstrap_headroom = 0;
  // Carried state/FIFO already have explicit requirements that cover their
  // current-token update/readout lineage. Refreshing them merely to reserve
  // the transient headroom injects avoidable noise on every token, so defer
  // refresh until the carried lineage actually lacks the required levels.
  int carried_bootstrap_headroom = 0;
  // Token-invariant plaintext cache (dgx: per-token MakeCKKSPackedPlaintext
  // re-encoding of BSGS diagonals is 89% of M1 wall time). "full" also caches
  // the BSGS diagonal tables, "masks" only the small select/broadcast/tap/
  // weight vectors, "off" disables; "auto" resolves to full (single layer) /
  // masks (chain). The byte budget bounds the cache; entries are chosen
  // greedily by per-token reuse count and everything else falls back to
  // on-the-fly encoding.
  std::string pt_cache = "auto";
  // DGX Spark 24-layer sweep: 5/10 GiB tie on latency while larger caches
  // increase unified-memory pressure and slow bootstrap/nonlinear phases.
  double pt_cache_gib = 5.0;
  // Expert knob: encode cached plaintexts at this consumption level using the
  // 5-arg MakeCKKSPackedPlaintext overload (already used by the forensic
  // path), shrinking entries by (depth+1-level)/(depth+1). Must be <= the
  // minimum ciphertext level at every consumption site of every cached entry
  // (the kernel does not derive this); 0 = full-level encode, always safe.
  int pt_cache_level = 0;
  // Optional lower-tower encoding level for projection-weight plaintexts.
  // A cache hit is accepted only when this level is no deeper than the
  // consuming ciphertext; otherwise evaluation falls back to an exact-level
  // miss encode, so an aggressive value cannot silently spend circuit depth.
  int pt_cache_weight_level = 0;
  // Encode uncached plaintexts at the consuming ciphertext's exact level.
  // This is opt-in until native parity is measured; unlike a fixed cache
  // level, the per-use level cannot underflow a consumption site.
  bool pt_miss_consumption_level = false;
  // Worker threads for the BSGS cache-miss encode path. Per-diagonal
  // MakeCKKSPackedPlaintext calls are independent host NTT work, but
  // FIDESlib/OpenFHE encode thread-safety on a shared (read-only)
  // CryptoContext is EXPECTED yet UNPROVEN: a startup self test encodes the
  // same diagonals serially and in parallel and falls back to serial (with a
  // logged warning) on any mismatch or exception. Default 1 keeps today's
  // strictly serial behavior. Low-rank projection is refuted (SVD rank-256
  // explodes PPL), so the 768-diagonal matmuls stay and the full 24-layer
  // cache cannot fit dgx memory; this is the memory-free lever for the miss
  // path (Grace has 20 cores).
  int encode_threads = 1;
  // Multi-stream slot packing (M4 throughput lever, single-layer mode only):
  // S independent decode streams at stride batch/S in the packed layouts
  // (hidden/proj/y all fit stride 4096 at ring 65536 with S=8). BSGS
  // diagonals and packed masks are stride-periodic so the SAME plaintexts
  // serve all streams; state ciphertexts replicate per stream (the conv FIFO
  // is packed and needs no replication); stream-base shifts decompose into
  // existing power-of-two rotations, so the rotation-key set is unchanged.
  // v1 replicates the single payload prompt across streams: per-stream
  // outputs must agree (identical data, circuit-independent lineages).
  int streams = 1;
  // Rotation-key plan (128-bit needs ring 2^17 where ~194 direct keys are
  // ~68 GiB and OOMed dgx; design in fhemamba/src/fhemamba/rotation_keys.py):
  //   full     every required index gets a direct key (current behavior);
  //   compact  only the signed power-of-two base keys that cover the NAF
  //            decompositions actually used;
  //   balanced base keys + the hottest non-base indices greedily by
  //            frequency x (NAF weight - 1) under --rotation-key-gib.
  // Composition happens inside the rotate() choke point; the math is
  // identical in all modes (rotations compose exactly and consume no levels).
  std::string rotation_keys = "full";
  double rotation_key_gib = 45.0;
  // Level-only alignment can preserve the proven EvalMult(x, 1) lowering,
  // ask SetLevel to adjust/drop in one call, or defer to the scale/level
  // adjustment already inside FIDESlib EvalAdd/EvalMult. Both alternatives
  // stay opt-in until real-CKKS parity passes.
  std::string level_align_mode = "unity";
  // Input-replicated BSGS (spec: fhemamba/src/fhemamba/bsgs_layout.py; the
  // slot-exact simulator is the authority and the C++ schedule is verified
  // bitwise against it). "1" = legacy dense-diagonal path (bit-identical);
  // "auto" = choose_window per matmul (in_proj r=7/window 4608, out_proj
  // r=10/window 3072 at batch 32768); an integer forces r (capped by fit).
  // Cuts per-token ct-pt for the two matmuls 2304 -> 264 and the full cache
  // to ~16 GiB. Requires --streams 1 (replicas and streams both partition
  // the batch: S * r * window <= batch; co-optimization is future work).
  std::string bsgs_replicas = "1";
  // Apply a real baby-step/giant-step decomposition to the replicated
  // diagonal groups. Off preserves the measured replicated schedule.
  bool replicated_true_bsgs = false;
  // Tighten each replicated projection window from m+n to m+r-1. The
  // existing masks interleave replica j at output offset i+j, so only r-1
  // guard slots are required. One extra filled input window prevents the last
  // active replica from reading an uninitialized tail during rotation. The
  // 130M out-projection grows from 10 to 20 active replicas.
  bool interleaved_replicated_projection = false;
  // Expand the contiguous Mamba B/C vectors into recurrent-state blocks by
  // logarithmic rotate-add replication instead of one mask per state slot.
  bool replicated_state_blocks = false;
  // In direct-drop alignment mode, discard projection-input towers that the
  // following inverse-norm multiply would discard from the linear result
  // anyway. This keeps residual/state branches unchanged while making the
  // rotation-heavy VMM execute at a cheaper late level.
  bool projection_late_level = false;
  double tolerance = 5e-2;
  std::set<int> bootstrap_before_token;
  // Diagnostic protocol simulation: decrypt and freshly encrypt every
  // carried state/FIFO ciphertext at selected token boundaries. This models
  // a client round trip but runs in one process; it is never an FHE-only or
  // process-separation claim.
  std::set<int> debug_client_reencrypt_before_token;
  // End-to-end decode protocol simulation for a chain payload carrying
  // autoregressive assets. The encrypted recurrent state remains server-side;
  // after each completed decode step the client decrypts final_norm, applies
  // lm_head + greedy argmax, looks up the selected embedding, and encrypts the
  // next input. This first implementation is one process and therefore does
  // not claim process or secret-key separation.
  bool autoregressive_client_loop = false;
  // Refresh recurrent states after decay*state + update, immediately before
  // readout. Refreshing only the carried input cannot remove the extra level
  // consumed by the recurrent ct-ct multiply.
  bool refresh_recurrent_state_post = false;
  std::set<int> refresh_recurrent_state_post_layers;
  // Periodic post-update state refresh for long autoregressive runs. Token 0
  // initializes the state; interval N refreshes after recurrent updates at
  // token indices N, 2N, ... . 0 leaves refresh entirely level-driven.
  int state_refresh_interval = 0;
  // Store each recurrent head group permanently as u = state / S, where S is
  // its public calibration maximum. The update 1/S and readout S factors are
  // folded into existing plaintext masks, so this changes neither the Mamba
  // formula nor multiplicative depth. It is opt-in until encrypted parity.
  bool normalized_recurrent_state = false;
  // Retain two-pass Meta-BTS for normalized carried state when deep chains
  // need a lower refresh noise floor than the faster single-BTS path.
  bool normalized_state_meta_bts = false;
  int bootstrap_level_budget_cts = 5;
  int bootstrap_level_budget_stc = 5;
  int bootstrap_bsgs_dim_cts = 0;
  int bootstrap_bsgs_dim_stc = 0;
  std::string security = "not-set";
  std::string secret_key_dist = "sparse-ternary";
  // Debug-only: decrypt at phase/bootstrap checkpoints and log value stats
  // (max |value|, non-finite slot count). Voids the zero-intermediate-decrypt
  // privacy claim; for NaN localization, never for reported runs.
  bool debug_decrypt = false;
  // Debug-only: decrypt only normalized recurrent-state inputs immediately
  // before bootstrap and record their maximum magnitude.
  bool debug_normalized_state_bootstrap_range = false;
  // Repeat-bootstrap probes are much more invasive than value telemetry and
  // can perturb the GPU state being diagnosed. Keep them opt-in.
  bool debug_refresh_probes = false;
  // Debug-only, cheap: decrypt just the residual at each layer boundary and
  // compare against that layer's exact-op boundary vectors (error-vs-depth
  // curve). 2 decrypts per layer per token; no refresh probes. Also voids the
  // zero-intermediate-decrypt claim.
  bool debug_layer_errors = false;
  // Global multiplier on the per-checkpoint magnitude bounds used for the
  // pre-bootstrap normalization (see checkpoint_bound). Applies to TRANSIENT
  // per-layer activation checkpoints.
  double bootstrap_norm_margin = 1.1;
  // Separate, tighter multiplier for CARRIED-FORWARD ciphertexts (SSM state,
  // conv FIFO). Refresh at deg 1 adds a roughly fixed noise floor that the
  // undo-multiply by the bound B then scales up, so the noise injected into a
  // cross-token lineage per refresh is ~proportional to B; a loose 1.5x of a
  // generic bound over-scales (esp. low-|m| layers) and amplifies the
  // token-over-token blowup. Tightening B to ~1.1x the measured |m| is the
  // dgx-only lever for the replicated-128bit token1 fail and the 24-layer
  // token-horizon (DESIGN.md dgx-only roadmap). B must stay >= true max |m|
  // (no clipping); 1.1 leaves 10% headroom over the measured bound.
  double state_bootstrap_margin = 1.1;
  // Meta-BTS (double bootstrap with residual amplification) on carried
  // lineages, gated polynomial inputs, and explicitly selected residual
  // layers: y1 = BTS(x_n) carries error e1 ~ eps; the residual
  // r = x_n - y1 = -e1 is amplified by 2^alpha (needs ONE live level, so the
  // carried refresh trigger fires one level earlier), bootstrapped again
  // (y2 = -e1*2^alpha + e2), scaled back and added: y1 + y2*2^-alpha =
  // x_n + e2*2^-alpha — refresh error drops eps -> ~eps*2^-alpha. Applied to
  // carried values and the high-sensitivity gated polynomial input. Cost per
  // refresh: +1 EvalBootstrap (~13 ms warm), +2 pt-mults, +1 add.
  bool meta_bts = false;
  // The local residual probe keeps 2^12 inside the bootstrap message range;
  // 2^16 exceeded it on every measured step.
  int meta_bts_alpha = 12;
  // Optional carried-state override. Negative inherits meta_bts_alpha so
  // existing runs keep identical behavior.
  int state_meta_bts_alpha = -1;
  // The Meta-BTS residual x - BTS(x) is especially sensitive to alignment
  // noise because its message is itself only bootstrap error. Allow this one
  // alignment to use SetLevel without changing the rest of the deep circuit.
  std::string meta_bts_residual_align_mode = "unity";
  // Apply Meta-BTS to the transient residual refresh only at selected layers.
  // The residual calibration bound grows sharply near the end of the 130M
  // model, so a single bootstrap's error floor is multiplied by O(10^3).
  // Keeping this selective avoids doubling every layer's residual refresh.
  std::set<int> meta_bts_residual_layers;
};

auto parse_args(int argc, char* argv[]) -> Config;

auto should_use_meta_bts(const Config& config, int active_layer,
                         bool carried, bool normalized_state,
                         std::string_view checkpoint) -> bool;

}  // namespace fhemamba::stage1
