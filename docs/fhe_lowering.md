# FHE Lowering Notes

This prototype separates three research tracks.

## Symbolic CKKS First

Version `0.2.0` added a symbolic CKKS layer tracker before OpenFHE lowering. It
records ciphertext-ciphertext products, ciphertext-plaintext products, rotations,
remaining levels, greedy bootstrap positions, and head/MIMO packing. This keeps
the research memo's assumptions executable. Current `0.3.x` code has targeted
OpenFHE execution harnesses, while full end-to-end encrypted model lowering
remains out of scope.

## Static B/C

Static B/C is the initial FHE target. The encrypted token embedding goes through
plaintext linear maps, then the recurrent state update is:

```text
h_t = alpha * h_{t-1} + B * u_t
y_t = C * h_t
```

Here `alpha`, `B`, and `C` are plaintext model parameters. The path is friendly
to CKKS-style inference because the state update can be scheduled as
ciphertext-plaintext multiplies and additions. The optional polynomial gate is
the main ciphertext-ciphertext multiplication in this mode.

## Dynamic B/C

Dynamic B/C keeps the Mamba-3-like token-dependent projections:

```text
B_t = W_B x_t
C_t = W_C x_t
h_t = alpha * h_{t-1} + B_t * u_t
y_t = C_t * h_t
```

This is more expressive, but it adds ciphertext-ciphertext products inside the
recurrence and readout. It should be treated as an accuracy baseline and then
distilled or constrained toward static/low-rank plaintext B/C for FHE inference.

## Backend Lowering Status

Targeted OpenFHE CKKS harnesses now map:

- rank channels to SIMD slots,
- state dimension to either packed slots or separate ciphertexts,
- static B/C to plaintext diagonals,
- the polynomial gate to one rescale level.

The current `estimate_block_cost` function is a backend-agnostic first pass, not
a replacement for an integrated full-model OpenFHE/FIDESlib/HEaaN schedule.
