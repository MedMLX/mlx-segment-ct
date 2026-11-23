"""Ordered FP32 transposed convolution for the VISTA3D decoders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from medmlx_core.typing import MlxRuntime

if TYPE_CHECKING:
    from mlx.core import array


_SOURCE = """
using Index = metal::conditional_t<LARGE_INDEX, ulong, uint>;
constexpr Index SPATIAL = Index(DEPTH) * HEIGHT * WIDTH;
constexpr Index OUTPUT_SPATIAL = 8 * SPATIAL;
constexpr uint BLOCK_CHANNELS = COLUMN_FRAGMENTS * 8;
constexpr bool FULL_COLUMNS = OUT_CHANNELS % BLOCK_CHANNELS == 0;
// Match Float32Operators.conv: one SIMD matrix operation per eight input
// channels, in ascending order, with the contracted size padded to 32.
constexpr uint CHUNKS = ((IN_CHANNELS + 31) / 32) * 4;
uint lane = thread_index_in_simdgroup;
uint fragment_row = ((lane / 4) & 4) + ((lane / 2) % 4);
uint fragment_column = ((lane / 4) & 2) * 2 + (lane % 2) * 2;
Index row_start = (Index(threadgroup_position_in_grid.y) * SIMDGROUPS
                  + simdgroup_index_in_threadgroup) * ROW_FRAGMENTS * 8;

uint phase = threadgroup_position_in_grid.z % 8;
uint column = threadgroup_position_in_grid.z / 8 * BLOCK_CHANNELS + fragment_column;
uint3 odd = uint3((phase >> 2) & 1, (phase >> 1) & 1, phase & 1);
uint3 input_position[ROW_FRAGMENTS];
Index input_batch[ROW_FRAGMENTS], output_offset[ROW_FRAGMENTS];
bool row_ok[ROW_FRAGMENTS];
#pragma unroll
for (uint row_fragment = 0; row_fragment < ROW_FRAGMENTS; ++row_fragment) {
    Index row = row_start + row_fragment * 8 + fragment_row;
    row_ok[row_fragment] = row < BATCH * SPATIAL;
    Index batch = row_ok[row_fragment] ? row / SPATIAL : 0;
    uint3 position = uint3(
        row / (Index(HEIGHT) * WIDTH) % DEPTH, row / WIDTH % HEIGHT, row % WIDTH);
    input_position[row_fragment] = position;
    input_batch[row_fragment] = batch;
    uint3 destination = 2 * position + odd;
    Index output_row = Index(destination.x) * (2 * HEIGHT) + destination.y;
    output_offset[row_fragment] = batch * OUT_CHANNELS * OUTPUT_SPATIAL
                                + output_row * (2 * WIDTH) + destination.z;
}

float2 output[ROW_FRAGMENTS][COLUMN_FRAGMENTS];
// Even positions use the center tap. Odd positions use tap 0 (next input)
// and tap 2 (same input). Enumerate the contributing taps in z/y/x order.
for (uint tap_mask = 0; tap_mask <= phase; ++tap_mask) {
    if (tap_mask & ~phase) continue;
    uint3 second = uint3((tap_mask >> 2) & 1, (tap_mask >> 1) & 1, tap_mask & 1);
    uint3 tap = 1 - odd + 2 * second;
    Index tap_offset = Index((tap.x * 3 + tap.y) * 3 + tap.z) * IN_CHANNELS * OUT_CHANNELS;
    Index input_offset[ROW_FRAGMENTS];
    bool in_bounds[ROW_FRAGMENTS];
    #pragma unroll
    for (uint i = 0; i < ROW_FRAGMENTS; ++i) {
        uint3 shifted = input_position[i] + odd - second;
        in_bounds[i] = all(shifted < uint3(DEPTH, HEIGHT, WIDTH));
        input_offset[i] = input_batch[i] * IN_CHANNELS * SPATIAL
                        + (Index(shifted.x) * HEIGHT + shifted.y) * WIDTH + shifted.z;
    }

    float2 accum[ROW_FRAGMENTS][COLUMN_FRAGMENTS] = {};
    for (uint chunk = 0; chunk < CHUNKS; ++chunk) {
        uint left = 8 * chunk + fragment_column;
        uint right = left + 1;
        uint weight_row = 8 * chunk + fragment_row;
        float2 a[ROW_FRAGMENTS], w[COLUMN_FRAGMENTS];
        #pragma unroll
        for (uint i = 0; i < ROW_FRAGMENTS; ++i) {
            a[i] = float2(
                in_bounds[i] && left < IN_CHANNELS ? X[input_offset[i] + left * SPATIAL] : 0.0f,
                in_bounds[i] && right < IN_CHANNELS ? X[input_offset[i] + right * SPATIAL] : 0.0f);
        }
        #pragma unroll
        for (uint j = 0; j < COLUMN_FRAGMENTS; ++j) {
            uint oc = column + 8 * j;
            Index index = tap_offset + Index(weight_row) * OUT_CHANNELS + oc;
            w[j] = float2(
                weight_row < IN_CHANNELS && (FULL_COLUMNS || oc < OUT_CHANNELS)
                    ? W[index] : 0.0f,
                weight_row < IN_CHANNELS && (FULL_COLUMNS || oc + 1 < OUT_CHANNELS)
                    ? W[index + 1] : 0.0f);
        }
        #pragma unroll
        for (uint i = 0; i < ROW_FRAGMENTS; ++i) {
            #pragma unroll
            for (uint j = 0; j < COLUMN_FRAGMENTS; ++j) {
                metal::simdgroup_float8x8 am, wm, cm;
                am.thread_elements()[0] = a[i].x;
                am.thread_elements()[1] = a[i].y;
                wm.thread_elements()[0] = w[j].x;
                wm.thread_elements()[1] = w[j].y;
                cm.thread_elements()[0] = accum[i][j].x;
                cm.thread_elements()[1] = accum[i][j].y;
                simdgroup_multiply_accumulate(cm, am, wm, cm);
                accum[i][j] = float2(cm.thread_elements()[0], cm.thread_elements()[1]);
            }
        }
    }
    // Each tap completes its channel contraction before its rounded addition.
    #pragma unroll
    for (uint i = 0; i < ROW_FRAGMENTS; ++i) {
        #pragma unroll
        for (uint j = 0; j < COLUMN_FRAGMENTS; ++j) {
            output[i][j] = tap_mask == 0 ? accum[i][j] : output[i][j] + accum[i][j];
        }
    }
}

#pragma unroll
for (uint i = 0; i < ROW_FRAGMENTS; ++i) {
    if (!row_ok[i]) continue;
    #pragma unroll
    for (uint j = 0; j < COLUMN_FRAGMENTS; ++j) {
        uint oc = column + 8 * j;
        Index index = output_offset[i] + oc * OUTPUT_SPATIAL;
        // Add the matching encoder value after all the rounded tap additions.
        if (FULL_COLUMNS || oc < OUT_CHANNELS) {
            Y[index] = ADD_SKIP ? output[i][j].x + SKIP[index] : output[i][j].x;
        }
        if (FULL_COLUMNS || oc + 1 < OUT_CHANNELS) {
            uint next = index + OUTPUT_SPATIAL;
            Y[next] = ADD_SKIP ? output[i][j].y + SKIP[next] : output[i][j].y;
        }
    }
}
"""


class AutomaticUpsample:
    """Apply kernel 3, stride 2, padding 1, and output padding 1 without bias.

    ``weight`` is the Torch ``ConvTranspose3d`` layout ``(in, out, 3, 3, 3)``.
    Map float32 ``(B, in, D, H, W)`` to ``(B, out, 2D, 2H, 2W)``. Write all
    eight output parities directly, without allocating convolution or interleave
    workspaces.
    """

    def __init__(self, weight: array, *, mx: MlxRuntime) -> None:
        """Pack the weights in tap, input channel, and output channel order."""
        self.mx = mx
        self.in_channels, self.out_channels = weight.shape[:2]
        self.weights = mx.contiguous(
            mx.reshape(mx.transpose(weight, (2, 3, 4, 0, 1)), (27, *weight.shape[:2]))
        )
        mx.eval(self.weights)
        self.kernel = mx.fast.metal_kernel(
            name="vista_ordered_stride2_deconv",
            input_names=["X", "W", "SKIP"],
            output_names=["Y"],
            source=_SOURCE,
            header="#include <metal_simdgroup_matrix>\n#pragma clang fp contract(off)\n",
            compile_options={"math_mode": "safe"},
        )

    def __call__(self, x: array, *, skip: array | None = None) -> array:
        """Run all taps and optionally add a matching float32 NCDHW skip before storing."""
        batch, _, depth, height, width = x.shape
        output_shape = (batch, self.out_channels, 2 * depth, 2 * height, 2 * width)
        if skip is not None and (skip.shape != output_shape or skip.dtype != self.mx.float32):
            raise ValueError("VISTA3D upsampling skip must match the float32 output shape")
        positions = batch * depth * height * width
        row_fragments = 4 if positions > 512 else 2
        column_fragments = (
            3
            if self.out_channels == 48 and positions > 512
            else min(4, (self.out_channels + 7) // 8)
        )
        simdgroups = 4 if positions > 512 else 2
        block_rows = 8 * row_fragments * simdgroups
        block_columns = 8 * column_fragments
        constants = {
            "BATCH": batch,
            "IN_CHANNELS": self.in_channels,
            "OUT_CHANNELS": self.out_channels,
            "DEPTH": depth,
            "HEIGHT": height,
            "WIDTH": width,
            "ROW_FRAGMENTS": row_fragments,
            "COLUMN_FRAGMENTS": column_fragments,
            "SIMDGROUPS": simdgroups,
            # Use the original 32-bit indexing on bundle windows. A standalone
            # upsample can exceed that range because it writes eight spatial phases.
            "LARGE_INDEX": max(x.size, self.weights.size, positions * self.out_channels * 8)
            >= 2**32,
            "ADD_SKIP": skip is not None,
        }
        return self.kernel(
            # With ADD_SKIP disabled, the kernel never reads the third binding.
            inputs=[x, self.weights, self.weights if skip is None else skip],
            template=list(constants.items()),
            grid=(
                32 * simdgroups,
                (positions + block_rows - 1) // block_rows,
                8 * ((self.out_channels + block_columns - 1) // block_columns),
            ),
            threadgroup=(32 * simdgroups, 1, 1),
            output_shapes=[output_shape],
            output_dtypes=[self.mx.float32],
        )[0]
