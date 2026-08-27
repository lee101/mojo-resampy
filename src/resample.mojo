"""Table-driven band-limited interpolation kernels for mojo-resampy."""

from std.sys.info import simd_width_of as simdwidthof


comptime F64Ptr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime F32Ptr = UnsafePointer[Float32, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 32768
comptime PARALLEL_GRAIN = 4096


@always_inline
def load_weights[W: Int](
    window: F64Ptr,
    delta: F64Ptr,
    offset: Int,
    eta: Float64,
) -> SIMD[DType.float64, W]:
    return (
        window.load[width=W](offset)
        + eta * delta.load[width=W](offset)
    )


def resample_f64_range(
    x: F64Ptr,
    times: F64Ptr,
    window: F64Ptr,
    delta: F64Ptr,
    dst: F64Ptr,
    n_series: Int,
    n_input: Int,
    n_output: Int,
    n_window: Int,
    table_size: Int,
    scale: Float64,
    time_scale: Float64,
    window_scale: Float64,
    phase_taps: Int,
    components: Int,
    t_start: Int,
    t_end: Int,
):
    comptime W = simdwidthof[DType.float64]()
    var index_step = Int(scale * Float64(table_size))
    if index_step < 1:
        index_step = 1

    for series in range(n_series):
        for t in range(t_start, t_end):
            var time_register = times[t] * time_scale
            var n = Int(time_register)
            var frac = scale * (time_register - Float64(n))
            var index_frac = frac * Float64(table_size)
            var left_offset = Int(index_frac)
            var left_eta = index_frac - Float64(left_offset)
            var left_count = min(
                n + 1, (n_window - left_offset) // index_step
            )
            frac = scale - frac
            index_frac = frac * Float64(table_size)
            var right_offset = Int(index_frac)
            var right_eta = index_frac - Float64(right_offset)
            var right_count = min(
                n_input - n - 1, (n_window - right_offset) // index_step
            )
            var packed_taps = abs(phase_taps)
            var left_base = left_offset * packed_taps
            var right_base = right_offset * packed_taps
            if phase_taps > 0 and right_offset == table_size:
                right_base = 1

            for component in range(components):
                var acc = Float64(0)
                if phase_taps == 0:
                    for i in range(left_count):
                        var wi = left_offset + i * index_step
                        var weight = window[wi] + left_eta * delta[wi]
                        var xi = (
                            (series * n_input + n - i) * components
                        ) + component
                        acc += weight * x[xi]
                    for k in range(right_count):
                        var wi = right_offset + k * index_step
                        var weight = window[wi] + right_eta * delta[wi]
                        var xi = (
                            (series * n_input + n + k + 1) * components
                        ) + component
                        acc += weight * x[xi]
                    var yi = (
                        (series * n_output + t) * components
                    ) + component
                    dst[yi] = acc * window_scale
                    continue

                var acc_vec = SIMD[DType.float64, W](0)
                var acc_vec_2 = SIMD[DType.float64, W](0)
                var i = 0
                while i + 2 * W <= left_count:
                    var wi = left_base + i
                    var xi = ((series * n_input + n - i) * components) + component
                    var weights = load_weights[W](
                        window, delta, wi, left_eta
                    )
                    var samples = (x + xi).strided_load[width=W](-components)
                    acc_vec += weights * samples
                    var i2 = i + W
                    wi = left_base + i2
                    xi = ((series * n_input + n - i2) * components) + component
                    weights = load_weights[W](
                        window, delta, wi, left_eta
                    )
                    samples = (x + xi).strided_load[width=W](-components)
                    acc_vec_2 += weights * samples
                    i += 2 * W
                while i + W <= left_count:
                    var wi = left_base + i
                    var xi = ((series * n_input + n - i) * components) + component
                    var weights = load_weights[W](
                        window, delta, wi, left_eta
                    )
                    var samples = (x + xi).strided_load[width=W](-components)
                    acc_vec += weights * samples
                    i += W
                acc += (acc_vec + acc_vec_2).reduce_add()
                while i < left_count:
                    var wi = left_base + i
                    var weight = window[wi] + left_eta * delta[wi]
                    var xi = ((series * n_input + n - i) * components) + component
                    acc += weight * x[xi]
                    i += 1

                acc_vec = SIMD[DType.float64, W](0)
                acc_vec_2 = SIMD[DType.float64, W](0)
                var k = 0
                while k + 2 * W <= right_count:
                    var wi = right_base + k
                    var xi = (
                        (series * n_input + n + k + 1) * components
                    ) + component
                    var weights = load_weights[W](
                        window, delta, wi, right_eta
                    )
                    if components == 1:
                        acc_vec += weights * x.load[width=W](xi)
                    else:
                        acc_vec += (
                            weights
                            * (x + xi).strided_load[width=W](components)
                        )
                    var k2 = k + W
                    wi = right_base + k2
                    xi = (
                        (series * n_input + n + k2 + 1) * components
                    ) + component
                    weights = load_weights[W](
                        window, delta, wi, right_eta
                    )
                    if components == 1:
                        acc_vec_2 += weights * x.load[width=W](xi)
                    else:
                        acc_vec_2 += (
                            weights
                            * (x + xi).strided_load[width=W](components)
                        )
                    k += 2 * W
                while k + W <= right_count:
                    var wi = right_base + k
                    var xi = (
                        (series * n_input + n + k + 1) * components
                    ) + component
                    var weights = load_weights[W](
                        window, delta, wi, right_eta
                    )
                    if components == 1:
                        acc_vec += weights * x.load[width=W](xi)
                    else:
                        acc_vec += (
                            weights
                            * (x + xi).strided_load[width=W](components)
                        )
                    k += W
                acc += (acc_vec + acc_vec_2).reduce_add()
                while k < right_count:
                    var wi = right_base + k
                    var weight = window[wi] + right_eta * delta[wi]
                    var xi = (
                        (series * n_input + n + k + 1) * components
                    ) + component
                    acc += weight * x[xi]
                    k += 1

                var yi = ((series * n_output + t) * components) + component
                dst[yi] = acc * window_scale


def resample_f64(
    x_addr: Int,
    times_addr: Int,
    window_addr: Int,
    delta_addr: Int,
    dst_addr: Int,
    n_series: Int,
    n_input: Int,
    n_output: Int,
    n_window: Int,
    table_size: Int,
    scale: Float64,
    time_scale: Float64,
    window_scale: Float64,
    phase_taps: Int,
    components: Int,
    use_parallel: Int,
):
    if use_parallel == 0 or n_series * n_output < PARALLEL_THRESHOLD:
        resample_f64_range(
            F64Ptr(unsafe_from_address=x_addr),
            F64Ptr(unsafe_from_address=times_addr),
            F64Ptr(unsafe_from_address=window_addr),
            F64Ptr(unsafe_from_address=delta_addr),
            F64Ptr(unsafe_from_address=dst_addr),
            n_series,
            n_input,
            n_output,
            n_window,
            table_size,
            scale,
            time_scale,
            window_scale,
            phase_taps,
            components,
            0,
            n_output,
        )
        return

    var chunks = (n_output + PARALLEL_GRAIN - 1) // PARALLEL_GRAIN

    def worker(
        chunk: Int,
    ) {
        imm x_addr,
        imm times_addr,
        imm window_addr,
        imm delta_addr,
        imm dst_addr,
        imm n_series,
        imm n_input,
        imm n_output,
        imm n_window,
        imm table_size,
        imm scale,
        imm time_scale,
        imm window_scale,
        imm phase_taps,
        imm components,
    }:
        var t_start = chunk * PARALLEL_GRAIN
        var t_end = min(t_start + PARALLEL_GRAIN, n_output)
        resample_f64_range(
            F64Ptr(unsafe_from_address=x_addr),
            F64Ptr(unsafe_from_address=times_addr),
            F64Ptr(unsafe_from_address=window_addr),
            F64Ptr(unsafe_from_address=delta_addr),
            F64Ptr(unsafe_from_address=dst_addr),
            n_series,
            n_input,
            n_output,
            n_window,
            table_size,
            scale,
            time_scale,
            window_scale,
            phase_taps,
            components,
            t_start,
            t_end,
        )

    for chunk in range(chunks):
        worker(chunk)


def resample_f32(
    x: F32Ptr,
    times: F64Ptr,
    window: F64Ptr,
    delta: F64Ptr,
    dst: F32Ptr,
    n_series: Int,
    n_input: Int,
    n_output: Int,
    n_window: Int,
    table_size: Int,
    scale: Float64,
    time_scale: Float64,
    window_scale: Float64,
    components: Int,
):
    var index_step = Int(scale * Float64(table_size))
    if index_step < 1:
        index_step = 1

    for series in range(n_series):
        for t in range(n_output):
            var time_register = times[t] * time_scale
            var n = Int(time_register)
            var frac = scale * (time_register - Float64(n))
            var index_frac = frac * Float64(table_size)
            var left_offset = Int(index_frac)
            var left_eta = index_frac - Float64(left_offset)
            var left_count = min(
                n + 1, (n_window - left_offset) // index_step
            )
            frac = scale - frac
            index_frac = frac * Float64(table_size)
            var right_offset = Int(index_frac)
            var right_eta = index_frac - Float64(right_offset)
            var right_count = min(
                n_input - n - 1, (n_window - right_offset) // index_step
            )

            for component in range(components):
                var acc = Float32(0)
                for i in range(left_count):
                    var wi = left_offset + i * index_step
                    var weight = window[wi] + left_eta * delta[wi]
                    var xi = ((series * n_input + n - i) * components) + component
                    acc = Float32(Float64(acc) + weight * Float64(x[xi]))

                for k in range(right_count):
                    var wi = right_offset + k * index_step
                    var weight = window[wi] + right_eta * delta[wi]
                    var xi = (
                        (series * n_input + n + k + 1) * components
                    ) + component
                    acc = Float32(Float64(acc) + weight * Float64(x[xi]))

                var yi = ((series * n_output + t) * components) + component
                dst[yi] = Float32(Float64(acc) * window_scale)


def resample_f64_to_f32(
    x: F64Ptr,
    times: F64Ptr,
    window: F64Ptr,
    delta: F64Ptr,
    dst: F32Ptr,
    n_series: Int,
    n_input: Int,
    n_output: Int,
    n_window: Int,
    table_size: Int,
    scale: Float64,
    time_scale: Float64,
    window_scale: Float64,
):
    var index_step = Int(scale * Float64(table_size))
    if index_step < 1:
        index_step = 1

    for series in range(n_series):
        for t in range(n_output):
            var time_register = times[t] * time_scale
            var n = Int(time_register)
            var frac = scale * (time_register - Float64(n))
            var index_frac = frac * Float64(table_size)
            var left_offset = Int(index_frac)
            var left_eta = index_frac - Float64(left_offset)
            var left_count = min(
                n + 1, (n_window - left_offset) // index_step
            )
            frac = scale - frac
            index_frac = frac * Float64(table_size)
            var right_offset = Int(index_frac)
            var right_eta = index_frac - Float64(right_offset)
            var right_count = min(
                n_input - n - 1, (n_window - right_offset) // index_step
            )
            var acc = Float32(0)
            for i in range(left_count):
                var wi = left_offset + i * index_step
                var weight = window[wi] + left_eta * delta[wi]
                var xi = series * n_input + n - i
                acc = Float32(Float64(acc) + weight * x[xi])
            for k in range(right_count):
                var wi = right_offset + k * index_step
                var weight = window[wi] + right_eta * delta[wi]
                var xi = series * n_input + n + k + 1
                acc = Float32(Float64(acc) + weight * x[xi])
            dst[series * n_output + t] = Float32(Float64(acc) * window_scale)


@export("mrs_resample_f64")
def mrs_resample_f64(
    x_addr: Int,
    times_addr: Int,
    window_addr: Int,
    delta_addr: Int,
    dst_addr: Int,
    n_series: Int,
    n_input: Int,
    n_output: Int,
    n_window: Int,
    table_size: Int,
    scale: Float64,
    time_scale: Float64,
    window_scale: Float64,
    phase_taps: Int,
    components: Int,
    use_parallel: Int,
) abi("C"):
    resample_f64(
        x_addr,
        times_addr,
        window_addr,
        delta_addr,
        dst_addr,
        n_series,
        n_input,
        n_output,
        n_window,
        table_size,
        scale,
        time_scale,
        window_scale,
        phase_taps,
        components,
        use_parallel,
    )


@export("mrs_resample_f64_range")
def mrs_resample_f64_range(
    x_addr: Int,
    times_addr: Int,
    window_addr: Int,
    delta_addr: Int,
    dst_addr: Int,
    n_series: Int,
    n_input: Int,
    n_output: Int,
    n_window: Int,
    table_size: Int,
    scale: Float64,
    time_scale: Float64,
    window_scale: Float64,
    phase_taps: Int,
    components: Int,
    t_start: Int,
    t_end: Int,
) abi("C"):
    resample_f64_range(
        F64Ptr(unsafe_from_address=x_addr),
        F64Ptr(unsafe_from_address=times_addr),
        F64Ptr(unsafe_from_address=window_addr),
        F64Ptr(unsafe_from_address=delta_addr),
        F64Ptr(unsafe_from_address=dst_addr),
        n_series,
        n_input,
        n_output,
        n_window,
        table_size,
        scale,
        time_scale,
        window_scale,
        phase_taps,
        components,
        t_start,
        t_end,
    )


@export("mrs_resample_f64_to_f32")
def mrs_resample_f64_to_f32(
    x_addr: Int,
    times_addr: Int,
    window_addr: Int,
    delta_addr: Int,
    dst_addr: Int,
    n_series: Int,
    n_input: Int,
    n_output: Int,
    n_window: Int,
    table_size: Int,
    scale: Float64,
    time_scale: Float64,
    window_scale: Float64,
    phase_taps: Int,
    components: Int,
    use_parallel: Int,
) abi("C"):
    _ = phase_taps
    _ = use_parallel
    resample_f64_to_f32(
        F64Ptr(unsafe_from_address=x_addr),
        F64Ptr(unsafe_from_address=times_addr),
        F64Ptr(unsafe_from_address=window_addr),
        F64Ptr(unsafe_from_address=delta_addr),
        F32Ptr(unsafe_from_address=dst_addr),
        n_series,
        n_input,
        n_output,
        n_window,
        table_size,
        scale,
        time_scale,
        window_scale,
    )


@export("mrs_resample_f32")
def mrs_resample_f32(
    x_addr: Int,
    times_addr: Int,
    window_addr: Int,
    delta_addr: Int,
    dst_addr: Int,
    n_series: Int,
    n_input: Int,
    n_output: Int,
    n_window: Int,
    table_size: Int,
    scale: Float64,
    time_scale: Float64,
    window_scale: Float64,
    phase_taps: Int,
    components: Int,
    use_parallel: Int,
) abi("C"):
    _ = phase_taps
    _ = use_parallel
    resample_f32(
        F32Ptr(unsafe_from_address=x_addr),
        F64Ptr(unsafe_from_address=times_addr),
        F64Ptr(unsafe_from_address=window_addr),
        F64Ptr(unsafe_from_address=delta_addr),
        F32Ptr(unsafe_from_address=dst_addr),
        n_series,
        n_input,
        n_output,
        n_window,
        table_size,
        scale,
        time_scale,
        window_scale,
        components,
    )
