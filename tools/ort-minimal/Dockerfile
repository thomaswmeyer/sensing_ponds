# Custom minimal ONNX Runtime Web build.
#
# Compiles ORT to WASM with only the operator kernels this project's model needs
# -- 5 operator groups instead of the several hundred in the stock build.
# Expected ~3 MB raw / ~1 MB gzipped, against 12.86 MB / 3.28 MB stock.
#
# Pinned to ORT v1.27.0 to match the onnxruntime-web version in web/package.json.
# The JS glue and the WASM binary are a matched pair -- mixing versions fails at
# runtime with unhelpful errors. If you bump one, bump the other.
#
# Build takes 1-3 hours on first run: it compiles emsdk, protobuf and ORT itself.
# See tools/ort-minimal/README.md.

FROM --platform=linux/amd64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      cmake \
      curl \
      git \
      ninja-build \
      python3 \
      python3-dev \
      python3-pip \
      xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir packaging numpy setuptools wheel

# ORT's build.sh refuses to run as root unless --allow_running_as_root is passed.
# Using a real unprivileged user is the better answer: emsdk writes into its own
# tree during install, and running that as root leaves root-owned artefacts that
# are awkward to extract.
RUN useradd -m -s /bin/bash builder && mkdir -p /src /out && chown builder:builder /src /out
USER builder

WORKDIR /src

# Shallow clone with submodules: the full history is ~4 GB and unnecessary.
ARG ORT_VERSION=v1.27.0
RUN git clone --depth 1 --branch ${ORT_VERSION} --recursive \
      https://github.com/microsoft/onnxruntime.git

# The operator config generated from our model. Regenerate whenever the model
# ARCHITECTURE or PRECISION changes -- weights do not matter, but a different
# backbone, or a change between fp16 and fp32, alters the op set and invalidates
# this build. See the header of the config file for how to regenerate it.
COPY --chown=builder:builder required_operators_and_types.config /src/ops.config

WORKDIR /src/onnxruntime

# --minimal_build extended:
#     Drops the ONNX format parser and the full optimiser set. Requires .ort
#     format models, which is why the export step must emit one.
# --enable_runtime_optimizations:
#     REQUIRED, and not merely a tuning knob -- without it this build throws
#     away the entire fp16 saving.
#
#     ORT's CPU execution provider has no fp16 kernels, so it upcasts every
#     fp16 initialiser to float32 during graph optimisation. On the normal
#     .onnx path that happens in memory at session creation, so the DOWNLOAD
#     stays fp16 and only RAM holds float32. But .ort format serialises the
#     post-optimisation graph, which bakes that upcast into the file: converting
#     our 2.69 MB gz fp16 model with fixed optimisations yields a 3.45 MB gz
#     .ort -- bigger than the fp32 ONNX we replaced.
#
#     Runtime optimisations defer fusion and the upcast to session creation, so
#     the .ort keeps fp16 weights on disk: 2.73 MB gz, matching the ONNX. The
#     op config must be the matching .with_runtime_opt.config, which is what
#     tools/ort-minimal/required_operators_and_types.config now holds.
# --include_ops_by_config:
#     Compiles only the kernels our model uses.
# --enable_reduced_operator_type_support:
#     Also drops unused data types per operator. Needs the type annotations the
#     config carries (generated with --enable_type_reduction).
# --disable_ml_ops / --disable_rtti / --disable_exceptions:
#     Classical-ML kernels, C++ RTTI and exceptions are all unused here.
# --skip_tests:
#     The test binaries are larger than the artefact we want.
RUN ./build.sh \
      --config MinSizeRel \
      --build_wasm \
      --minimal_build extended \
      --enable_runtime_optimizations \
      --include_ops_by_config /src/ops.config \
      --enable_reduced_operator_type_support \
      --disable_ml_ops \
      --disable_rtti \
      --disable_exceptions \
      --skip_tests \
      --parallel \
      --emsdk_version 3.1.59

# Artefacts land here; extract with `docker cp` (see README).
RUN find build/Linux/MinSizeRel -maxdepth 1 \( -name '*.wasm' -o -name '*.mjs' -o -name '*.js' \) \
      -exec cp {} /out/ \; && \
    ls -la /out
