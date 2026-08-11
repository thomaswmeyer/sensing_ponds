# Custom minimal ONNX Runtime Web build.
#
# Compiles ORT to WASM with only the operator kernels this project's model needs
# -- 7 operator groups instead of the several hundred in the stock build.
# Expected ~3 MB raw / ~1 MB gzipped, against 13.5 MB / 3.5 MB stock.
#
# Pinned to ORT v1.20.1 to match the onnxruntime-web version in web/package.json.
# The JS glue and the WASM binary are a matched pair -- mixing versions fails at
# runtime with unhelpful errors.
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

WORKDIR /src

# Shallow clone with submodules: the full history is ~4 GB and unnecessary.
ARG ORT_VERSION=v1.20.1
RUN git clone --depth 1 --branch ${ORT_VERSION} --recursive \
      https://github.com/microsoft/onnxruntime.git

# The operator config generated from our model. Regenerate whenever the model
# ARCHITECTURE changes -- weights do not matter, but a different backbone or a
# change between fp32 and INT8 alters the op set and invalidates this build.
COPY required_operators_and_types.config /src/ops.config

WORKDIR /src/onnxruntime

# --minimal_build extended:
#     Drops the ONNX format parser and runtime optimisers. Requires .ort format
#     models, which is why the training export must emit one.
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
      --include_ops_by_config /src/ops.config \
      --enable_reduced_operator_type_support \
      --disable_ml_ops \
      --disable_rtti \
      --disable_exceptions \
      --skip_tests \
      --parallel \
      --emsdk_version 3.1.59

# Artefacts land here; extract with `docker cp` (see README).
RUN mkdir -p /out && \
    find build/Linux/MinSizeRel -maxdepth 1 \( -name '*.wasm' -o -name '*.mjs' -o -name '*.js' \) \
      -exec cp {} /out/ \; && \
    ls -la /out
