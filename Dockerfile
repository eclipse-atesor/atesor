# Atesor AI - RISC-V Porting Sandbox
# 
# This Dockerfile creates a native RISC-V 64-bit development environment using Alpine Linux.
# It uses QEMU user-mode emulation via Docker's multi-platform support (binfmt_misc).

# Base image: Alpine Linux for RISC-V 64-bit
FROM alpine:latest

LABEL maintainer="Atesor AI"
LABEL description="Minimal RISC-V 64-bit sandbox for automated software porting"

# Install essentials for building and analysis
RUN apk add --no-cache \
    build-base \
    cmake \
    git \
    curl \
    bash \
    python3 \
    ca-certificates \
    file \
    tar \
    xz \
    pkgconfig \
    meson \
    ninja \
    autoconf \
    automake \
    libtool \
    gettext-dev \
    nasm \
    perl \
    linux-headers \
    zlib-dev \
    samurai \
    py3-pip \
    py3-jinja2 \
    py3-jsonschema \
    gfortran \
    texinfo \
    patch \
    diffutils \
    protobuf-dev \
    protoc \
    abseil-cpp-dev \
    openssl-dev \
    libpng-dev \
    util-linux \
    coreutils

# Install Go from the official riscv64 tarball instead of `apk add go`.
ARG GO_VERSION=1.26.5
RUN curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-riscv64.tar.gz" \
    | tar -xz -C /usr/local \
    && /usr/local/go/bin/go version

# Install a pinned Rust toolchain via rustup instead of `apk add rust cargo`.
# Alpine's distro rust lags upstream and skews between images; a runtime
# `apk add rust cargo` from an LLM would otherwise shadow whatever we bake
# in. Alpine ships `rustup` (musl-native rustup-init) in the community
# repo, so we do not need gcompat here.
ARG RUST_TOOLCHAIN=1.90.0
ENV CARGO_HOME=/root/.cargo
ENV RUSTUP_HOME=/root/.rustup
RUN apk add --no-cache rustup \
    && rustup-init -y \
        --profile minimal \
        --default-toolchain ${RUST_TOOLCHAIN} \
        --no-modify-path \
    && /root/.cargo/bin/rustc --version \
    && /root/.cargo/bin/cargo --version

# Set up environment variables for native compilation inside the sandbox
ENV CC=gcc
ENV CXX=g++
# Go environment: use direct proxy to avoid network issues in isolated containers
ENV GOPATH=/root/go
ENV PATH="${CARGO_HOME}/bin:${GOPATH}/bin:/usr/local/go/bin:${PATH}"
ENV GOPROXY=https://proxy.golang.org,direct
ENV GONOSUMCHECK=*
ENV GOFLAGS=-buildvcs=false
# Pin the bundled toolchain hard: never let `go` auto-download a newer
# version mid-build, because such downloads regularly OOM-kill (exit 137)
# under QEMU emulation and produce empty stderr that masks the real error.
ENV GOTOOLCHAIN=local

# Set up workspace structure matching the agent's expectations
WORKDIR /workspace
RUN mkdir -p /workspace/repos /workspace/output /workspace/logs /workspace/patches

# Volume for persistent output
VOLUME /workspace/output

# Health check to ensure the environment is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD gcc --version || exit 1

# Keep container running for the agent to connect and execute commands
CMD ["tail", "-f", "/dev/null"]
