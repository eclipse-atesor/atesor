# Atesor AI - RISC-V Porting Sandbox
#
# Native riscv64 build environment on Alpine Linux (musl). Under the qemu
# target the image runs emulated via binfmt_misc; on the native target the
# alpine platform keeps this same Dockerfile and runs it on real riscv64
# hardware.

# Base image: Alpine Linux for RISC-V 64-bit. The name is fully
# qualified, because podman on a native machine may have no short-name
# alias for "alpine".
FROM docker.io/library/alpine:latest

LABEL maintainer="Atesor AI"
LABEL description="Minimal RISC-V 64-bit sandbox for automated software porting"

# Shared build toolchain and frequently needed libraries. Package-specific
# dependencies are installed by the agent at run time, so this layer stays
# stable and cacheable.
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

# Go toolchain: pinned and checksum-verified from the official riscv64
# tarball, not `apk add go`. The distro package floats with the Alpine
# release, so image rebuilds would silently change the compiler.
ARG GO_VERSION=1.26.5
ARG GO_SHA256=d4a24dd4484d3f86b99c2d300af0dea5d184557e6d61eb7aba19ff61662750e3
# SHA256 source: https://go.dev/dl/?mode=json&include=all
RUN mkdir -p /opt/atesor-downloads \
    && curl -fsSL \
        "https://go.dev/dl/go${GO_VERSION}.linux-riscv64.tar.gz" \
        -o /opt/atesor-downloads/go.tar.gz \
    && echo "${GO_SHA256}  /opt/atesor-downloads/go.tar.gz" \
        | sha256sum -c - \
    && tar -xzf /opt/atesor-downloads/go.tar.gz -C /usr/local \
    && rm -f /opt/atesor-downloads/go.tar.gz \
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

# Compile natively inside the sandbox; Atesor never cross-compiles.
ENV CC=gcc
ENV CXX=g++
ENV GOPATH=/root/go
ENV PATH="${CARGO_HOME}/bin:${GOPATH}/bin:/usr/local/go/bin:${PATH}"
# Fetch modules through the proxy to avoid network issues in isolated
# containers.
ENV GOPROXY=https://proxy.golang.org,direct
ENV GONOSUMCHECK=*
# -buildvcs=false: builds run as root, and Go's VCS stamping would trip
# git's repository ownership check.
ENV GOFLAGS=-buildvcs=false
# Pin the bundled toolchain hard: never let `go` auto-download a newer
# version mid-build, because such downloads regularly OOM-kill (exit 137)
# under QEMU emulation and produce empty stderr that masks the real error.
ENV GOTOOLCHAIN=local

# The agent's path constants (src/config.py) expect this layout in the
# container.
WORKDIR /workspace
RUN mkdir -p /workspace/repos /workspace/output /workspace/logs /workspace/patches

# An anonymous volume: each new container gets a new, empty one. The host
# process writes recipes and reports to `OUTPUT_DIR` (src/config.py).
VOLUME /workspace/output

# Health check: a working gcc stands in for the whole toolchain.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD gcc --version || exit 1

# Idle forever; the agent drives builds with `docker exec` (native target:
# `podman exec` over SSH).
CMD ["tail", "-f", "/dev/null"]
