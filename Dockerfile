# syntax=docker/dockerfile:1
# --- Aşama 1: frontend (Leptos/WASM) derlemesi ---------------------------
FROM rust:1-slim-bookworm AS frontend-builder
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN rustup target add wasm32-unknown-unknown \
    && curl -fsSL https://github.com/trunk-rs/trunk/releases/latest/download/trunk-x86_64-unknown-linux-gnu.tar.gz \
       | tar -xz -C /usr/local/bin
WORKDIR /src
COPY crates/dq-web ./crates/dq-web
WORKDIR /src/crates/dq-web
RUN trunk build --release

# --- Aşama 2: backend (dq-server) derlemesi -------------------------------
FROM rust:1-slim-bookworm AS backend-builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config libssl-dev build-essential clang \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates/dq-core ./crates/dq-core
COPY crates/dq-ingest ./crates/dq-ingest
COPY crates/dq-index ./crates/dq-index
COPY crates/dq-llm ./crates/dq-llm
COPY crates/dq-guard ./crates/dq-guard
COPY crates/dq-rag ./crates/dq-rag
COPY crates/dq-server ./crates/dq-server
# dq-web workspace disi tutuluyor (root Cargo.toml exclude eder); yalnizca
# derlenebilir bir stub birakiyoruz ki workspace cozumlemesi bozulmasin.
RUN cargo build --release -p dq-server

# --- Aşama 3: calisma zamani imaji -----------------------------------------
FROM debian:bookworm-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl tesseract-ocr tesseract-ocr-tur tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --uid 10001 dqapp

WORKDIR /app
COPY --from=backend-builder /src/target/release/dq-server ./dq-server
COPY --from=frontend-builder /src/crates/dq-web/dist ./static
COPY config ./config

RUN mkdir -p /app/data /app/models && chown -R dqapp:dqapp /app
USER dqapp

ENV DQ_CONFIG=/app/config/default.toml \
    DQ_STATIC_DIR=/app/static \
    DQ_SERVER_HOST=0.0.0.0 \
    RUST_LOG=info

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/api/live || exit 1

ENTRYPOINT ["/app/dq-server"]
