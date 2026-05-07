# ══════════════════════════════════════════════════════════════════════════════
# Dockerfile — IDBI Gift Voucher Recommendation Engine
#
# Mirrors the exact Dockerfile architecture from the project report:
#   - python:3.9-slim base image (Debian-based)
#   - libgomp1 installation (required for implicit/ALS parallelism)
#   - UV multi-stage copy from ghcr.io/astral-sh/uv
#   - UV_FROZEN=1 for strict lock-file compliance
# ══════════════════════════════════════════════════════════════════════════════

# Stage 1: Pull the UV binary (Rust-based package manager)
FROM ghcr.io/astral-sh/uv:latest AS uv-source

# Stage 2: Application image
FROM python:3.9-slim

# ── Environment flags (as per report Table 3.3) ────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_FROZEN=1
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# ── System dependencies ────────────────────────────────────────────────────
# libgomp1: OpenMP runtime for implicit/ALS parallel CPU computation
# This resolves the critical libgomp compatibility barrier (Section 3.4)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
        git && \
    rm -rf /var/lib/apt/lists/*

# ── Copy UV binary from the UV stage ──────────────────────────────────────
COPY --from=uv-source /uv /usr/local/bin/uv

# ── Set working directory ──────────────────────────────────────────────────
WORKDIR /app

# ── Copy dependency files first (layer cache optimisation) ─────────────────
COPY pyproject.toml uv.lock requirements.txt ./

# ── Install dependencies via UV (frozen = strict lock-file compliance) ─────
# --no-install-project: install deps only, not the project package itself
RUN uv sync --frozen --no-install-project

# ── Copy application source code ───────────────────────────────────────────
COPY . .

# ── Default command: run the training pipeline ─────────────────────────────
CMD ["uv", "run", "python", "train_pipeline.py"]
