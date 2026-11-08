FROM debian:bookworm-slim
ARG TARGETARCH
ARG TARGETVARIANT

WORKDIR /usr/src

COPY ./pyproject.toml ./
RUN \
    apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
    \
    && python3 -m venv .venv \
    && .venv/bin/pip3 install --no-cache-dir -U \
        setuptools \
        wheel \
    && .venv/bin/pip3 install --no-cache-dir \
        -e '.[zeroconf,hass]' \
    \
    && rm -rf /var/lib/apt/lists/*

COPY ./ ./

# Keep every cache on the /data volume instead of the container's filesystem.
# Model files already go there (--download-dir defaults to the first --data-dir),
# but huggingface_hub also writes a cache that no per-call argument reaches: the
# Xet chunk cache used during downloads is $HF_HOME/xet and can run to several
# GB. Anything else falls back to $XDG_CACHE_HOME. Unset, all of that lands in
# $HOME/.cache - which is thrown away on every container recreate, and is an
# unwritable /.cache when the container runs as a uid with no passwd entry.
#
# These have to be environment variables: huggingface_hub reads them into
# constants at import time, so setting them from Python would be too late.
ENV HF_HOME=/data \
    XDG_CACHE_HOME=/data

EXPOSE 10300

# The server only starts listening once the model is loaded, which means
# downloading it on first run -- hence the long start period, during which
# failures don't count against --retries. --retries covers the other direction:
# transcription runs on the event loop, so a check can time out behind a long
# request without the server being unhealthy.
HEALTHCHECK --interval=30s --timeout=20s --start-period=5m --retries=3 \
    CMD ["/usr/src/.venv/bin/python3", "-m", "wyoming_faster_whisper.health_check"]

ENTRYPOINT ["bash", "docker_run.sh"]
