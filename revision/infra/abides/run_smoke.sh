#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLATFORM="${ABIDES_PLATFORM:-linux/amd64}"
REVISION="f9cbe51"
RMSC_IMAGE="ace-rl-abides-rmsc04:${REVISION}"
GYM_IMAGE="ace-rl-abides-gym:${REVISION}"

docker info >/dev/null

docker buildx build \
  --platform "${PLATFORM}" \
  --target abides-rmsc04 \
  --load \
  --tag "${RMSC_IMAGE}" \
  "${SCRIPT_DIR}"

docker run --rm --platform "${PLATFORM}" "${RMSC_IMAGE}" \
  --config rmsc04 \
  --seed 7 \
  --end-time 09:35:00 \
  --num-noise-agents 1000 \
  --num-value-agents 102 \
  --num-momentum-agents 12

docker buildx build \
  --platform "${PLATFORM}" \
  --target abides-gym \
  --load \
  --tag "${GYM_IMAGE}" \
  "${SCRIPT_DIR}"

docker run --rm --platform "${PLATFORM}" "${GYM_IMAGE}"
docker run --rm --platform "${PLATFORM}" \
  --entrypoint python "${GYM_IMAGE}" -m pip check
