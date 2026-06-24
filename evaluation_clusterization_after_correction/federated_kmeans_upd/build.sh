#!/usr/bin/env bash
set -euo pipefail

docker build -t fc_kmeans_upd:latest . -f Dockerfile
