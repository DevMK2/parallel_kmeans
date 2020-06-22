#!/usr/bin/env bash

KM_ROOT=$(builtin cd "`dirname "${BASH_SOURCE[0]}"`" > /dev/null && pwd)
. "${KM_ROOT}/lap/devel/setup.bash"
export PATH=${KM_ROOT}/kmeans/utils:${PATH}