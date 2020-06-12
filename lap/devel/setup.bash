#!/usr/bin/env bash

LAP_LOOT=$(builtin cd "`dirname "${BASH_SOURCE[0]}"`" > /dev/null && pwd)/..
export PATH=${LAP_LOOT}/bin:${PATH}
