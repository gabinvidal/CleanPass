#!/bin/sh
# Builds the test libraries used by the Python validation scripts (Linux).
#   OIDN=/path/to/oidn-2.5.1.x86_64.linux sh build_linux.sh
set -e
cd "$(dirname "$0")"
: "${OIDN:?set OIDN to an OIDN 2.5 SDK folder}"
S=../../src
L="-I$OIDN/include -L$OIDN/lib -lOpenImageDenoise -Wl,-rpath,$OIDN/lib -pthread"
g++ -std=c++17 -O2 -fPIC -shared -o libfirefly.so capi.cpp $S/cp_core.cpp $S/cp_oidn.cpp $L
g++ -std=c++17 -O2 -fPIC -shared -o libdeflicker.so capi2.cpp $S/cp_deflicker.cpp -pthread
g++ -std=c++17 -O2 -fPIC -shared -o libpipe.so capi3.cpp $S/cp_pipeline.cpp $S/cp_core.cpp $S/cp_deflicker.cpp $S/cp_oidn.cpp $L
g++ -std=c++17 -g -O1 -fsanitize=address,undefined asan_main.cpp $S/cp_core.cpp -pthread -o asan_main
g++ -std=c++17 -g -O1 -fsanitize=address,undefined asan_pipe.cpp $S/cp_pipeline.cpp $S/cp_core.cpp $S/cp_deflicker.cpp -pthread -o asan_pipe
echo "built"
# full pipeline (motion vectors, depth, variance, stabilised denoise, AOV denoise)
g++ -std=c++17 -O2 -fPIC -shared -o libpipe4.so capi4.cpp $S/cp_pipeline.cpp $S/cp_core.cpp $S/cp_deflicker.cpp $S/cp_oidn.cpp $L
