#!/usr/bin/env python3
"""Compile transformer kernels with NVRTC without a GPU. Usage: python tools/check_cuda_kernels.py /path/to/libnvrtc.so.12"""
import ctypes as C
import sys
from pathlib import Path
lib = C.CDLL(sys.argv[1])
source = (Path(__file__).resolve().parents[1] / 'src/main/resources/xyz/gatoware/synapse/backend/transformer.cu').read_bytes()
for arch in (b'compute_52', b'compute_86'):
    program = C.c_void_p()
    rc = lib.nvrtcCreateProgram(C.byref(program), C.c_char_p(source), C.c_char_p(b'transformer.cu'), 0, None, None)
    if rc: raise RuntimeError(f'nvrtcCreateProgram: {rc}')
    try:
        options = (C.c_char_p * 1)(b'--gpu-architecture=' + arch)
        rc = lib.nvrtcCompileProgram(program, 1, options)
        size = C.c_size_t()
        lib.nvrtcGetProgramLogSize(program, C.byref(size))
        log = C.create_string_buffer(size.value)
        lib.nvrtcGetProgramLog(program, log)
        if log.value: print(log.value.decode())
        if rc: raise RuntimeError(f'nvrtcCompileProgram: {rc}')
        print(f'{arch.decode()}: compiled successfully (no GPU execution)')
    finally:
        lib.nvrtcDestroyProgram(C.byref(program))
