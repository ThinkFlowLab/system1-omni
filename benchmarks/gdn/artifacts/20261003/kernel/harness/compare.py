"""Isolated native GDN inverse A/B. Run through the GPU scheduler."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def write(row):
    with (ROOT / 'analysis/results.jsonl').open('a') as f:
        f.write(json.dumps(row) + '\n')
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    plan = json.loads((ROOT / 'plan.json').read_text())
    for path, expected in plan['hashes'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected, path
    if args.probe:
        write({'kind': 'cpu_probe', 'passed': True, 'files': len(plan['hashes'])})
        return

    import torch

    assert os.environ['CUDA_VISIBLE_DEVICES'] == '2'
    assert sorted(os.sched_getaffinity(0)) == list(range(16))
    torch.set_num_threads(16)
    prop = torch.cuda.get_device_properties(0)
    assert str(prop.uuid) == 'cbf66259-f4ab-0ede-1811-82037dde5924'
    write({'kind': 'hardware', 'uuid': str(prop.uuid), 'name': prop.name,
           'sm_count': prop.multi_processor_count, 'pid': os.getpid()})
    env = dict(os.environ, CUA_S1_CUDA_LIB=plan['validation_library'])
    started = time.perf_counter()
    command = [plan['kernel_test_binary'], '--ignored', '--nocapture', '--test-threads=1']
    with (ROOT / 'analysis/recurrent-reference.log').open('w') as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=180)
    write({'kind': 'recurrent_reference', 'command': command, 'exit_code': result.returncode,
           'seconds': time.perf_counter() - started})
    assert result.returncode == 0, (ROOT / 'analysis/recurrent-reference.log').read_text()[-2000:]

    libraries = {}
    for name in ['baseline', 'candidate']:
        lib = ctypes.CDLL(plan['libraries'][name])
        lib.cs1_gdn_workspace_floats.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.cs1_gdn_workspace_floats.restype = ctypes.c_size_t
        lib.cs1_gdn_prefill.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3 + [ctypes.c_float, ctypes.c_void_p]
        lib.cs1_gdn_prefill.restype = ctypes.c_int
        libraries[name] = lib
    stream = torch.cuda.current_stream()
    pointer = lambda x: ctypes.c_void_p(x.data_ptr())
    generator = torch.Generator(device='cpu').manual_seed(20261003)
    for t in [107, 936, 3399]:
        def random(shape):
            return (torch.rand(shape, generator=generator) * 2 - 1).to(torch.bfloat16).cuda()
        k = random((t, 16, 128))
        q = (0.8 * k.float() + 0.2 * random(k.shape).float()).to(torch.bfloat16)
        v = random((t, 48, 128))
        g = (random((t, 48)).float() - 1) * 0.01
        beta = (random((t, 48)).float() * 0.5 + 0.5).to(torch.bfloat16)
        outputs = {name: torch.empty_like(v) for name in libraries}
        workspace = torch.empty(libraries['baseline'].cs1_gdn_workspace_floats(t, 48),
                                dtype=torch.float32, device='cuda')

        def forward(name):
            status = libraries[name].cs1_gdn_prefill(pointer(q), pointer(k), pointer(v), pointer(g),
                pointer(beta), pointer(outputs[name]), pointer(workspace), t, 48, 16,
                128 ** -0.5, ctypes.c_void_p(stream.cuda_stream))
            assert status == 0, status
            return outputs[name]

        # Match the production workspace layout; compare only fully written fields.
        fields, offset = {}, 0
        for field, size, dtype in [('u', 48 * ((t + 63) // 64) * 64 * 128 * 2, torch.bfloat16),
                ('w', 48 * ((t + 63) // 64) * 64 * 128 * 2, torch.bfloat16),
                ('qd', 48 * ((t + 63) // 64) * 64 * 128 * 2, torch.bfloat16),
                ('kd', 48 * ((t + 63) // 64) * 64 * 128 * 2, torch.bfloat16),
                ('p', 48 * ((t + 63) // 64) * 64 * 64 * 4, torch.float32),
                ('pb', 48 * ((t + 63) // 64) * 64 * 64 * 2, torch.bfloat16),
                ('decay', 48 * ((t + 63) // 64) * 4, torch.float32)]:
            if field != 'p':
                fields[field] = workspace[offset // 4:(offset + size) // 4].view(dtype)
            offset = (offset + size + 255) // 256 * 256
        reference_fields = {}
        baseline = None
        for name in ['baseline', 'candidate']:
            started = time.perf_counter()
            output = forward(name)
            torch.cuda.synchronize()
            if baseline is None:
                baseline = output.clone()
                reference_fields = {field: part.clone() for field, part in fields.items()}
            differences = {field: {'changed': int((part.view(torch.int16 if part.dtype == torch.bfloat16 else torch.int32) != reference_fields[field].view(torch.int16 if part.dtype == torch.bfloat16 else torch.int32)).count_nonzero()),
                'max_abs': float((part.float() - reference_fields[field].float()).abs().max())}
                for field, part in fields.items()}
            write({'kind': 'workspace_fidelity', 'tokens': t, 'config': name, 'differences': differences})
            assert all(differences[field]['changed'] == 0 for field in fields), differences
            delta = output.float() - baseline.float()
            max_abs = float(delta.abs().max().item())
            relative_l2 = float(torch.linalg.vector_norm(delta) /
                                torch.linalg.vector_norm(baseline.float()).clamp_min(1e-12))
            assert torch.isfinite(output).all().item() and max_abs == 0.0 and relative_l2 == 0.0
            for _ in range(10):
                forward(name)
            torch.cuda.synchronize()
            write({'kind': 'feasibility', 'tokens': t, 'config': name, 'passed': True,
                   'max_abs': max_abs, 'relative_l2': relative_l2,
                   'seconds': time.perf_counter() - started})
        for run, order in [(1, ['baseline', 'candidate']), (2, ['candidate', 'baseline'])]:
            for name in order:
                torch.cuda.synchronize()
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                before = time.perf_counter_ns()
                start.record(stream)
                for _ in range(100):
                    output = forward(name)
                end.record(stream)
                end.synchronize()
                wall_ms = (time.perf_counter_ns() - before) / 1e6 / 100
                delta = output.float() - baseline.float()
                write({'kind': 'measurement', 'tokens': t, 'config': name, 'pass': run,
                       'iterations': 100, 'device_span_ms_per_call': start.elapsed_time(end) / 100,
                       'wall_ms_per_call': wall_ms, 'max_abs': float(delta.abs().max().item())})
        del q, k, v, g, beta, workspace, outputs, baseline, fields, reference_fields
    write({'kind': 'complete', 'pid': os.getpid()})


if __name__ == '__main__':
    main()
