# Core validation (developer tests)

The algorithm in `src/cp_core.*` has no Nuke dependency, so it is validated
outside Nuke on real path-traced renders.

1. `pip install mitsuba numpy scipy pillow`, then `python render_scenes.py`
   (noisy frames, light-group AOVs, albedo / normal, 1536 spp reference) and
   `OUTDIR=data256 SPP=256 python render_scenes.py noisy`.
2. `OIDN=/path/to/oidn-2.5.1.x86_64.linux sh build_linux.sh`
3. `./asan_main` (AddressSanitizer / UBSan fuzz run, NaN / inf, all options)
4. `python synth.py` - 500 fireflies injected into the converged render and its
   AOVs (exact ground truth); `python evaluate.py` - real low-spp fireflies.
5. Deflicker / full pipeline: `python render_seq.py noisy` and `python render_seq.py ref`
   (moving-camera sequence + per-frame reference), then
   `python gt_eval.py '{"firefly":0,"deflicker":1}'` (error and flicker against the reference).
   `./asan_pipe` fuzzes the whole pipeline.

## Temporal sequences (render_seq2 / eval_seq2, render_seq3)

6. `python render_seq2.py aux|anim|fixed|ref` then `python mv.py`: a moving camera and
   a moving ball, with exact motion vectors and depth, a variance pass, two light
   groups, animated and fixed seed versions and a 768 spp reference per frame (in `d/`).
   `eval_seq2.py` loads them and scores err / flicker against the reference
   through `libpipe4.so` (`pipe4.py`).
7. `python render_seq3.py vol|hair|dof|mblur` then `... ref`: the same shot with fog,
   sub-pixel hair, strong depth of field or fast motion blur, each with a reference.
8. `reuse_test.cpp`: checks that a frame computed from the work its neighbour prepared
   (when rendering in order) is identical to the same frame computed on its own.
