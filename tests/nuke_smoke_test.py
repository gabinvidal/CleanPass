"""CleanPass - smoke test.

Run it from a terminal:   "C:\\Program Files\\Nuke17.0v3\\Nuke17.0.exe" -t tests\\nuke_smoke_test.py
or paste it in Nuke's Script Editor. It builds a small synthetic render with
fireflies, a NaN, an inf and a "spec" AOV, runs the node in both modes and
prints PASS / FAIL for each check. Then deflicker, denoise (with and without
stabilisation), light-group AOV denoise and motion vectors on small animated
expressions.
"""
import math

import nuke

FIREFLIES = [(100, 100, 60.0), (40, 150, 25.0), (180, 60, 300.0), (181, 61, 220.0)]  # last two: a cluster
NAN_PX, INF_PX = (60, 60), (150, 200)
SAFE_PX = (20, 20)


def build_source():
    for n in nuke.allNodes():
        if n.name().startswith('cp_test_'):
            nuke.delete(n)
    fmt = '256 256 1 cp_test_fmt'
    nuke.addFormat(fmt)
    const = nuke.nodes.Constant(name='cp_test_const', color=[0.4, 0.35, 0.3, 1], format='cp_test_fmt')
    noise = nuke.nodes.Noise(name='cp_test_noise', inputs=[const], size=3, gain=0.6, octaves=2)
    noise['opacity'].setValue(0.15)
    cond = ' || '.join('(x==%d && y==%d)' % (x, y) for x, y, _ in FIREFLIES)
    val = ' : '.join('(x==%d && y==%d) ? %g' % (x, y, v) for x, y, v in FIREFLIES)
    add = nuke.nodes.AddChannels(name='cp_test_addspec', inputs=[noise], channels='spec')
    expr = nuke.nodes.Expression(name='cp_test_expr', inputs=[add])
    # rgb: noisy background + fireflies + one NaN + one inf
    for i, c in enumerate('rgb'):
        expr['expr%d' % i].setValue('(%s) ? (%s : 0) : (x==%d && y==%d) ? sqrt(-1) : (x==%d && y==%d) ? 1/0 : %s'
                                    % (cond, val, NAN_PX[0], NAN_PX[1], INF_PX[0], INF_PX[1], c))
    # spec AOV: carries the fireflies, small value elsewhere
    expr2 = nuke.nodes.Expression(name='cp_test_spec', inputs=[expr])
    expr2['channel0'].setValue('spec')
    for i in range(3):
        expr2['expr%d' % i].setValue('(%s) ? (%s : 0) - 0.3 : 0.02' % (cond, val))
    return expr2


def finite(v):
    return not (math.isnan(v) or math.isinf(v))


def run():
    src = build_source()
    results = []

    def check(name, ok, info=''):
        results.append(ok)
        print('%s  %s %s' % ('PASS' if ok else 'FAIL', name, info))

    for mode, label in ((1, 'statistical'), (0, 'OIDN')):
        node = nuke.nodes.CleanPass(name='cp_test_node_%d' % mode, inputs=[src])
        node['firefly_enable'].setValue(True)
        node['reference'].setValue(mode)
        node['aov_layers'].setValue('spec')
        node['output_matte'].setValue(True)
        print('\n--- %s ---' % label)
        for x, y, v in FIREFLIES:
            out = node.sample('red', x + 0.5, y + 0.5)
            check('firefly %d,%d removed' % (x, y), out < 2.0, '(%.2f -> %.3f)' % (v, out))
        a = src.sample('red', SAFE_PX[0] + 0.5, SAFE_PX[1] + 0.5)
        b = node.sample('red', SAFE_PX[0] + 0.5, SAFE_PX[1] + 0.5)
        check('clean pixel untouched', a == b, '(%r vs %r)' % (a, b))
        for name, (x, y) in (('NaN', NAN_PX), ('inf', INF_PX)):
            out = node.sample('red', x + 0.5, y + 0.5)
            check('%s repaired' % name, finite(out), '(%r)' % out)
        m = node.sample('firefly.matte', FIREFLIES[0][0] + 0.5, FIREFLIES[0][1] + 0.5)
        check('matte channel', m > 0.9, '(%.3f)' % m)
        x, y, v = FIREFLIES[0]
        spec = node.sample('spec.red', x + 0.5, y + 0.5)
        rgb = node.sample('red', x + 0.5, y + 0.5)
        check('spec AOV corrected', spec < 1.0, '(%.2f -> %.3f)' % (v - 0.3, spec))
        check('AOV no negative', spec >= -1e-4)
        nuke.delete(node)

    # ---- deflicker / denoise on per-frame random noise -------------------
    print('\n--- deflicker / denoise ---')
    const = nuke.nodes.Constant(name='cp_test_flat', color=[0.5, 0.5, 0.5, 1], format='cp_test_fmt')
    # Per-frame random grain. Nuke caches an Expression by its hash, which does
    # not change with `frame`, so a Noise with an animated z offset is used:
    # fine grain, a different pattern on every frame, like render noise.
    noisy = nuke.nodes.Noise(name='cp_test_flicker', inputs=[const], size=1.2, octaves=1, gain=0.5)
    noisy['zoffset'].setExpression('frame * 13.7')
    expr_src = nuke.nodes.NoOp(name='cp_test_noop')
    nuke.root()['first_frame'].setValue(1)
    nuke.root()['last_frame'].setValue(20)
    nuke.frame(10)

    def spatial_std(node):
        vals = [node.sample('red', 60.5 + 3 * i, 60.5 + 3 * j) for i in range(12) for j in range(12)]
        m = sum(vals) / len(vals)
        return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

    # Pure noise with no albedo / normal is the hardest case for the motion
    # search (it partly fits the noise), so only a clear reduction is checked.
    base = spatial_std(noisy)
    df = nuke.nodes.CleanPass(name='cp_test_df', inputs=[noisy])
    df['firefly_enable'].setValue(False)
    df['deflicker_enable'].setValue(True)
    s = spatial_std(df)
    check('deflicker reduces frame-to-frame noise', s < 0.95 * base, '(std %.4f -> %.4f)' % (base, s))
    df['df_frames'].setValue(1)
    s2 = spatial_std(df)
    check('deflicker 2 frames is stronger', s2 < 0.95 * s, '(std %.4f)' % s2)
    dn = nuke.nodes.CleanPass(name='cp_test_dn', inputs=[noisy])
    dn['firefly_enable'].setValue(False)
    dn['denoise_enable'].setValue(True)
    s3 = spatial_std(dn)
    check('denoise reduces noise', s3 < 0.5 * base, '(std %.4f -> %.4f)' % (base, s3))
    dn['stabilize'].setValue(False)
    s4 = spatial_std(dn)
    dn['stabilize'].setValue(True)
    check('denoise without stabilisation works too', s4 < 0.5 * base, '(std %.4f)' % s4)

    # ---- light groups: denoised AOVs keep adding up to the beauty --------
    print('\n--- AOVs ---')
    for layer in ('lgA', 'lgB'):
        nuke.Layer(layer, ['%s.red' % layer, '%s.green' % layer, '%s.blue' % layer])
    split = nuke.nodes.Expression(name='cp_test_split', inputs=[noisy])
    split['channel0'].setValue('lgA')
    for i, c in enumerate('rgb'):
        split['expr%d' % i].setValue('0.6 * %s' % c)
    split2 = nuke.nodes.Expression(name='cp_test_split2', inputs=[split])
    split2['channel0'].setValue('lgB')
    for i, c in enumerate('rgb'):
        split2['expr%d' % i].setValue('%s - lgA.%s' % (c, ['red', 'green', 'blue'][i]))
    print('(input AOVs: lgA %.4f, lgB %.4f, beauty %.4f)' % (split2.sample('lgA.red', 40.5, 90.5),
                                                          split2.sample('lgB.red', 40.5, 90.5), split2.sample('red', 40.5, 90.5)))
    lg = nuke.nodes.CleanPass(name='cp_test_lg', inputs=[split2])
    lg['firefly_enable'].setValue(False)
    lg['denoise_enable'].setValue(True)
    lg['stabilize'].setValue(False)
    lg['aov_layers'].setValue('lg*')
    worst, info = 0.0, ''
    for i in range(6):
        x, y = 40.5 + 30 * i, 90.5 + 11 * i
        a, b, r = lg.sample('lgA.red', x, y), lg.sample('lgB.red', x, y), lg.sample('red', x, y)
        if abs(a + b - r) >= worst:
            worst, info = abs(a + b - r), 'at %g,%g: %.4f + %.4f vs %.4f' % (x, y, a, b, r)
    check('AOVs add up to the denoised beauty', worst < 1e-3, '(max error %.2e %s)' % (worst, info))
    vals = [lg.sample('lgA.red', 60.5 + 3 * i, 60.5 + 3 * j) for i in range(12) for j in range(12)]
    m = sum(vals) / len(vals)
    sa = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5
    check('AOVs are denoised', sa < 0.5 * 0.6 * base, '(std %.4f -> %.4f)' % (0.6 * base, sa))

    # ---- motion vectors: a fast move (beyond the motion search) ----------
    print('\n--- motion vectors ---')
    nuke.Layer('motionvector', ['motionvector.red', 'motionvector.green', 'motionvector.blue'])
    # a non-periodic texture moving 7 px per frame (beyond the default search of 4)
    tex = nuke.nodes.Noise(name='cp_test_tex', inputs=[const], size=9, gain=0.7, octaves=3)
    tex['opacity'].setValue(0.6)
    xf = nuke.nodes.Transform(name='cp_test_xf', inputs=[tex])
    xf['translate'].setExpression('frame * 7', 0)
    clean = nuke.nodes.Expression(name='cp_test_clean', inputs=[xf])
    clean['expr3'].setValue('a')
    mov = nuke.nodes.Expression(name='cp_test_move', inputs=[xf])
    for i, c in enumerate('rgb'):
        mov['expr%d' % i].setValue('%s + 0.3 * (random(x * 1.37, y * 2.11, frame * 3.7) - 0.5)' % c)
    mvx = nuke.nodes.Expression(name='cp_test_mvx', inputs=[mov])
    mvx['channel0'].setValue('motionvector')
    mvx['expr0'].setValue('7')
    mvx['expr1'].setValue('0')
    mvx['expr2'].setValue('0')
    movT = nuke.nodes.TimeOffset(name='cp_test_moveoff', inputs=[mvx])
    movT['time_offset'].setValue(-1)
    cleanT = nuke.nodes.TimeOffset(name='cp_test_cleanoff', inputs=[clean])
    cleanT['time_offset'].setValue(-1)

    def err_vs_clean(node):
        e = [node.sample('red', 40.5 + 11 * i, 40.5 + 13 * j) - cleanT.sample('red', 40.5 + 11 * i, 40.5 + 13 * j)
             for i in range(15) for j in range(13)]
        return (sum(v * v for v in e) / len(e)) ** 0.5

    e0 = err_vs_clean(movT)
    mv = nuke.nodes.CleanPass(name='cp_test_mv', inputs=[movT])
    mv['firefly_enable'].setValue(False)
    mv['deflicker_enable'].setValue(True)
    e1 = err_vs_clean(mv)
    mv['use_vectors'].setValue(True)
    mv['motion_layer'].fromScript('motionvector')
    e2 = err_vs_clean(mv)
    check('motion vectors follow a fast move', e2 < 0.9 * e0 and e2 <= 1.15 * e1,
          '(error %.4f, deflicker without vectors %.4f, with %.4f)' % (e0, e1, e2))

    # ---- separate inputs ----------------------------------------------------
    print('\n--- separate inputs ---')
    check('inputs: src, albedo, normal, motion, mask', mv.maxInputs() == 5, '(%d)' % mv.maxInputs())
    # the same vectors, from their own input (in rgb) instead of the src layer
    mvrgb = nuke.nodes.Shuffle2(name='cp_test_mvrgb', inputs=[movT])
    mvrgb['in1'].setValue('motionvector')
    mv2 = nuke.nodes.CleanPass(name='cp_test_mv2', inputs=[movT])
    mv2['deflicker_enable'].setValue(True)
    mv2['use_vectors'].setValue(True)
    mv2.setInput(3, mvrgb)
    e3 = err_vs_clean(mv2)
    check('motion vectors from the motion input', abs(e3 - e2) <= 0.02 * e2 + 1e-6, '(%.4f vs layer %.4f)' % (e3, e2))
    mv2['use_depth'].setValue(True)
    mv2.forceValidate()
    check('depth adds an input before the mask', mv2.maxInputs() == 6, '(%d)' % mv2.maxInputs())
    # moving the connections when the layout changes (what the panel does)
    import cleanpass_ui
    mk = nuke.nodes.Constant(name='cp_test_mask', color=[1, 1, 1, 1], format='cp_test_fmt')
    mv2.setInput(5, mk)
    mv2['use_vectors'].setValue(False)
    cleanpass_ui.rewire(mv2, [0, 1, 2, 3, 4, 5], [0, 1, 2, 4, 5])
    names = [x.name() if x else None for x in [mv2.input(i) for i in range(mv2.maxInputs())]]
    check('rewire keeps depth and mask', mv2.maxInputs() == 5 and mv2.input(4) == mk and mv2.input(3) is None,
          '(inputs %s)' % names)
    mv2['use_vectors'].setValue(True)
    cleanpass_ui.rewire(mv2, [0, 1, 2, 4, 5], [0, 1, 2, 3, 4, 5])
    mv2.setInput(3, mvrgb)
    names = [x.name() if x else None for x in [mv2.input(i) for i in range(mv2.maxInputs())]]
    check('rewire back', mv2.input(5) == mk and mv2.input(3) == mvrgb, '(inputs %s)' % names)
    # a copy / paste keeps the 6 inputs where they are
    for n in nuke.allNodes():
        n.setSelected(n.name() in ('cp_test_mv2', 'cp_test_mvrgb', 'cp_test_mask', 'cp_test_moveoff'))
    import os, tempfile
    nk = os.path.join(tempfile.gettempdir(), 'cleanpass_smoke_copy.nk')
    nuke.nodeCopy(nk)
    nuke.nodePaste(nk)
    pasted_nodes = list(nuke.selectedNodes())
    p2 = [n for n in pasted_nodes if n.Class() == 'CleanPass']
    if p2:
        p2[0].forceValidate()  # Nuke reads the number of inputs lazily
    ok = bool(p2) and p2[0].maxInputs() == 6 and p2[0].input(3) is not None and p2[0].input(5) is not None and \
        p2[0].input(3).Class() == 'Shuffle2' and p2[0].input(5).Class() == 'Constant'
    check('copy / paste keeps the inputs', ok, '(%s)' % ([x.Class() if x else None for x in [p2[0].input(i) for i in range(p2[0].maxInputs())]] if p2 else 'no node'))
    # albedo / normal from their own inputs, with the stabilisation (it needs their t-1 / t+1)
    alb = nuke.nodes.Constant(name='cp_test_alb', color=[0.5, 0.5, 0.5, 1], format='cp_test_fmt')
    nrm = nuke.nodes.Constant(name='cp_test_nrm', color=[0, 0, 1, 1], format='cp_test_fmt')
    sg = nuke.nodes.CleanPass(name='cp_test_sep', inputs=[noisy])
    sg['denoise_enable'].setValue(True)
    sg.setInput(1, alb)
    sg.setInput(2, nrm)
    s_sep = spatial_std(sg)
    check('denoise + stabilisation with albedo / normal inputs', finite(s_sep) and s_sep < 0.5 * base,
          '(std %.4f -> %.4f)' % (base, s_sep))

    for n in pasted_nodes + [sg, alb, nrm, mk, mv2, mvrgb, mv, movT, cleanT, mvx, mov, clean, xf, tex, lg, split2,
                             split, df, dn, noisy, expr_src, const]:
        nuke.delete(n)

    print('\n%d / %d checks passed' % (sum(results), len(results)))
    return all(results)


if __name__ == '__main__':
    run()
