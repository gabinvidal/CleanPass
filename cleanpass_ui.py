"""CleanPass - small Python helpers called by the node.

rewire(): the motion vector and depth inputs only exist while their option is
on (Setup tab), so turning one on or off changes the input numbers. The node
calls this to move the existing connections (mask included, which stays last)
to their new numbers. Layouts are lists of input kinds:
0 src, 1 albedo, 2 normal, 3 motion, 4 depth, 5 mask.
"""
import nuke


def rewire(node, old, new):
    if node is None:
        return
    conn = {}
    for i, kind in enumerate(old):
        try:
            conn[kind] = node.input(i)
        except Exception:
            conn[kind] = None
    # let Nuke pick up the new number of inputs before connecting them
    try:
        node.forceValidate()
    except Exception:
        pass
    for j, kind in enumerate(new):
        try:
            node.setInput(j, conn.get(kind))
        except Exception as exc:
            print('[CleanPass] could not reconnect input %d: %s' % (j, exc))
