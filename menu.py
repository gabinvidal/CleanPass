"""CleanPass - menu entry."""
import os

import nuke

_ICON = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'icons', 'CleanPass.png').replace('\\', '/')

if nuke.GUI:
    nuke.menu('Nodes').addCommand('Filter/CleanPass', "nuke.createNode('CleanPass')", icon=_ICON)
