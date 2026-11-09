"""CleanPass - loads the compiled node for the running Nuke version.

The node is a compiled NDK plugin, so there is one build per Nuke major version
in bin/<major>/CleanPass.dll. The Intel Open Image Denoise runtime it
needs is bundled once in oidn/bin/.
"""
import os

import nuke

_HERE = os.path.dirname(os.path.abspath(__file__))
# Private copy of the Open Image Denoise runtime (see build.ps1): own file
# names, its own TBB and C++ runtime, so it can neither clash with Nuke's
# libraries nor with another plugin's OIDN. The device modules
# (OIDN_CleanPass25_device_cpu / _cuda) are found by the core next to it.
_OIDN_DLLS = ('MSVCP_CP.dll', 'tbbCP.dll', 'OIDN_CleanPass25_core.dll', 'OIDN_CleanPass25.dll')


def _preload_oidn():
    """Load the bundled OIDN DLLs so CleanPass.dll can resolve them."""
    if os.name != 'nt':
        return
    import ctypes
    k32 = ctypes.WinDLL('kernel32', use_last_error=True)
    k32.LoadLibraryW.restype = ctypes.c_void_p
    k32.LoadLibraryW.argtypes = [ctypes.c_wchar_p]
    oidn_bin = os.path.join(_HERE, 'oidn', 'bin')
    for name in _OIDN_DLLS:
        path = os.path.join(oidn_bin, name)
        if not os.path.exists(path):
            print('[CleanPass] missing %s - run build.bat (see README).' % path)
            return
        if not k32.LoadLibraryW(path):
            print('[CleanPass] could not load %s (error %d)' % (path, ctypes.get_last_error()))
            return


def _load():
    major = str(nuke.NUKE_VERSION_MAJOR)
    bin_dir = os.path.join(_HERE, 'bin', major)
    ext = {'nt': '.dll'}.get(os.name, '.so')
    if not os.path.exists(os.path.join(bin_dir, 'CleanPass' + ext)):
        print('[CleanPass] no build for Nuke %s in %s - run build.bat (see README).' % (major, bin_dir))
        return
    _preload_oidn()
    nuke.pluginAddPath(bin_dir)
    try:
        nuke.load('CleanPass')
    except RuntimeError as exc:
        print('[CleanPass] failed to load: %s' % exc)


_load()
