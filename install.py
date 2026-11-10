"""CleanPass - installer.

Copies this folder to ~/.nuke/plugins/CleanPass and registers it in
~/.nuke/init.py. Safe to run several times. Close Nuke first.

The Nukepedia package leaves out the Open Image Denoise runtime (about 45 MB,
too large for the site): the first install downloads it from the GitHub
release and checks it against the hashes below.

    python install.py            (or double-click install.bat)
"""
import hashlib
import os
import shutil
import sys
import tempfile
import zipfile

NAME = 'CleanPass'
BEGIN = '# --- CleanPass START ---'
END = '# --- CleanPass END ---'
KEEP = ('bin', 'oidn', 'icons', 'tests', 'init.py', 'menu.py', 'cleanpass_ui.py', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md')


RUNTIME_URL = 'https://github.com/gabinvidal/CleanPass/releases/download/v1.0/CleanPass_v1.0.zip'
RUNTIME = {  # oidn/bin/<file>: sha256
    'OIDN_CleanPass25.dll': 'eb7cef2f974087e05b7825d6a9ae2915158ec812e910e8b06c442c226a9c5d81',
    'OIDN_CleanPass25_core.dll': '0b5cf18a97c8124914f9dbbce6886d4c286b69036e6d4700cef12e5bbd8ca0e0',
    'OIDN_CleanPass25_device_cpu.dll': '309b38c72faa3b107a8f5f16c2e2755932573d9991f9fc5bdd53f3871352ff90',
    'OIDN_CleanPass25_device_cuda.dll': '9a26cfbae0c94c93260a76d98b98b8e84382195f58e355631e3f9f280df218e3',
    'tbbCP.dll': '84e03a04e24eaa3379614f912632b8a78637e713a639c794633a589b90fae91e',
    'MSVCP_CP.dll': '7c26614e1d733892c2deac7e245ce115504b1d80592dd0a01b08e3e5a55f89ca',
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ensure_runtime(here):
    """Downloads the OIDN runtime into oidn/bin when it is not there (Nukepedia package)."""
    oidn_bin = os.path.join(here, 'oidn', 'bin')
    if all(os.path.exists(os.path.join(oidn_bin, f)) for f in RUNTIME):
        return
    print('Downloading the Open Image Denoise runtime (about 50 MB) from the GitHub release...')
    import urllib.request
    os.makedirs(oidn_bin, exist_ok=True)
    tmp = os.path.join(tempfile.gettempdir(), 'CleanPass_runtime.zip')
    try:
        urllib.request.urlretrieve(RUNTIME_URL, tmp)
        with zipfile.ZipFile(tmp) as z:
            for name, digest in RUNTIME.items():
                data = z.read('CleanPass/oidn/bin/' + name)
                if hashlib.sha256(data).hexdigest() != digest:
                    sys.exit('The downloaded %s does not match the expected file: install stopped.' % name)
                with open(os.path.join(oidn_bin, name), 'wb') as fh:
                    fh.write(data)
            for lic in ('LICENSE.txt', 'third-party-programs.txt', 'third-party-programs-oneTBB.txt',
                        'third-party-programs-DPCPP.txt'):
                try:
                    with open(os.path.join(here, 'oidn', lic), 'wb') as fh:
                        fh.write(z.read('CleanPass/oidn/' + lic))
                except KeyError:
                    pass
    except Exception as exc:
        sys.exit('Could not download the runtime (%s).\nDownload the full package instead: %s' % (exc, RUNTIME_URL))
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass
    print('Runtime OK.')


def nuke_dir():
    return os.path.join(os.path.expanduser('~'), '.nuke')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(here, 'bin')):
        sys.exit('No bin/ folder: build the plugin first (build.bat) or use a release zip.')
    ensure_runtime(here)
    dest = os.path.join(nuke_dir(), 'plugins', NAME)
    if os.path.normcase(os.path.abspath(dest)) != os.path.normcase(here):
        os.makedirs(dest, exist_ok=True)
        for item in KEEP:
            src = os.path.join(here, item)
            if not os.path.exists(src):
                continue
            dst = os.path.join(dest, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
        print('Copied to', dest)

    init_py = os.path.join(nuke_dir(), 'init.py')
    text = open(init_py, encoding='utf-8').read() if os.path.exists(init_py) else ''
    if BEGIN in text and END in text:
        start, stop = text.index(BEGIN), text.index(END) + len(END)
        text = (text[:start] + text[stop:]).rstrip('\n') + '\n'
    # relative to ~/.nuke, like Nuke's own examples (no __file__ needed)
    block = "%s\nimport nuke\nnuke.pluginAddPath('./plugins/%s')\n%s\n" % (BEGIN, NAME, END)
    with open(init_py, 'w', encoding='utf-8') as fh:
        fh.write(text + ('\n' if text and not text.endswith('\n') else '') + block)
    print('Registered in', init_py)
    print('Done - start Nuke and Tab-search "CleanPass".')


if __name__ == '__main__':
    main()
