"""CleanPass - installer.

Copies this folder to ~/.nuke/plugins/CleanPass and registers it in
~/.nuke/init.py. Safe to run several times. Close Nuke first.

    python install.py            (or double-click install.bat)
"""
import os
import shutil
import sys

NAME = 'CleanPass'
BEGIN = '# --- CleanPass START ---'
END = '# --- CleanPass END ---'
KEEP = ('bin', 'oidn', 'icons', 'tests', 'init.py', 'menu.py', 'cleanpass_ui.py', 'README.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md')


def nuke_dir():
    return os.path.join(os.path.expanduser('~'), '.nuke')


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(here, 'bin')):
        sys.exit('No bin/ folder: build the plugin first (build.bat) or use a release zip.')
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
