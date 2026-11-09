"""CleanPass - removes the init.py entry (and optionally the folder)."""
import os
import shutil

from install import BEGIN, END, NAME, nuke_dir


def main():
    init_py = os.path.join(nuke_dir(), 'init.py')
    if os.path.exists(init_py):
        text = open(init_py, encoding='utf-8').read()
        if BEGIN in text and END in text:
            start, stop = text.index(BEGIN), text.index(END) + len(END)
            with open(init_py, 'w', encoding='utf-8') as fh:
                fh.write((text[:start] + text[stop:]).rstrip('\n') + '\n')
            print('Removed the entry from', init_py)
    dest = os.path.join(nuke_dir(), 'plugins', NAME)
    if os.path.isdir(dest) and input('Delete %s ? [y/N] ' % dest).strip().lower() == 'y':
        shutil.rmtree(dest)
        print('Deleted', dest)


if __name__ == '__main__':
    main()
