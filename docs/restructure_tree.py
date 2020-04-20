import argparse
import os, errno
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", dest="root_dir", required=True)
    parser.add_argument("--language", dest="language", required=True)

    args = parser.parse_args()
    root_dir = args.root_dir
    language = args.language
    print("Restructuring tree '%s' for language '%s'" % (root_dir, language))

    rst_files = Path(root_dir).rglob("*.*.rst")

    other_language_files = []
    src_files = []
    configurable_rst_pattern = re.compile(r".+\..+\.rst$")
    for path in rst_files:
        path = str(path)
        if configurable_rst_pattern.search(path):
            if path.endswith(".%s.rst" % language):
                src_files.append(path)
            else:
                other_language_files.append(path)

    # Remove rst files dedicated for other languages.
    for p in other_language_files:
        os.remove(os.path.join(root_dir, p))

    src_files = [os.path.join(root_dir, path) for path in src_files]
    dst_files = [path.replace(".%s." % language, ".") for path in src_files]

    # To avoid OSError on Windows make sure that dst files does not exist.
    for dst_file in dst_files:
        try:
            os.remove(dst_file)
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise e
    # Rename language source files.
    for src_file, dst_file in zip(src_files, dst_files):
        os.rename(src_file, dst_file)

    print("Restructuring done.")

if __name__ == "__main__":
    main()