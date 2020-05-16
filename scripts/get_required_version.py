import os
import argparse
import re

SRC_ENVIRON = "ARRUS_SRC_PATH"

def main():
    parser = argparse.ArgumentParser(description="Returns required version for given requirement.")
    parser.add_argument("--source_dir", dest="source_dir",
                        type=str, required=False,
                        default=os.environ.get(SRC_ENVIRON, None))
    parser.add_argument("--requirement", dest="requirement",
                        type=str, required=True)

    args = parser.parse_args()
    source_dir = args.source_dir
    requirement = args.requirement

    cmakelists_path = os.path.join(source_dir, "CMakeLists.txt")
    pattern = r".*find_package.*\(.*%s\s+([0-9\.]+).*" % requirement
    with open(cmakelists_path, "r") as f:
        for line in f:
            results = re.findall(pattern, line)
            if len(results) == 1:
                print(results[0])
                return
            elif len(results) > 1:
                raise ValueError("Found multiple entries for the same requirement in %s" % cmakelists_path)

    raise ValueError("Requirement %s was not found in %s" % (requirement, cmakelists_path))

if __name__ == "__main__":
    main()