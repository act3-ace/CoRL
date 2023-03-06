"""Generate the code reference pages and navigation."""
import mkdocs_gen_files

from pathlib import Path
from pprint import pprint
import importlib
import os

import mkdocs_gen_files

import string
import yaml
from pathlib import Path
import os
local_package_def_file = 'mkdocs-packages-docs.yml'

# List of defualt modules to load for auto code documentation in mkdocs
default_modules = yaml.safe_load(Path(local_package_def_file).read_text())["default_modules"] if os.path.exists(local_package_def_file) else ["corl"]

nav = mkdocs_gen_files.Nav()

def process_path(module, m_file, path):
    module_path = path.relative_to(m_file).with_suffix("")
    doc_path = path.relative_to(m_file).with_suffix(".md")
    full_doc_path = Path(f"reference/{module}", doc_path)

    parts = list(module_path.parts)
    parts[-1] = f"{parts[-1]}.py"
    nav[parts] = doc_path
    return module_path,full_doc_path

def dict_to_file(dict_out, fd, path="", spaces = 2):
    """
    Creates the directory structure + names needed to support the MKDOCS
    readthedocs theme.
    """
    for k, v in dict_out.items():
        if isinstance(v, dict) and v:
            fd.write("  " * spaces + f"- \"{k}\":\n")
            dict_to_file(v, fd, path + "/" + k, spaces+1)
        else:
            file_name = string.capwords(k.replace("_"," ").replace(".md",""))
            fd.write("  " * spaces + f"- \"{file_name}\": reference{path}/{k}\n")

def posix_paths_to_dict(paths):
    """
    Converts the list of filename paths to dictionary structure
    """
    path_dict = {}
    for path in paths:
        parts = path.parts
        d = path_dict
        for part in parts[1:]:
            if part not in d:
                d[part] = {}
            d = d[part]
    return path_dict

def gen_files(module: str, temp, m_file) -> None:
    for path in sorted(Path(m_file).glob("**/*.py")):
        module_path, full_doc_path = process_path(module, m_file, path)

        if "configuration_generators" not in str(full_doc_path):
            with mkdocs_gen_files.open(full_doc_path, "w") as fd:
                ident = module + "." + ".".join(module_path.parts)
                print("::: " + ident, file=fd)

            temp.append(full_doc_path)

            mkdocs_gen_files.set_edit_path(full_doc_path, path)


temp = []

for item in default_modules:
    try:
        # The file gets executed upon import, as expected.
        module = importlib.import_module(item)
        gen_files(item, temp, os.path.dirname(module.__file__))
    except:
        print("failed to import " + item)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
