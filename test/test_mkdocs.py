import pathlib

import pytest
from mktestdocs import check_md_file


# Note the use of `str`, makes for pretty output
@pytest.mark.parametrize("fpath", pathlib.Path("docs").glob("**/*.md"), ids=str)
def test_files_good(fpath):
    """This will take any codeblock that starts with ```python and run it, checking
    for any errors that might happen. This means that if your docs contain asserts,
    that you get some unit-tests for free!

    Args:
        fpath (_type_): _description_
    """

    # TODO fix evaluation file
    if "user_guide.md" not in str(fpath):
        check_md_file(fpath=fpath)
