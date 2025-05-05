from .converter import *
from .tools import *
from .graphs import *
from .station_data_pull import *
from .headers import *

import pathlib
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def _get_version():
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


__version__ = _get_version()
