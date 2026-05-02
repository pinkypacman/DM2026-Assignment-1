"""Backward-compat shim — Problem 5 was refactored into the `problem5/` package.

Use `import problem5 as q5` going forward; this module re-exports the public
API of `problem5` so existing callers keep working.  Run with
`python -m problem5` from the asg2/ directory for the CLI.
"""

from problem5 import *  # noqa: F401,F403
from problem5 import __all__  # noqa: F401


def main():
    from problem5.__main__ import main as _main
    _main()


if __name__ == "__main__":
    main()
