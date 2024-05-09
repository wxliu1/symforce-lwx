# mypy: allow-untyped-defs
# aclint: py3
"""
Parse lcm defintion files and generate bindings in different languages.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import typing as T

from skymarshal.language_plugin import SkymarshalLanguage  # pylint: disable=unused-import
from skymarshal.package_map import parse_lcmtypes


def parse_args(
    languages: T.Sequence[T.Type[SkymarshalLanguage]], args: T.Optional[T.Sequence[str]] = None
) -> argparse.Namespace:
    """Parse the argument list and return an options object."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_path", nargs="+")
    parser.add_argument("--debug-tokens", action="store_true")
    parser.add_argument("--print-def", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--excluded-path",
        nargs="+",
        action="append",
        help="Path to ignore when building package map",
    )

    parser.add_argument(
        "--package-prefix",
        default="",
        help="Add this package name as a prefix to the declared package on types",
    )

    parser.add_argument(
        "--no-source-paths",
        action="store_true",
        help="Don't include the LCM source path in generated files",
    )

    for lang in languages:
        lang.add_args(parser)

    if args:
        # Caller has provided the argument strings explicitly.
        options = parser.parse_args(args)
    else:
        # Use the command-line args.

        # NOTE(will): Support argfiles, e.g. @argfile as the first argument. argparse supports this
        # directly with fromfile_prefix_chars="@", but this was causing issues for any arguments
        # that started with @, even if those arguments were inside the argfile. This approach is
        # much narrower in only supporting an argfile as the first argument.
        if len(sys.argv) >= 2 and sys.argv[1].startswith("@"):
            with open(sys.argv[1][1:]) as argfile:
                args = shlex.split(argfile.read())
            options = parser.parse_args(args + sys.argv[2:])
        else:
            options = parser.parse_args()

    return options


def main(
    languages: T.Sequence[T.Type[SkymarshalLanguage]],
    args: T.Sequence[str] = None,
    print_generated: bool = True,
) -> None:
    """The primary executable for generating lcmtypes code from struct definitions.
    This is mostly an example of how to use the generator."""

    options = parse_args(languages, args)
    package_map = parse_lcmtypes(
        options.source_path,
        verbose=options.verbose,
        print_debug_tokens=options.debug_tokens,
        cache_parser=True,
        include_source_paths=not options.no_source_paths,
    )

    packages = list(package_map.values())

    if options.print_def:
        print(packages)

    files = {}

    for lang in languages:
        files.update(lang.create_files(packages, options))

    # Write any generated files that have changed.
    for filename, content in files.items():
        dirname = os.path.dirname(filename)
        if bool(dirname) and not os.path.exists(dirname):
            os.makedirs(dirname, 0o755)
        with open(filename, mode="wb") as output_file:
            if isinstance(content, str):
                output_file.write(content.encode("utf-8"))
            else:
                output_file.write(content)

    if print_generated:
        print(f"Generated {len(files)} files")
