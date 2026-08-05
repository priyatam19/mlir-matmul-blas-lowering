#!/usr/bin/env python3
"""Extract textual PTX assemblies from MLIR gpu.binary objects."""

import argparse
import pathlib
import re


def decode_mlir_string(value: str) -> str:
    output = bytearray()
    index = 0
    while index < len(value):
        if value[index] == "\\" and index + 2 < len(value):
            encoded = value[index + 1 : index + 3]
            if all(character in "0123456789abcdefABCDEF" for character in encoded):
                output.append(int(encoded, 16))
                index += 3
                continue
        output.extend(value[index].encode("utf-8"))
        index += 1
    return output.decode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=pathlib.Path)
    parser.add_argument("output_dir", type=pathlib.Path)
    args = parser.parse_args()
    text = args.input.read_text(encoding="utf-8")
    assemblies = re.findall(r'assembly = "((?:\\[0-9A-Fa-f]{2}|[^"\\])*)"', text)
    if not assemblies:
        raise SystemExit(f"no gpu.binary PTX assemblies found in {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, assembly in enumerate(assemblies):
        (args.output_dir / f"kernel_{index}.ptx").write_text(
            decode_mlir_string(assembly), encoding="utf-8", newline="\n"
        )
    print(f"extracted {len(assemblies)} PTX module(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
