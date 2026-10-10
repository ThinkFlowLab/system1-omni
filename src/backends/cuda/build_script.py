#!/usr/bin/env python3
"""Read a backend build script as text and report what it declares.

``contract.md`` requires ``build.sh`` to accept an output directory and a compute
capability, and to build exactly the library the manifest names for exactly the
architectures it names. Checking that means reading the script — never running
it: CI must not execute arbitrary repository code to decide whether a manifest is
honest.

The parsing is shell-shaped but deliberately shallow. It evaluates the two forms
a build script actually uses (``${NAME:-default}`` and ``$NAME``, including
nesting) and ignores comments, without pretending to be a shell.
"""

from __future__ import annotations

import os
import re

# `sm_90a` / `compute_89` / `arch=compute_90` in a build script. Only used to
# catch a build script that hard-codes one architecture while the manifest
# claims more; it is not a substitute for reading the script.
# A `-gencode` flag and its value: `-gencode "arch=compute_89,code=sm_89"`, or with
# `=`. This is the flag that decides what nvcc actually builds, which is not
# necessarily what an `arch=` variable says.
_GENCODE = re.compile(r"-gencode[=\s]+(\"[^\"]*\"|\S+)")

_ARCH_IN_SCRIPT = re.compile(r"\b(?:sm|compute)_(\d+)[af]?\b")
_BUILD_SCRIPT_ARCH_FIELD = re.compile(r"\b(?:CUDA_COMPUTE_CAP|ARCH|arch)\b")
_SCRIPT_RESOLVED = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$")

_TOKEN = re.compile(r"\$\{[^{}]*\{[^{}]*\}[^{}]*\}|\$\{[^{}]*\}|\$[A-Za-z_]\w*|\$\d")

def _strip_comment(line):
    """Remove a shell comment, respecting quotes.

    `# defaults to CUDA_COMPUTE_CAP, else 89.` must not be read as an assignment
    or as a compute capability, so comments are removed before any scanning.
    """
    out = []
    quote = ""
    for index, char in enumerate(line):
        if quote:
            if char == quote:
                quote = ""
            out.append(char)
            continue
        if char in ("'", '"'):
            quote = char
            out.append(char)
            continue
        if char == "#" and (index == 0 or line[index - 1].isspace()):
            break
        out.append(char)
    return "".join(out)


def _balanced_parameter(value):
    """Return (name, default) for a value that is exactly one ``${...}``.

    Shell parameter expansions nest (``${2:-${CUDA_COMPUTE_CAP:-89}}``), so the
    closing brace has to be matched by counting rather than by a regex, which
    would stop at the inner brace.
    """
    if not value.startswith("${"):
        return None
    depth = 0
    for index, char in enumerate(value):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                if index != len(value) - 1:
                    return None  # trailing text, not a lone parameter expansion
                interior = value[2:index]
                name, separator, default = interior.partition(":-")
                return name, (default if separator else "")
    return None


def _resolve_token(token, script_env, depth=0):
    """Return (value, known) for one shell token inside a larger string.

    ``known`` is False when the token depends on something CI cannot know, such
    as a positional parameter the release process supplies. An unknown token must
    not collapse to an empty string: 'we cannot tell' and 'the value is empty'
    have to stay distinguishable, or an unexpanded path would be read as a
    filename.
    """
    if depth > 8:
        return "", False
    value = token.strip().strip('"').strip("'")
    parameter = _balanced_parameter(value)
    if parameter is not None:
        name, default = parameter
        if name.isdigit() or ":" in name:
            if default:
                return _resolve_token(default, script_env, depth + 1)
            return "", False
        assigned = script_env.get(name, "").strip()
        if not assigned:
            return _resolve_token(default, script_env, depth + 1) if default else ("", False)
        return _resolve_token(assigned, script_env, depth + 1)
    for name, assigned in script_env.items():
        if value == "$" + name:
            return _resolve_token(assigned, script_env, depth + 1)
    if value.startswith("$"):
        return "", False
    return value, True


# A shell expansion inside a larger word. Alternation order matters: the nested
# form has to be tried before the flat form, and the flat form must stop at the
# first `}`. `$` is only a parameter start when followed by a name, `{` or a
# digit, so the literal prefix in `lib_${arch}.so` is not swallowed by `\$\d`.
def _expand_string(text, script_env):
    """Substitute every shell token in ``text``, keeping the literal tail.

    ``"$out/libqwen3_5_cuda.so"`` -> ``libqwen3_5_cuda.so``: the prefix is
    unknowable but the filename is not, and the filename is what the manifest is
    checked against.
    """
    out = []
    position = 0
    for match in _TOKEN.finditer(text):
        out.append(text[position:match.start()])
        value, _known = _resolve_token(match.group(0), script_env)
        out.append(value)
        position = match.end()
    out.append(text[position:])
    return "".join(out)


def _output_name(text, script_env):
    """The filename a build script writes, from its last ``-o`` argument.

    The last one wins, for the same reason the last ``arch=`` assignment does: a
    script that builds more than one target ends with the one a plain invocation
    produces. The argument may be quoted and may mix a literal prefix with an
    expansion (``"$out/liblaya_cuda.so"``). Expansion is applied to the word
    first and quotes are trimmed after, because the closing quote of
    ``-o "$out/libx.so"`` is only trailing relative to the expanded word.
    """
    name = ""
    for line in text.splitlines():
        for match in re.finditer(r"(?<!\S)-o\s+(\S+)", line):
            expanded = _expand_string(match.group(1), script_env)
            candidate = os.path.basename(expanded).strip().strip('"').strip("'")
            if candidate:
                name = candidate
    return name


def gencode_targets(lines, initial_env=None):
    """Architectures that actually reach nvcc, from the ``-gencode`` flags.

    The environment is tracked line by line, so `arch=89 ... nvcc ... arch=90 ...
    nvcc` resolves each invocation against the value in force at that point. A
    literal in the flag wins over any variable, because the literal is what nvcc
    receives: a script that assigns `arch=${2:-89}` and then writes
    `-gencode arch=compute_90,code=sm_90` builds sm_90, and reading only the
    assignment would report a target the compiler never sees.
    """
    env = dict(initial_env or {})
    found = []
    for line in lines:
        match = _SCRIPT_RESOLVED.match(line)
        if match and not _GENCODE.search(line):
            env[match.group(1)] = match.group(2)
            continue
        for flag in _GENCODE.finditer(line):
            expanded = _expand_string(flag.group(1).strip('"'), env)
            for value in re.findall(r"(?:sm|compute)_(\d+)", expanded):
                target = int(value)
                if target not in found:
                    found.append(target)
    return sorted(found)


def parse_build_script(text):
    """Extract the declared architectures and output name from a build script.

    The script is read as text: it is never executed, because CI must not run
    arbitrary repository code to decide whether a manifest is honest.
    """
    lines = [_strip_comment(line) for line in text.splitlines()]

    # Architecture names may be reassigned per target (`arch=89` ... `arch=90`),
    # so every declaration is collected rather than only the final value. A
    # manifest claim is about what the script can reach, not about its last
    # assignment. `env` still keeps the final value, which is what the output
    # path expands against.
    env = {}
    architectures = []
    for line in lines:
        match = _SCRIPT_RESOLVED.match(line)
        if not match:
            continue
        name, raw = match.group(1), match.group(2)
        env[name] = raw
        if not _BUILD_SCRIPT_ARCH_FIELD.search(name):
            continue
        resolved = _expand_string(raw, env).strip()
        if resolved.isdigit():
            value = int(resolved)
            if value not in architectures:
                architectures.append(value)

    output = _output_name("\n".join(lines), env)

    # Architectures appearing literally anywhere in the script. Kept separate
    # from `gencode_architectures` because it is the weaker signal: it also
    # matches a target named in a comment that survived, or in a variable that
    # is never passed to nvcc.
    literal = []
    for match in _ARCH_IN_SCRIPT.finditer("\n".join(lines)):
        value = int(match.group(1))
        if value not in literal:
            literal.append(value)

    gencode = gencode_targets(lines)

    return {"architectures": architectures, "literal_architectures": literal,
            "gencode_architectures": gencode,
            "output": output, "env": env}

