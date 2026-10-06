"""Cache key for the native libraries compiled at first use.

A library depends on its C++ source, the operating-system family and the CPU architecture. It
does not depend on the OS release: the builds link statically (`-static` on Windows). The key
used to include platform.platform(), which carries the exact Windows build, so a library built on
one machine was never found on another and the Python engines ran instead, about 250x slower on
refinery L3. That made it impossible to ship prebuilt libraries in an installer.
"""
from __future__ import annotations

import platform

ABI_VERSION = "1"


def native_abi_tag() -> bytes:
    return f"{platform.system()}-{platform.machine().lower()}-abi{ABI_VERSION}".encode()
