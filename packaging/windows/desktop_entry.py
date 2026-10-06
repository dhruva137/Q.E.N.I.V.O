"""Entry point frozen into QENIVO.exe by PyInstaller (see qenivo_desktop.spec)."""
import sys

from qenivo.desktop.app import main

if __name__ == "__main__":
    sys.exit(main())
