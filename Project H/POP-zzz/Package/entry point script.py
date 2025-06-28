#!/usr/bin/env python3
import os
import sys
from halos.core.halos_core import HALOSCore
from halos.gui.main_window import HALOSApp

def main():
    # Initialize core system
    core = HALOSCore()
    
    # Check for CLI/server mode
    if '--cli' in sys.argv:
        from halos.cli import start_cli
        start_cli(core)
    elif '--server' in sys.argv:
        from halos.api import start_server
        start_server(core)
    else:
        # Launch GUI
        app = HALOSApp(core)
        app.mainloop()

if __name__ == "__main__":
    main()