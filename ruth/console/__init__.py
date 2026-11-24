"""Optional live monitoring + computational-steering console.

Enabled via the "console" section of config.json; see README.md. The console
runs on the master rank only and hooks the simulation via an end-of-step
callback — when disabled nothing here is imported.
"""

from .console import SimulationConsole


def create_console(console_args):
    """Create the console and start its HTTP server thread."""
    console = SimulationConsole(console_args)
    console.start()
    return console
