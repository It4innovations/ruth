import os
import sys
import logging

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()

console_logger = logging.getLogger()
console_logger.setLevel(LOGLEVEL)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(LOGLEVEL)
console_logger.addHandler(handler)
