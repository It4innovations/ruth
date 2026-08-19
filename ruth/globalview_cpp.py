"""Re-export of the compiled globalview_cpp module under the ruth namespace.

Importing the same pybind11 .so under two module names registers its types
twice and fails; this shim keeps ruth.globalview_cpp and globalview_cpp
pointing at one module object.
"""
from globalview_cpp import *  # noqa: F401,F403
from globalview_cpp import GlobalView, FCDRecord  # noqa: F401
