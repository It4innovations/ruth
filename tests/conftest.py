import pytest
import glob
import os


@pytest.fixture(autouse=True)
def cleanup_h5_pickle():
    """Remove any .h5 and .pickle files in the current working directory before and after each test.

    This keeps tests isolated w.r.t. generated FCD HDF5 files and saved pickles. It operates only in
    the current working directory (does not recurse into subdirectories) to avoid removing repository assets.
    """
    # pre-test cleanup
    for p in glob.glob("*.h5") + glob.glob("*.pickle"):
        try:
            os.remove(p)
        except Exception:
            pass

    yield

    # post-test cleanup
    for p in glob.glob("*.h5") + glob.glob("*.pickle"):
        try:
            os.remove(p)
        except Exception:
            pass

