"""Standalone Vuer viewer (no robot) to debug the controller JS patch in a
normal browser. Loads patched vuer, serves a dummy image, prints any client
errors that bubble up. Open http://localhost:8012 in a browser console."""

import sys
import time
from pathlib import Path
from multiprocessing import shared_memory

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tj_controller_teleop import TJControllerViewer, patch_vuer, install_nocache_patch

RES = (720, 1280)
img_shape = (RES[0], 2 * RES[1], 3)


def main():
    patch_vuer()
    install_nocache_patch()
    shm = shared_memory.SharedMemory(create=True,
                                     size=int(np.prod(img_shape)) * np.uint8().itemsize)
    img = np.ndarray(img_shape, dtype=np.uint8, buffer=shm.buf)
    img[:] = 80  # gray
    viewer = TJControllerViewer(shm.name, img_shape, port=8012)
    viewer.start()
    print("viewer up on http://localhost:8012", flush=True)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()

