# -*- coding: utf-8 -*-
import os

SUPPORTED_WORKFLOWS = {"auto", "classic", "fbmn", "librarysearch"}
ENTROPY_USI_TIMEOUT_SEC = int(os.environ.get("ENTROPY_USI_TIMEOUT_SEC", 8))
LIBRARY_RESULTS_CACHE_TIMEOUT_SEC = 86400

# SharedPeaks normalisation: x / (x + K) maps raw peak counts to [0, 1].
K_SHARED_PEAKS = 5

# MZErrorPPM multiplicative precision factor: exp(-|ppm| / PPM_SCALE).
# MZ_PRECISION_FLOOR prevents over-penalising low-resolution instruments.
PPM_SCALE = 10.0
MZ_PRECISION_FLOOR = 0.5
