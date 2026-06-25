# -*- coding: utf-8 -*-
import os

SUPPORTED_WORKFLOWS = {"auto", "classic", "fbmn", "librarysearch"}
ENTROPY_USI_TIMEOUT_SEC = int(os.environ.get("ENTROPY_USI_TIMEOUT_SEC", 8))
LIBRARY_RESULTS_CACHE_TIMEOUT_SEC = 86400

# SharedPeaks normalisation: x / (x + K) maps raw peak counts to [0, 1].
K_SHARED_PEAKS = 5

# MZ precision factor: f + (1−f)·exp(−max(0, |ppm|−t)² / (2·S²))
# Full credit within the tolerance band; Gaussian decay beyond it.
PPM_TOLERANCE = 5.0   # t — tolerance threshold (ppm), fixed
PPM_SIGMA = 10.0      # S — Gaussian decay width (ppm), adjustable
PPM_FLOOR = 0.5       # f — minimum factor, adjustable

# Library diversity soft-cap: min(unique_source_libraries, K) / K → [0, 1].
K_LIBRARY_DIVERSITY = 5
