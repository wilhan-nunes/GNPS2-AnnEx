# -*- coding: utf-8 -*-
import os

SUPPORTED_WORKFLOWS = {"auto", "classic", "fbmn", "librarysearch"}
ENTROPY_USI_TIMEOUT_SEC = int(os.environ.get("ENTROPY_USI_TIMEOUT_SEC", 8))
LIBRARY_RESULTS_CACHE_TIMEOUT_SEC = 86400
