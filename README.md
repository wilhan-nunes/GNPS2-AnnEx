## TopKonfidence Dashboard

This workspace now includes a GNPS2 annotation confidence dashboard at `/dashinterface2/`.

### What it does

1. Accepts a GNPS2 FBMN, Classical Networking, or Library Search task ID.
1. Retrieves library matches data from GNPS2, including the direct `librarymatches` JSON view used by Library Search.
1. Groups matches per `#Scan#` by core structure while ignoring stereochemistry.
1. Computes a confidence score from:
    - Structure support fraction (how many hits agree on the same core structure)
    - Mean cosine score
    - Max cosine score
1. Displays confidence badges:
    - Green: Reliable
    - Yellow: Less Reliable
    - Red: Unreliable

### URL parameters

You can prefill values with query parameters:

`/dashinterface2/?task=<task_id>&workflow=auto`

`workflow` can be `auto`, `classic`, `fbmn`, or `librarysearch`.
