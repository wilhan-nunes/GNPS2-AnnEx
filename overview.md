# GNPS2-AnnEx Overview (Simple)

## What this app does

GNPS2-AnnEx helps you review GNPS2 library annotations and gives each spectrum a **confidence score**.  
It is designed to answer one main question:  
**“How trustworthy is this annotation?”**

## High-level process

1. You enter a GNPS2 task ID (FBMN, Classical Networking, or Library Search).
2. The app loads library match results for that task.
3. For each scan, it groups hits that point to the same **core structure** (while ignoring stereochemistry differences).
4. It calculates a confidence score (0 to 100) using multiple evidence signals.
5. It assigns a confidence label:
   - **Reliable** (green)
   - **Less Reliable** (yellow)
   - **Unreliable** (red)

## Confidence metrics (most important)

The final confidence score combines these signals:

1. **Structure agreement (50%)**
   - How many matches agree on the same core structure.
   - More agreement = stronger confidence.

2. **Spectral entropy similarity (25%)** *(optional)*
   - Compares the query spectrum to the selected library spectrum using entropy similarity.
   - This is controlled by a checkbox in the input card.
   - If disabled (or unavailable), the app uses mean cosine as fallback.

3. **Best cosine score (10%)**
   - Uses the strongest cosine match as an extra quality signal.

4. **Library class quality (15%)**
   - Uses `Library_Class` values:
     - **Class 1** = highest confidence source (best)
     - **Class 2** = medium
     - **Class 3** = lower confidence source
   - More Class 1 support increases the score; more Class 3 support lowers it.

## Confidence labels

After combining the metrics, the score is translated to:

- **Reliable**: score >= 80
- **Less Reliable**: 55 to 79.9
- **Unreliable**: < 55

## About the entropy checkbox

- **Checked**: entropy similarity is included (and spectrum peak requests are made if needed).
- **Unchecked**: entropy requests are skipped, and scoring uses cosine-based fallback for that part.

## What you see in results

For each scan, the app shows:

- Proposed compound
- Support (agreeing matches / total matches)
- Entropy metric source (entropy vs fallback)
- Cosine values
- Library class quality and class hit profile
- Final score and confidence label
