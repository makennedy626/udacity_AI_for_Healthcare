# Validation Plan

**Your Name: Matthew Kennedy**

**Intended Use Statement**

This algorithm can be used by a radiologist to aid in the diagnoses of Alzheimer's progression.

**Ground Truth Acquisition Methodology:**

The data comes from ![Medical Decathlon](medicaldecathlon.com) and has been "labeled and verified by an expert human rater, and with the best effort to mimic the accuracy required for clinical use," making it a silver standard.

**Algorithm Performance Standard:**

The algorithm's performance standard is based off of the Dice Similarity Score and the Jaccard Score.

**Target Population**

The algorithm is to be run on MRI scans of the hippocampus that are saved as a DICOM file. The ages and other demographic measurements were unavailable, so it is assumed to be able to run on any age and demographic.