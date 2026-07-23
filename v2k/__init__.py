"""M0: monocular video -> canonical quadruped keypoint trajectory.

Produces the same `data/processed/<clip>.npz` dict that
`retarget/skeleton.py::extract_keypoints` emits, so `retarget/retarget.py`
runs on video-derived input with zero changes.
"""
