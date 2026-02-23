#!/usr/bin/env python3
"""Draw gesture sequence state machine (ROS node version)."""

import json


# ===========================================================
# Drawing gesture sequence configuration  (edit as needed)
# ===========================================================
DRAW_START_GESTURE = "Victory"  # begin draw session (trained gesture)
DRAW_END_GESTURE = "Thumb_Up"  # finish session and publish (trained gesture)
DRAW_LANDMARK_IDX = 8  # landmark to trace (8 = index fingertip)
PINCH_THRESHOLD = 0.08  # normalised distance threshold for pinch
#   active draw = thumb tip (4) and index tip (8) within PINCH_THRESHOLD
# ===========================================================


def is_pinch(landmarks):
    """Return True when thumb tip (4) and index tip (8) are pinched together."""
    t, i = landmarks[4], landmarks[8]
    dx, dy = t["x"] - i["x"], t["y"] - i["y"]
    return (dx * dx + dy * dy) ** 0.5 < PINCH_THRESHOLD


class DrawSequence:
    """Two-state draw sequence machine.

    idle  → (DRAW_START_GESTURE) → running
    running + pinch              → collect point
    running + DRAW_END_GESTURE   → publish via publish_fn, back to idle
    """

    def __init__(self, publish_fn):
        """
        Args:
            publish_fn: callable(json_str) — called with the completed path JSON.
        """
        self.running = False
        self._path = []
        self._publish = publish_fn

    @property
    def path(self):
        return self._path

    def update(self, gesture, point_3d, landmarks):
        """Call once per frame with the first hand's data.

        Args:
            gesture:   gesture string (may be None)
            point_3d:  [x, y, z] world coords of DRAW_LANDMARK_IDX (may be None)
            landmarks: list of 21 normalized landmark dicts {'x', 'y', 'z'}
        """
        if not self.running:
            if gesture == DRAW_START_GESTURE:
                self.running = True
                self._path = []
        else:
            # End gesture is checked first so Closed_Fist always wins over pinch
            if gesture == DRAW_END_GESTURE:
                self._publish(json.dumps(self._path))
                self.running = False
                self._path = []
            elif is_pinch(landmarks):
                if point_3d is not None:
                    self._path.append(point_3d)
