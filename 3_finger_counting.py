"""
Phase 3: Rule-based Finger Counter
------------------------------------
Uses MediaPipe Hands landmarks (from Phase 2) to count extended fingers (0-5)
with pure geometry — no ML training required.

Logic:
- For index/middle/ring/pinky: finger is "extended" if the tip landmark
  is ABOVE the pip (knuckle) landmark (smaller y = higher on screen).
- For the thumb: it moves sideways rather than up/down, so we compare
  the tip's x-position to the ip joint's x-position, and flip the
  comparison depending on whether it's the left or right hand.

Run this file directly to test live via webcam.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Landmark indices for each finger: (tip, pip/ip)
FINGER_LANDMARKS = {
    "index": (8, 6),
    "middle": (12, 10),
    "ring": (16, 14),
    "pinky": (20, 18),
}
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2


def is_finger_extended(landmarks, tip_idx, pip_idx):
    """Non-thumb fingers: extended if tip is above the pip joint."""
    return landmarks[tip_idx].y < landmarks[pip_idx].y


def is_thumb_extended(landmarks, handedness_label):
    """
    Thumb: compare x-coordinates since it extends sideways.
    MediaPipe's 'handedness_label' is from the camera's perspective
    (mirrored), so 'Right' hand thumb extends to the LEFT on screen, etc.
    Tune/flip this if you're not mirroring your webcam feed.
    """
    tip_x = landmarks[THUMB_TIP].x
    mcp_x = landmarks[THUMB_MCP].x

    if handedness_label == "Right":
        return tip_x < mcp_x
    else:  # "Left"
        return tip_x > mcp_x


def count_fingers(landmarks, handedness_label):
    count = 0
    extended = {}

    # Thumb
    thumb_ext = is_thumb_extended(landmarks, handedness_label)
    extended["thumb"] = thumb_ext
    count += thumb_ext

    # Other four fingers
    for name, (tip, pip) in FINGER_LANDMARKS.items():
        ext = is_finger_extended(landmarks, tip, pip)
        extended[name] = ext
        count += ext

    return count, extended


def main():
    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1,
    ) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            # Mirror the frame so it feels natural (like a mirror)
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            count_text = "0"

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = handedness.classification[0].label  # "Left" or "Right"
                    count, extended = count_fingers(hand_landmarks.landmark, label)
                    count_text = str(count)

                    mp_drawing.draw_landmarks(
                        frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                    )

                    # Debug: show which fingers are extended
                    y0 = 90
                    for i, (fname, is_ext) in enumerate(extended.items()):
                        color = (0, 255, 0) if is_ext else (0, 0, 255)
                        cv2.putText(
                            frame, f"{fname}: {is_ext}", (10, y0 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
                        )

            # Big count display
            cv2.putText(
                frame, f"Count: {count_text}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3,
            )

            cv2.imshow("Phase 3: Rule-Based Finger Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()