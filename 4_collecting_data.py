"""
Phase 4: Collect landmark data for the number-gesture recognizer.

How it works:
- Show a hand gesture for a digit (0-5) in front of the webcam.
- Press the number key for that digit ONCE to start a "burst capture."
- For the next 30 seconds, slowly move/rotate/tilt your hand while
  keeping the same gesture. The script automatically grabs a sample
  roughly every 0.3s during that window, landing at ~100 samples.
- Repeat for each digit until you have ~100 samples each.
- Press 's' to save everything to landmarks.csv, or 'q' to quit
  without saving.

Requires: opencv-python, mediapipe
    pip install opencv-python mediapipe
"""

import csv
import time

import cv2
import mediapipe as mp

# ---- Settings you can tweak ----
BURST_DURATION = 30.0       # seconds the burst capture runs for
CAPTURE_INTERVAL = 0.3      # seconds between auto-captures (30s / 0.3s ~= 100 samples)
TARGET_SAMPLES_PER_DIGIT = 100
OUTPUT_FILE = "landmarks.csv"
VALID_DIGITS = ["0", "1", "2", "3", "4", "5"]

# ---- MediaPipe setup (same as phase 3) ----
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

# In-memory storage for collected samples: list of (label, [63 floats])
collected_samples = []
# Track counts per digit so you can see progress on screen
sample_counts = {d: 0 for d in VALID_DIGITS}


def normalize_landmarks(hand_landmarks):
    """
    Convert MediaPipe's 21 landmarks into a flat list of normalized
    (x, y, z) coordinates, relative to the wrist, and scaled so hand
    size/distance from camera doesn't matter.
    """
    points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    # Landmark 0 is the wrist -- use it as the origin.
    wrist_x, wrist_y, wrist_z = points[0]
    shifted = [(x - wrist_x, y - wrist_y, z - wrist_z) for (x, y, z) in points]

    # Scale by wrist -> middle-finger-MCP (landmark 9) distance,
    # so a hand close to the camera and a hand far away produce
    # similar-sized numbers.
    ref_x, ref_y, ref_z = shifted[9]
    scale = (ref_x**2 + ref_y**2 + ref_z**2) ** 0.5
    if scale == 0:
        scale = 1e-6  # avoid divide-by-zero on a bad frame

    normalized = []
    for (x, y, z) in shifted:
        normalized.extend([x / scale, y / scale, z / scale])

    return normalized  # length 63 (21 landmarks * 3 coords)


def main():
    cap = cv2.VideoCapture(0)

    burst_active = False
    burst_label = None
    burst_end_time = 0.0
    last_capture_time = 0.0

    print("Ready. Press 0-5 to start a burst capture for that digit.")
    print("Press 's' to save to CSV, 'q' to quit without saving.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural movement
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        current_landmarks = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            current_landmarks = normalize_landmarks(hand_landmarks)

        now = time.time()

        # If a burst is running, auto-capture on the interval
        if burst_active:
            remaining = burst_end_time - now
            if remaining <= 0:
                burst_active = False
                burst_label = None
            elif current_landmarks is not None and (now - last_capture_time) >= CAPTURE_INTERVAL:
                collected_samples.append((burst_label, current_landmarks))
                sample_counts[burst_label] += 1
                last_capture_time = now

        # ---- On-screen info ----
        y = 30
        if burst_active:
            cv2.putText(frame, f"BURST capturing digit {burst_label}  ({remaining:.1f}s left)",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y += 30
        else:
            cv2.putText(frame, "Press 0-5 to start a burst capture",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30

        counts_text = "  ".join(f"{d}:{sample_counts[d]}" for d in VALID_DIGITS)
        cv2.putText(frame, counts_text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Landmark Collection - Phase 4", frame)

        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key) if key != 255 else ""

        if key_char in VALID_DIGITS and not burst_active:
            burst_active = True
            burst_label = key_char
            burst_end_time = now + BURST_DURATION
            last_capture_time = 0.0  # capture immediately on first frame
            print(f"Starting burst for digit {burst_label}...")

        elif key_char == "s":
            save_to_csv()
            print(f"Saved {len(collected_samples)} samples to {OUTPUT_FILE}")

        elif key_char == "q":
            break

    cap.release()
    cv2.destroyAllWindows()


def save_to_csv():
    header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for label, landmarks in collected_samples:
            writer.writerow(landmarks + [label])


if __name__ == "__main__":
    main()