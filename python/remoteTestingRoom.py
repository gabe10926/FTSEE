#%%
# Remote Testing Room - Capture frames for Claude to analyze
# Press 's' to save a snapshot, 'q' to quit

import cv2
import numpy as np
import os
from datetime import datetime

# Import detection from vision.py
from vision import runPipeline, BLUE_HSV_LOWER, BLUE_HSV_UPPER, RED_HSV_LOWER_1, RED_HSV_UPPER_1, RED_HSV_LOWER_2, RED_HSV_UPPER_2
from yellow_ball import detect_blue, detect_red, detect_yellow

# Create output directory
OUTPUT_DIR = "/Users/tanayvinaykya/Desktop/robotics/vision/test_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_debug_frames(frame, frame_id):
    """Save frame and all debug masks for Claude to analyze"""

    img_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    blue_mask, _ = detect_blue(frame)
    red_mask, _ = detect_red(frame)
    yellow_mask, _ = detect_yellow(frame)

    motif, output, mask_combined, mask_blue, mask_red, mask_yellow = runPipeline(frame)

    blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\n{'='*50}")
    print(f"Frame {frame_id} Analysis:")
    print(f"{'='*50}")
    print(f"Detected motif: {motif}")
    print(f"\nBlue contours found: {len(blue_contours)}")
    for i, c in enumerate(blue_contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(f"  B{i+1}: area={area:.0f}, pos=({x},{y}), size={w}x{h}")

    print(f"\nRed contours found: {len(red_contours)}")
    for i, c in enumerate(red_contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(f"  R{i+1}: area={area:.0f}, pos=({x},{y}), size={w}x{h}")

    print(f"\nYellow contours found: {len(yellow_contours)}")
    for i, c in enumerate(yellow_contours):
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        print(f"  Y{i+1}: area={area:.0f}, pos=({x},{y}), size={w}x{h}")

    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_1_original.png", frame)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_2_output.png", output)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_3_mask_blue.png", blue_mask)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_4_mask_red.png", red_mask)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_5_mask_yellow.png", yellow_mask)
    cv2.imwrite(f"{OUTPUT_DIR}/{frame_id}_6_mask_combined.png", mask_combined)

    print(f"\nSaved to {OUTPUT_DIR}/{frame_id}_*.png")
    print(f"Share these images with Claude for analysis!")

    return motif

def main():
    cap = cv2.VideoCapture(0)

    print("="*50)
    print("REMOTE TESTING ROOM")
    print("="*50)
    print("Press 's' to save snapshot for Claude analysis")
    print("Press 'q' to quit")
    print("="*50)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run pipeline for live preview
        motif, output, mask_combined, mask_blue, mask_red, mask_yellow = runPipeline(frame)

        # Show windows
        cv2.imshow("Live Output", output)
        cv2.imshow("Blue Mask", mask_blue)
        cv2.imshow("Red Mask", mask_red)
        cv2.imshow("Yellow Mask", mask_yellow)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            frame_count += 1
            timestamp = datetime.now().strftime("%H%M%S")
            frame_id = f"frame_{timestamp}_{frame_count}"
            save_debug_frames(frame, frame_id)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# %%
# Quick single-frame test - run this cell to capture one frame
def capture_single_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        timestamp = datetime.now().strftime("%H%M%S")
        save_debug_frames(frame, f"single_{timestamp}")
        return frame
    return None

# Uncomment to capture:
# capture_single_frame()
