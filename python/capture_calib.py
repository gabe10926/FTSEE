#%%
import cv2
import os

OUT_DIR = "calib_imgs"
WIDTH, HEIGHT = 1280, 720
TARGET = 25

os.makedirs(OUT_DIR, exist_ok=True)

existing = sorted(f for f in os.listdir(OUT_DIR) if f.startswith("img_") and f.endswith(".jpg"))
count = len(existing)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera opened at {actual_w}x{actual_h} (requested {WIDTH}x{HEIGHT})")
if (actual_w, actual_h) != (WIDTH, HEIGHT):
    print("WARNING: camera did not honor the requested resolution.")

print("SPACE = save frame   |   q = quit")

PATTERN = (9, 6)

while True:
    ok, frame = cap.read()
    if not ok:
        print("camera read failed")
        break

    preview = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    if found:
        cv2.drawChessboardCorners(preview, PATTERN, corners, found)

    status = f"{count}/{TARGET}   detected: {'YES' if found else 'no'}"
    cv2.putText(preview, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0) if found else (0, 0, 255), 2)

    cv2.imshow("calib capture", preview)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    if key == ord(' '):
        count += 1
        path = os.path.join(OUT_DIR, f"img_{count:02d}.jpg")
        cv2.imwrite(path, frame)
        print(f"saved {path}   (chessboard {'detected' if found else 'NOT detected'})")
        if count >= TARGET:
            print(f"reached {TARGET} images; you can quit now.")

cap.release()
cv2.destroyAllWindows()
print(f"done. {count} images in {OUT_DIR}/")

# %%
