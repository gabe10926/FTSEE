#%%

import cv2, numpy as np, glob, json

PATTERN = (9, 6)                # inner corners for a 10x7-square board
SQUARE_MM = 21.59               # 0.85" squares (8.5" long side / 10 squares)

objp = np.zeros((PATTERN[0]*PATTERN[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * (SQUARE_MM/1000.0)

objpoints, imgpoints = [], []

files = sorted(glob.glob("calib_imgs/*.jpg"))
if not files:
    raise SystemExit("no images found in calib_imgs/ — run capture_calib.py first (cwd must be project root)")

for f in files:
    img = cv2.imread(f)
    img = cv2.rotate(img, cv2.ROTATE_180)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCorners(gray, PATTERN, None)
    if not ok:
        print("skip", f)
        continue
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    objpoints.append(objp)
    imgpoints.append(corners)
    print("ok", f)

h, w = gray.shape
rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

if rms < 0.5:
    quality = "GREAT"
elif rms < 1.0:
    quality = "acceptable"
elif rms < 1.5:
    quality = "marginal"
else:
    quality = "RECAPTURE — add more varied board poses (tilt/rotate/near/far)"

print(f"RMS reprojection error: {rms:.4f} px  [{quality}]")
print(f"fx = {fx:.4f}")
print(f"fy = {fy:.4f}")
print(f"cx = {cx:.4f}")
print(f"cy = {cy:.4f}")
print(f"dist = {dist.ravel()[:5]}")
print(json.dumps({
    "K": [fx, fy, cx, cy],
    "DIST_COEFFS": [float(x) for x in dist.ravel()[:5]],
    "IMAGE_WIDTH_PX": w, "IMAGE_HEIGHT_PX": h
}, indent=2))

# %%
