# FTSEE (BioBuzzAutoDog)

Vision work for FTC DECODE (2025-2026): finding the green and purple balls with a camera, tracking them in field coordinates, and driving the robot to chase them like a dog.

<img src="https://github.com/user-attachments/assets/fefaeac9-8e91-4be8-9d23-9d82a193d1e3" alt="Ball detection screen recording" width="800">

<img src="https://github.com/user-attachments/assets/1b8c6aac-84fd-48d8-ba3c-4ead6370b630" alt="Ball tracking screen recording" width="800">

## What's here

### `docs/`

- `ftc-auto-dog.pdf` — design document for the auto dog: pixel to field projection, per-track Kalman filter, gating, Mahalanobis association, and pose time sync.

### `python/`

Desktop computer vision tests, from [FTCDecodeVision](https://github.com/tanayvin1216/FTCDecodeVision).

- `vision.py` — main AprilTag-anchored green/purple ball detection pipeline
- `vision_fixed.py`, `vision_debug.py`, `vision2.py` — variants and debug views of the pipeline
- `ballDetect.py` — ball detection plus tracking with an OpenCV tracker
- `SimpleBallTracker.py` — `SimpleBallTracker` class: per-frame green/purple detection plus CSRT tracking, meant for robot integration
- `LuxonisTracking.py` — 3D ball trajectory tracking with a Luxonis OAK-D (depthai 3.x), with CSV logging
- `empiricalTrajectory.py` — empirical trajectory recorder
- `remoteTestingRoom.py` — saves frames and debug masks for offline analysis
- `capture_calib.py`, `calibrate.py`, `calib_imgs/` — chessboard capture and camera intrinsics calibration (the `K` and distortion values the design doc's pixel to field projection needs)

### `robot/`

On-robot Kotlin code from the `tanayVision` branch of [FTC-SigmaCorns-22377/Decode](https://github.com/FTC-SigmaCorns-22377/Decode/tree/tanayVision), kept at its original `TeamCode` paths. These files depend on the rest of that repo and do not build on their own.

- `GlobalShutterVision/eyebrain.kt` — `EyeBrain`, the `VisionProcessor` port of `vision.py` for the global shutter camera
- `opmode/test/VisionDriveTest.kt` — teleop that uses `EyeBrain` on the `globalShutter` webcam to turn toward and chase a ball
- `opmode/test/ShooterExperimentTest.kt` — shooter experiment teleop for camera system integration
- `control/localization/VisionTracker.kt`, `constants/Limelight.kt`, `res/xml/teamwebcamcalibrations.xml` — team localization and camera calibration code the vision work relies on (written by Sigma Corns teammates)

## Running the Python tests

```
cd python
python -m venv .venv && source .venv/bin/activate
pip install opencv-python numpy
python vision.py
```

`vision.py` uses a local USB webcam directly and requests 1280x720 (720p). It
automatically selects the first readable camera, or a specific device can be
chosen with `python vision.py --camera 1`. Press `q` or `Esc` to quit. The
`depthai` package is only needed by the Luxonis OAK-D scripts.
