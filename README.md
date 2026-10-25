# AeroGuard: AI Drone Surveillance System Using Real-Time Object Tracking

AeroGuard is a prototype real-time video surveillance dashboard intended to detect and track supported objects in a live camera stream and monitor user-defined polygon zones.

This README deliberately separates **confirmed project details** from **items that must be checked or filled in**. A placeholder, suggested command, or test plan is not proof that a feature has been implemented or validated.

## 1. Confirmed Project Setup

The following details are supported by the available project context and supplied source-code description.

| Item                                 | Confirmed detail                                                                                                         |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Project name                         | AeroGuard                                                                                                                |
| Project type                         | AI-assisted live video surveillance and object tracking prototype                                                        |
| Host environment                     | Windows laptop using Visual Studio Code                                                                                  |
| GPU                                  | User reports an NVIDIA GeForce GTX 1650-class GPU; exact model string should be checked on the machine                   |
| Camera                               | Android phone using IP Webcam                                                                                            |
| Camera resolution used during tuning | 640×480                                                                                                                  |
| Backend                              | Python with Flask                                                                                                        |
| Image/video processing               | OpenCV (`cv2`)                                                                                                           |
| Detection model                      | Ultralytics YOLO11n, configured as `yolo11n.pt`                                                                          |
| Tracking                             | Ultralytics tracking call with `persist=True` and `tracker="bytetrack.yaml"`                                             |
| Inference size                       | `INFER_SIZE = 320` in the supplied source description                                                                    |
| Maximum processing width             | `MAX_FRAME_WIDTH = 640` in the supplied source description                                                               |
| Confidence threshold                 | `CONFIDENCE = 0.35` in the supplied source description                                                                   |
| Device selection                     | CUDA device `0` when `torch.cuda.is_available()` is true; otherwise CPU                                                  |
| Half precision                       | Enabled conditionally when CUDA is available in the described code                                                       |
| Camera source setting                | `CAMERA_SOURCE`, with default value `"0"` in the described code                                                          |
| Web server                           | Flask configured to listen on `0.0.0.0:5000` in the described code                                                       |
| Video delivery                       | `/video_feed` MJPEG endpoint                                                                                             |
| Dashboard                            | `templates/index.html` is the known frontend path                                                                        |
| Zone interface                       | Frontend uses click-to-place polygon points                                                                              |
| Zone/backend note                    | Backend zone collection and legacy single-zone reference are described; the frontend may only draw the legacy first zone |
| Reported performance                 | User reports 30+ FPS at startup and an average of 25+ FPS afterward                                                      |
| Performance evidence quality         | Informal user-reported observations, not a controlled benchmark                                                          |

### Important scope clarification

The project is described as a **drone surveillance system**, but the known implementation context documents a phone camera stream feeding a laptop-based detection pipeline. Do not claim autonomous drone flight, onboard drone processing, or direct flight-control integration unless those functions exist in separate code and have been tested.

## 2. Features Supported by the Available Project Context

These items are described in the supplied project information and source summary. Verify current files and runtime behavior before presenting them as fully tested.

- Live video input from a configurable camera source.
- YOLO11n detection restricted to the class IDs in `TARGET_CLASSES`.
- Object tracking through ByteTrack.
- Annotation and JPEG encoding of processed frames.
- Browser video stream through `/video_feed`.
- Flask routes described for statistics and camera start/stop controls.
- Polygon zone configuration through the frontend.
- Multi-zone-related backend routes, with a possible mismatch between backend and frontend visualization.
- A capture-thread/latest-frame approach in the updated code description, intended to reduce processing backlog.

## 3. Placeholders and Details to Verify

Fill in or verify these before publishing, submitting, or sharing the project.

| Placeholder or unknown                                   | What to do                                                                                                    |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Student name, roll number, college, guide, academic year | Replace with actual academic details in the project report.                                                   |
| Exact laptop model and RAM                               | Check Windows Settings or Task Manager.                                                                       |
| Exact GPU name and VRAM                                  | Check Task Manager → Performance → GPU or Device Manager.                                                     |
| Python version                                           | Run `python --version` in the project environment.                                                            |
| Flask, OpenCV, PyTorch, Ultralytics versions             | Record installed versions from the same virtual environment used to run the app.                              |
| `requirements.txt` presence and content                  | Check the repository. Do not assume it exists or invent pinned versions.                                      |
| Exact IP Webcam stream URL                               | Copy it from the phone app. Do not use the example URL as a real address.                                     |
| Exact `TARGET_CLASSES` mapping                           | Read it from the current `app.py`.                                                                            |
| Exact API HTTP methods and response schemas              | Inspect the route definitions in the current `app.py`.                                                        |
| Zone coordinate system and save behavior                 | Confirm whether points are pixels or normalized coordinates and whether a request replaces or appends zones.  |
| Exact zone-entry/event behavior                          | Test the implementation and inspect event logic.                                                              |
| Multi-zone frontend support                              | Verify whether all zones are rendered and editable.                                                           |
| Whether CUDA was active during the reported FPS test     | Check PyTorch CUDA status and GPU utilization during a run.                                                   |
| Sustained performance                                    | Run a timed test and record results; the existing 25+ FPS figure is informal.                                 |
| Detection accuracy and alert reliability                 | No formal precision, recall, mAP, false-alarm, or missed-event results are available in the supplied context. |
| Camera-disconnect recovery and long-run stability        | Test explicitly; earlier freezing and stuttering were reported.                                               |

## 4. Technology Stack

- Python
- Flask
- OpenCV (`cv2`)
- PyTorch
- Ultralytics YOLO11n
- ByteTrack
- HTML, CSS, and JavaScript
- Android IP Webcam
- Windows and Visual Studio Code
- NVIDIA GPU, when supported by the installed PyTorch environment

## 5. Requirements

Confirmed from the current project context:

- Windows laptop and VS Code.
- Python application using Flask, OpenCV, PyTorch, and Ultralytics.
- Android phone running IP Webcam.
- A browser to view the dashboard.
- An NVIDIA GPU is reported by the user; GPU acceleration depends on compatible drivers and PyTorch.

**To verify:** exact Python version, dependency versions, available RAM/VRAM, browser version, and whether the current virtual environment has CUDA-enabled PyTorch.

## 6. Installation

Run these commands from the project directory in VS Code. They are setup instructions, not evidence that dependencies are already pinned or installed.

### 6.1 Create a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
```

If activation is blocked, use the appropriate activation command for your shell and follow your machine's security policy.

### 6.2 Install dependencies

If a `requirements.txt` file exists:

```powershell
pip install -r requirements.txt
```

If it does not exist, inspect imports in `app.py` and install the required packages. Select a PyTorch build compatible with the installed NVIDIA driver using the official guide:

https://pytorch.org/get-started/locally/

Do not invent package versions. After a working setup is confirmed, record them in a requirements file.

### 6.3 Verify GPU availability

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
```

The source description selects CUDA device 0 if CUDA is available, otherwise CPU. This command checks the active Python environment, not every environment installed on the laptop.

## 7. Camera Setup

1. Start the IP Webcam server on the Android phone.
2. Configure the stream to **640×480**, matching the known baseline used during tuning.
3. Ensure the phone and laptop can reach each other on the local network or hotspot.
4. Copy the exact stream URL shown by IP Webcam.
5. Configure `CAMERA_SOURCE` using the method implemented in the current code.

If `app.py` reads `CAMERA_SOURCE` from the environment, PowerShell syntax is:

```powershell
$env:CAMERA_SOURCE = "http://PHONE_IP:PORT/video"
```

`PHONE_IP:PORT` and the endpoint path are placeholders. Replace them with the exact values from the phone app. The described default value `"0"` typically represents a local camera index, not the phone's URL.

## 8. Run AeroGuard

The known application entry point is described as `app.py`. From the activated project environment:

```powershell
python app.py
```

The described Flask configuration uses port `5000`. If the server starts successfully, try:

```text
http://127.0.0.1:5000
```

Confirm the actual terminal output and current configuration. Access from another device requires the laptop's local IP address, reachable network connectivity, and appropriate firewall access.

## 9. Configuration Reference

Values below are supported by the supplied source description. Check the current code before modifying them.

| Setting           | Described value                            | Meaning                              |
| ----------------- | ------------------------------------------ | ------------------------------------ |
| `MODEL_NAME`      | `yolo11n.pt`                               | Model weights name                   |
| `CAMERA_SOURCE`   | Environment variable, default `"0"`        | Camera source                        |
| `INFER_SIZE`      | `320`                                      | Model inference image size           |
| `MAX_FRAME_WIDTH` | `640`                                      | Maximum processing width             |
| `CONFIDENCE`      | `0.35`                                     | Detection confidence threshold       |
| `DEVICE`          | `0` when CUDA available, otherwise `"cpu"` | Inference device                     |
| `USE_HALF`        | CUDA-dependent                             | Conditional half-precision inference |
| Tracker           | `bytetrack.yaml`                           | Tracker configuration                |
| JPEG quality      | `75` in the updated code description       | JPEG stream quality                  |
| Flask bind        | `0.0.0.0:5000`                             | Server interface and port            |

A lower inference size may improve speed but can reduce detection detail for small or distant objects. Test changes under repeatable conditions.

## 10. Dashboard Workflow

The described workflow is:

1. Start IP Webcam.
2. Start `app.py`.
3. Confirm video and statistics update.
4. Click polygon vertices on the video to draft a zone.
5. Save or apply the zone using the frontend controls.
6. Test object detection and tracking.
7. Test zone entry and exit behavior and record what the app actually displays.
8. Stop the camera or processing through the UI, or terminate the server.

**To verify:** exact button labels, event rules, whether zones persist after restart, and whether multiple zones are all visible in the current frontend.

## 11. API Reference

Route names below are described in the available source context. Confirm methods, request bodies, status codes, and response JSON in the current `app.py` before treating this table as a formal API contract.

| Endpoint               | Described purpose             | Verification needed                                                                    |
| ---------------------- | ----------------------------- | -------------------------------------------------------------------------------------- |
| `/video_feed`          | MJPEG video stream            | Confirm response headers and stream behavior.                                          |
| `/api/stats`           | Application statistics/status | Record actual JSON fields.                                                             |
| `/api/start`           | Start processing              | Confirm accepted method and response.                                                  |
| `/api/stop`            | Stop processing               | Confirm accepted method and response.                                                  |
| `/api/zone`            | Legacy zone operation         | Frontend is described as sending `{"points": draft}`. Confirm replace/append behavior. |
| `/api/zones`           | Multi-zone operations         | Confirm methods and payload.                                                           |
| `/api/zones/<zone_id>` | Delete a zone                 | Confirm identifier type and response.                                                  |
| `/api/zones/clear`     | Clear zones                   | Confirm accepted method.                                                               |
| `/api/zone/reset`      | Reset zone                    | Described frontend calls this route.                                                   |

Conceptual payload only:

```json
{
  "points": [
    [100, 100],
    [300, 100],
    [280, 260],
    [120, 250]
  ]
}
```

The coordinates above are examples, not actual saved zone coordinates. Confirm the coordinate system and validation logic in code.

## 12. Performance: Confirmed Observation vs. Required Test

### User-reported observation

- The application reached 30+ FPS at startup.
- It later averaged 25+ FPS.
- Reducing the phone stream from 1080p to 640×480 improved performance.
- Video/dashboard stuttering or freezing had previously been reported.

These are user reports, not independently measured benchmark results. The measurement duration and exact FPS definition are not yet documented.

### Recommended stability test

1. Warm up inference for about 60 seconds.
2. Run for at least 10 minutes.
3. Record FPS, GPU utilization, memory use, and visible stutters once per minute.
4. Repeat at least three times before reporting an average.
5. State whether the metric is inference FPS, processed-frame FPS, or browser playback FPS.

| Run | Duration | Average FPS | Minimum FPS | GPU use     | Stutter/freeze notes |
| --- | -------- | ----------- | ----------- | ----------- | -------------------- |
| 1   | 10 min   | `[measure]` | `[measure]` | `[measure]` | `[record]`           |
| 2   | 10 min   | `[measure]` | `[measure]` | `[measure]` | `[record]`           |
| 3   | 10 min   | `[measure]` | `[measure]` | `[measure]` | `[record]`           |

## 13. Project Structure

Known paths include `app.py` and `templates/index.html`. The full repository listing has not been verified. A possible layout is shown below; remove entries that do not exist in the actual project.

```text
AeroGuard/
├── app.py
├── README.md
├── templates/
│   └── index.html
├── yolo11n.pt          # Optional local model weights
├── bytetrack.yaml      # May be supplied by Ultralytics
└── requirements.txt    # Verify whether present
```

## 14. Limitations and Responsible Use

- Detection depends on lighting, camera angle, distance, occlusion, and supported classes.
- Tracking IDs are temporary and may change.
- No formal detection-accuracy or intrusion-event metrics are available in the supplied context.
- Long-duration stability and camera-disconnect recovery need explicit tests.
- The frontend may not yet display every backend zone.
- The known setup uses a phone camera stream and laptop processing. Do not claim autonomous drone control or onboard processing without separate implementation evidence.
- The Flask development server should not be exposed to the public internet. Restrict access to trusted networks.
- Use surveillance only in authorized settings, respect privacy, and do not rely on this prototype for safety-critical decisions.

## 15. Future Improvements

- Add reliable camera timeout, reconnect, and shutdown handling.
- Complete multi-zone creation, naming, editing, and visualization.
- Add timestamped event history and configurable cooldowns.
- Measure inference FPS, stream FPS, and end-to-end latency separately.
- Add automated API tests and structured logging.
- Add authentication and network access controls before broader deployment.
- Pin dependencies and document a reproducible setup.

## 16. References

- Ultralytics documentation: https://docs.ultralytics.com/
- Ultralytics tracking mode: https://docs.ultralytics.com/modes/track/
- Flask documentation: https://flask.palletsprojects.com/
- OpenCV documentation: https://docs.opencv.org/
- PyTorch installation guide: https://pytorch.org/get-started/locally/
- Python documentation: https://docs.python.org/3/
