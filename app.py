import os
import time
import threading
import traceback
import uuid

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "yolo11n.pt"
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "0")

INFER_SIZE = 320
MAX_FRAME_WIDTH = 640
CONFIDENCE = 0.35
JPEG_QUALITY = 75

TARGET_CLASSES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
}

DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()

EVENT_COOLDOWN = 3.0
MAX_ZONES = 20

print("=" * 55)
print("AEROGUARD MULTI-ZONE SURVEILLANCE")
print("=" * 55)
print("Model:", MODEL_NAME)
print("Device:", DEVICE)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

model = YOLO(MODEL_NAME)
print("Model loaded successfully.")

# ============================================================
# SHARED STATE
# ============================================================

lock = threading.Lock()
capture_lock = threading.Lock()
stop_event = threading.Event()

camera = None
worker = None

latest_frame = None
latest_frame_id = 0

state = {
    "running": False,
    "camera_connected": False,
    "error": None,
    "fps": 0.0,
    "people": 0,
    "vehicles": 0,
    "objects": [],
    "events": [],
    "zones": [],
    "zone": [],
    "frame": None,
    "frame_id": 0,
}

last_event_times = {}
event_times_lock = threading.Lock()


# ============================================================
# EVENTS
# ============================================================

def add_event(message):
    event = {
        "time": time.strftime("%H:%M:%S"),
        "message": str(message),
    }

    with lock:
        state["events"].insert(0, event)
        state["events"] = state["events"][:30]


# ============================================================
# CAMERA HELPERS
# ============================================================

def resize_frame(frame):
    height, width = frame.shape[:2]

    if width > MAX_FRAME_WIDTH:
        scale = MAX_FRAME_WIDTH / width
        new_width = MAX_FRAME_WIDTH
        new_height = max(1, int(height * scale))

        frame = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )

    return frame


def open_camera():
    source = CAMERA_SOURCE.strip()

    if source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)

    # Best effort only. Some IP camera backends ignore this.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Cannot open camera source: {source}. "
            "Check the camera index or phone stream URL."
        )

    return cap


# ============================================================
# POLYGON VALIDATION
# ============================================================

def valid_polygon(points):
    if not isinstance(points, list) or len(points) < 3:
        return None

    cleaned = []

    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return None

        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError, OverflowError):
            return None

        if not np.isfinite([x, y]).all():
            return None

        cleaned.append([
            float(np.clip(x, 0.0, 1.0)),
            float(np.clip(y, 0.0, 1.0)),
        ])

    if len(cleaned) > 300:
        indices = np.linspace(0, len(cleaned) - 1, 300, dtype=int)
        cleaned = [cleaned[i] for i in indices]

    polygon = np.asarray(cleaned, dtype=np.float32)

    if abs(cv2.contourArea(polygon)) < 0.0001:
        return None

    return cleaned


def create_zone_pixels(points, width, height):
    if len(points) < 3:
        return None

    try:
        values = np.asarray(points, dtype=np.float64)

        if values.ndim != 2 or values.shape[1] != 2:
            return None

        if not np.isfinite(values).all():
            return None

        values = np.clip(values, 0.0, 1.0)

        scale = np.array([
            max(0, width - 1),
            max(0, height - 1),
        ])

        polygon = np.rint(values * scale).astype(np.int32)

        if abs(cv2.contourArea(polygon)) < 1:
            return None

        return polygon

    except (ValueError, TypeError, OverflowError, cv2.error):
        return None


def sync_legacy_zone():
    state["zone"] = (
        state["zones"][0]["points"] if state["zones"] else []
    )


# ============================================================
# MULTI-ZONE MANAGEMENT
# ============================================================

def get_zones():
    with lock:
        return [
            {
                "id": zone["id"],
                "name": zone["name"],
                "points": zone["points"],
            }
            for zone in state["zones"]
        ]


def add_zone(name, points):
    clean_points = valid_polygon(points)

    if clean_points is None:
        return None, "Draw a valid polygon with at least 3 points."

    with lock:
        if len(state["zones"]) >= MAX_ZONES:
            return None, f"Maximum of {MAX_ZONES} zones reached."

        zone = {
            "id": uuid.uuid4().hex[:8],
            "name": name,
            "points": clean_points,
        }

        state["zones"].append(zone)
        sync_legacy_zone()

    add_event(f"Restricted zone added: {name}")
    return zone, None


def remove_zone(zone_id):
    with lock:
        original_count = len(state["zones"])

        state["zones"] = [
            zone for zone in state["zones"]
            if zone["id"] != zone_id
        ]

        if len(state["zones"]) == original_count:
            return False

        sync_legacy_zone()

    with event_times_lock:
        for key in list(last_event_times):
            if key[0] == zone_id:
                last_event_times.pop(key, None)

    add_event(f"Restricted zone deleted: {zone_id}")
    return True


# ============================================================
# INTRUSION EVENT COOLDOWN
# ============================================================

def intrusion_event(zone, track_id, class_id, name):
    key = (
        zone["id"],
        track_id if track_id is not None else -1,
        class_id,
    )
    now = time.monotonic()

    with event_times_lock:
        previous = last_event_times.get(key, 0)

        if now - previous <= EVENT_COOLDOWN:
            return

        last_event_times[key] = now

    track_label = f"#{track_id}" if track_id is not None else "untracked"
    add_event(
        f"INTRUSION | {zone['name']} | {name} {track_label}"
    )


# ============================================================
# YOLO DETECTION AND ZONE PROCESSING
# ============================================================

def process_frame(frame):
    height, width = frame.shape[:2]

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        imgsz=INFER_SIZE,
        conf=CONFIDENCE,
        classes=list(TARGET_CLASSES.keys()),
        device=DEVICE,
        half=USE_HALF,
        verbose=False,
    )

    people = 0
    vehicles = 0
    objects = []

    with lock:
        zones = [
            {
                "id": zone["id"],
                "name": zone["name"],
                "points": list(zone["points"]),
            }
            for zone in state["zones"]
        ]

    pixel_zones = []

    for zone in zones:
        polygon = create_zone_pixels(
            zone["points"], width, height
        )

        if polygon is None:
            continue

        pixel_zones.append((zone, polygon))

        cv2.polylines(
            frame, [polygon], True, (0, 220, 255), 2
        )

        label_x = int(polygon[0][0])
        label_y = max(20, int(polygon[0][1]) - 8)

        cv2.putText(
            frame,
            zone["name"][:30],
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )

    result = results[0]

    if result.boxes is not None:
        for box in result.boxes:
            coords = box.xyxy[0].detach().cpu().tolist()

            if len(coords) != 4 or not np.isfinite(coords).all():
                continue

            x1, y1, x2, y2 = [
                int(round(value)) for value in coords
            ]

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            try:
                class_value = float(box.cls[0].item())
                confidence = float(box.conf[0].item())
            except (ValueError, TypeError, RuntimeError):
                continue

            if not np.isfinite([class_value, confidence]).all():
                continue

            class_id = int(class_value)

            if class_id not in TARGET_CLASSES:
                continue

            name = TARGET_CLASSES[class_id]
            track_id = None

            if box.id is not None:
                try:
                    value = float(box.id[0].item())
                    if np.isfinite(value):
                        track_id = int(value)
                except (ValueError, TypeError, RuntimeError):
                    pass

            if class_id == 0:
                people += 1
                color = (50, 220, 50)
            else:
                vehicles += 1
                color = (255, 170, 40)

            # Bottom-center anchor for zone intrusion tests.
            anchor = (int((x1 + x2) / 2), y2)
            entered_zones = []

            for zone, polygon in pixel_zones:
                try:
                    inside = cv2.pointPolygonTest(
                        polygon,
                        (float(anchor[0]), float(anchor[1])),
                        False,
                    ) >= 0
                except (cv2.error, ValueError, TypeError):
                    inside = False

                if not inside:
                    continue

                entered_zones.append(zone["id"])
                color = (0, 0, 255)
                intrusion_event(zone, track_id, class_id, name)

            label = name

            if track_id is not None:
                label += f" #{track_id}"

            label += f" {confidence:.0%}"

            cv2.rectangle(
                frame, (x1, y1), (x2, y2), color, 2
            )
            cv2.circle(frame, anchor, 4, color, -1)

            cv2.putText(
                frame,
                label,
                (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

            objects.append({
                "id": track_id,
                "label": name,
                "confidence": round(confidence, 3),
                "intruding": bool(entered_zones),
                "zones": entered_zones,
                "box": [x1, y1, x2, y2],
            })

    return frame, people, vehicles, objects


# ============================================================
# CAMERA CAPTURE THREAD
# ============================================================

def capture_loop(cap, capture_done, capture_error):
    global latest_frame, latest_frame_id

    failures = 0

    try:
        while not stop_event.is_set():
            ok, frame = cap.read()

            if not ok or frame is None:
                failures += 1

                if failures >= 10:
                    capture_error["message"] = (
                        "Camera read failed repeatedly."
                    )
                    break

                time.sleep(0.03)
                continue

            failures = 0
            frame = resize_frame(frame)

            # A single-slot buffer: always replace with newest frame.
            with capture_lock:
                latest_frame = frame
                latest_frame_id += 1

    except Exception as exc:
        capture_error["message"] = str(exc)
        print("[CAPTURE ERROR]", repr(exc))
        traceback.print_exc()

    finally:
        capture_done.set()


# ============================================================
# INFERENCE WORKER
# ============================================================

def camera_loop():
    global camera, latest_frame, latest_frame_id

    cap = None
    capture_thread = None
    capture_done = threading.Event()
    capture_error = {"message": None}

    try:
        add_event("Opening camera stream")
        print("[CAMERA] Opening:", CAMERA_SOURCE)

        cap = open_camera()

        with lock:
            camera = cap
            state["camera_connected"] = True
            state["error"] = None

        with capture_lock:
            latest_frame = None
            latest_frame_id = 0

        capture_thread = threading.Thread(
            target=capture_loop,
            args=(cap, capture_done, capture_error),
            daemon=True,
            name="AeroGuardCapture",
        )
        capture_thread.start()

        add_event("Camera connected")

        last_frame_id = 0
        previous_time = time.perf_counter()
        fps_ema = 0.0

        while not stop_event.is_set():
            with capture_lock:
                current_id = latest_frame_id
                frame = (
                    latest_frame.copy()
                    if latest_frame is not None
                    else None
                )

            if frame is None or current_id == last_frame_id:
                if capture_done.is_set():
                    raise RuntimeError(
                        capture_error["message"]
                        or "Camera capture thread stopped."
                    )

                time.sleep(0.002)
                continue

            # Skip captured frames that arrived while inference was busy.
            last_frame_id = current_id

            frame, people, vehicles, objects = process_frame(frame)

            now = time.perf_counter()
            elapsed = max(now - previous_time, 0.001)
            previous_time = now

            current_fps = 1.0 / elapsed
            fps_ema = (
                current_fps
                if fps_ema == 0
                else 0.85 * fps_ema + 0.15 * current_fps
            )

            cv2.putText(
                frame,
                f"FPS: {fps_ema:.1f} | DEVICE: {DEVICE}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

            if not ok:
                continue

            with lock:
                state["running"] = True
                state["camera_connected"] = True
                state["error"] = None
                state["fps"] = round(fps_ema, 1)
                state["people"] = people
                state["vehicles"] = vehicles
                state["objects"] = objects
                state["frame"] = encoded.tobytes()
                state["frame_id"] += 1

    except Exception as exc:
        print("[CAMERA WORKER ERROR]", repr(exc))
        traceback.print_exc()

        with lock:
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["camera_connected"] = False

        add_event(f"Camera worker error: {type(exc).__name__}: {exc}")

    finally:
        stop_event.set()

        # Some network-camera reads can block. Do not wait indefinitely.
        if cap is not None:
            cap.release()

        if capture_thread is not None:
            capture_thread.join(timeout=1.0)

        with lock:
            camera = None
            state["running"] = False
            state["camera_connected"] = False
            state["frame"] = None

        with capture_lock:
            latest_frame = None

        print("[CAMERA] Worker stopped")


# ============================================================
# VIDEO STREAM
# ============================================================

def mjpeg_stream():
    last_frame_id = -1

    while True:
        with lock:
            frame = state["frame"]
            frame_id = state["frame_id"]
            running = state["running"]

        if not running and frame is None:
            time.sleep(0.05)
            continue

        if frame is None or frame_id == last_frame_id:
            time.sleep(0.005)
            continue

        last_frame_id = frame_id

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Cache-Control: no-cache\r\n\r\n"
            + frame
            + b"\r\n"
        )


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.route("/api/stats")
def api_stats():
    with lock:
        return jsonify({
            "running": state["running"],
            "camera_connected": state["camera_connected"],
            "error": state["error"],
            "fps": state["fps"],
            "people": state["people"],
            "vehicles": state["vehicles"],
            "objects": state["objects"],
            "events": state["events"],
            "zones": state["zones"],
            "zone": state["zone"],
        })


@app.route("/api/start", methods=["POST"])
def start_camera():
    global worker

    with lock:
        if worker is not None and worker.is_alive():
            return jsonify({
                "ok": True,
                "message": "Camera is already running or stopping.",
            })

        state["running"] = True
        state["camera_connected"] = False
        state["error"] = None
        state["fps"] = 0.0
        state["people"] = 0
        state["vehicles"] = 0
        state["objects"] = []
        state["frame"] = None
        state["frame_id"] = 0

    stop_event.clear()

    worker = threading.Thread(
        target=camera_loop,
        daemon=True,
        name="AeroGuardInference",
    )
    worker.start()

    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
def stop_camera():
    stop_event.set()

    with lock:
        state["running"] = False

    add_event("Camera stop requested")
    return jsonify({"ok": True})


@app.route("/api/zone", methods=["POST"])
def set_zone():
    payload = request.get_json(silent=True) or {}

    name = str(payload.get("name", "")).strip()

    if not name:
        with lock:
            name = f"Zone {len(state['zones']) + 1}"

    name = name[:40]
    points = payload.get("points", [])

    zone, error = add_zone(name, points)

    if error:
        return jsonify({"ok": False, "error": error}), 400

    return jsonify({
        "ok": True,
        "zone": zone,
        "zones": get_zones(),
    })


@app.route("/api/zones", methods=["GET"])
def list_zones():
    return jsonify({"ok": True, "zones": get_zones()})


@app.route("/api/zones/<zone_id>", methods=["DELETE"])
def delete_zone(zone_id):
    if not remove_zone(zone_id):
        return jsonify({
            "ok": False,
            "error": "Zone not found.",
        }), 404

    return jsonify({
        "ok": True,
        "zones": get_zones(),
    })


@app.route("/api/zones/clear", methods=["POST"])
def clear_zones():
    with lock:
        state["zones"] = []
        sync_legacy_zone()

    with event_times_lock:
        last_event_times.clear()

    add_event("All restricted zones cleared")
    return jsonify({"ok": True, "zones": []})


@app.route("/api/zone/reset", methods=["POST"])
def reset_zone():
    # Legacy compatibility: remove the first saved zone.
    with lock:
        removed = state["zones"].pop(0) if state["zones"] else None
        sync_legacy_zone()
        remaining_zone = state["zone"]

    if removed:
        with event_times_lock:
            for key in list(last_event_times):
                if key[0] == removed["id"]:
                    last_event_times.pop(key, None)

        add_event(f"Restricted zone deleted: {removed['name']}")

    return jsonify({
        "ok": True,
        "zones": get_zones(),
        "zone": remaining_zone,
    })


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )
