import os
import time
import threading
from flask import Flask, request, jsonify, Response, send_from_directory
from smart_home_api import execute_command

WEBAPP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp")
latest_jpeg_frame = None

# Central state shared with the Raspberry Pi client
remote_state = {
    "waiting_for_point": False,
    "pending_calibration_name": None,
    "calibrated_objects": {},
    "status_text": "Waiting for Raspberry Pi...",
    "status_color": [0, 0, 255],
    "clear_requested": False
}

flask_app = Flask(__name__)

@flask_app.route("/")
def index():
    return send_from_directory(WEBAPP_DIR, "index.html")

@flask_app.route("/calibrate", methods=["POST"])
def calibrate_route():
    global remote_state
    if remote_state["waiting_for_point"]:
        return jsonify({"ok": False, "error": f"Already armed for '{remote_state['pending_calibration_name']}'."}), 409

    data = request.get_json(silent=True) or {}
    obj_name = (data.get("name") or "").strip()
    if not obj_name:
        return jsonify({"ok": False, "error": "Missing 'name'"}), 400

    remote_state["pending_calibration_name"] = obj_name
    remote_state["waiting_for_point"] = True
    print(f"[Server] Calibration requested from web UI for: '{obj_name}'")
    return jsonify({"ok": True, "message": f"Armed - point at the {obj_name} on the Pi camera."})

@flask_app.route("/cancel", methods=["POST"])
def cancel_route():
    global remote_state
    remote_state["waiting_for_point"] = False
    remote_state["pending_calibration_name"] = None
    return jsonify({"ok": True})

@flask_app.route("/clear", methods=["POST"])
def clear_route():
    global remote_state
    remote_state["calibrated_objects"] = {}
    remote_state["waiting_for_point"] = False
    remote_state["pending_calibration_name"] = None
    remote_state["clear_requested"] = True
    print("[Server] Clear command sent to Pi.")
    return jsonify({"ok": True})

@flask_app.route("/status", methods=["GET"])
def status_route():
    return jsonify(remote_state)

@flask_app.route("/sync_state", methods=["POST"])
def sync_state_route():
    global remote_state
    data = request.get_json(silent=True) or {}
    
    if "calibrated_objects" in data:
        remote_state["calibrated_objects"] = data["calibrated_objects"]
    if "status_text" in data:
        remote_state["status_text"] = data["status_text"]
    if "active_prompts" in data:
        remote_state["active_prompts"] = data["active_prompts"]

    response_payload = {
        "waiting_for_point": remote_state["waiting_for_point"],
        "pending_calibration_name": remote_state["pending_calibration_name"],
        "clear_requested": remote_state["clear_requested"]
    }
    
    remote_state["clear_requested"] = False
    return jsonify(response_payload)

@flask_app.route("/update_frame", methods=["POST"])
def update_frame_route():
    global latest_jpeg_frame
    if request.files and "frame" in request.files:
        latest_jpeg_frame = request.files["frame"].read()
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 400

# =========================================================
# NEW ROUTE: LISTENS FOR THE PI'S TRIGGER COMMAND
# =========================================================
@flask_app.route("/trigger_command", methods=["POST"])
def trigger_command_route():
    data = request.get_json(silent=True) or {}
    device = data.get("device")
    action = data.get("action", "turn on")
    print(device, action)
    
    if device:
        print(f"\n[Server] Received target lock from Pi! Executing '{action}' on '{device}'...")
        # Spin up a background thread on the PC to handle the SDK execution
        threading.Thread(
            target=execute_command, 
            args=(device, "toggle"), 
            daemon=True
        ).start()
        return jsonify({"ok": True})
        
    return jsonify({"ok": False, "error": "No device specified"}), 400

def mjpeg_generator():
    global latest_jpeg_frame
    while True:
        if latest_jpeg_frame is not None:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + latest_jpeg_frame + b"\r\n")
        time.sleep(0.03)

@flask_app.route("/video_feed")
def video_feed():
    return Response(mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print("[Server] Starting PC Server on port 3000...")
    flask_app.run(host="0.0.0.0", port=3000, threaded=True)