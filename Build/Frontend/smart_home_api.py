import urllib.request
import urllib.error
import json
import subprocess

def execute_command(device_id, action):
    """
    Executes a smart home command with ZERO dependencies to prevent 
    protobuf/click version conflicts with MediaPipe and Flask.
    """
    command_string = f"{action} the {device_id}"
    print(f"\n[Smart Home] Attempting to execute: '{command_string}'")

    # ==============================================================
    # METHOD 1: HTTP WEBHOOKS (RECOMMENDED HACKATHON METHOD)
    # Requires zero pip installs. Use IFTTT.com to link a webhook 
    # to your Google Assistant, Kasa, Hue, or Smart Life devices.
    # ==============================================================
    
    # 1. Go to IFTTT.com -> Create -> "If Webhooks (Receive a web request), Then Google Assistant/Smart Life (Turn on device)"
    # 2. Get your key from: https://ifttt.com/maker_webhooks
    IFTTT_EVENT_NAME = f"{action}_{device_id}".replace(" ", "_").lower()
    IFTTT_KEY = "YOUR_IFTTT_WEBHOOK_KEY_HERE" 
    
    url = f"https://maker.ifttt.com/trigger/{IFTTT_EVENT_NAME}/with/key/{IFTTT_KEY}"

    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                print(f"[Smart Home] Success! Triggered {device_id} via Webhook.")
                return True
    except urllib.error.URLError as e:
        print(f"[Smart Home] Webhook failed or not configured: {e.reason}")
        print("[Smart Home] Falling back to isolated Google Assistant SDK...")

    # ==============================================================
    # METHOD 2: ISOLATED VIRTUAL ENVIRONMENT (FALLBACK)
    # If you MUST use the Google Assistant SDK, you cannot run it in 
    # the same Python environment as YOLO/Mediapipe. 
    # ==============================================================
    
    # Create a separate folder/venv just for google-assistant-sdk, 
    # then point this path specifically to THAT environment's python.exe
    # Example (Windows): "C:/Users/madde/assistant_env/Scripts/python.exe"
    # Example (Raspberry Pi): "/home/pi/assistant_env/bin/python"
    
    ISOLATED_PYTHON_PATH = r"C:\Users\madde\Coding Projects\Shellhacks\Shellhacks 2026\Web App\backend\venv\Scripts\python.exe"
    
    MODEL_ID = "point-smart-model-1"
    INSTANCE_ID = "point-smart-device-1"

    try:
        # We use a completely separate Python executable so its broken 
        # dependencies cannot touch our currently running AI script.
        subprocess.run([
            ISOLATED_PYTHON_PATH, 
            "-m", "googlesamples.assistant.grpc.textinput",
            "--device-model-id", MODEL_ID,
            "--device-id", INSTANCE_ID
        ], input=command_string + "\n", check=True, capture_output=True, text=True)
        
        print(f"[Smart Home] Success via Isolated SDK!")
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        error_msg = getattr(e, 'stderr', str(e))
        print(f"[Smart Home] Isolated SDK Error: {error_msg}")
        print("-> To fix: Set ISOLATED_PYTHON_PATH to a separate venv containing the Google SDK.")
        return False
execute_command("South Lamp", "turn on")