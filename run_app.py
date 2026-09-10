#!/usr/bin/env python3
"""
ColorAI — Application Launcher
Deep Image Colorization Using U-Net in CIE Lab Color Space

Run this script to launch both the backend inference service and frontend web studio:
    python run_app.py
"""

import os
import sys
import time
import socket
import threading
import webbrowser

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def find_available_port(starting_port=5000, max_attempts=20):
    """Finds an open TCP port on localhost."""
    for port in range(starting_port, starting_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return starting_port

def open_browser_later(url, delay_seconds=1.2):
    """Opens the user's default browser after server initialization."""
    def _open():
        time.sleep(delay_seconds)
        print(f"[ColorAI Launcher] Opening browser at: {url}")
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"[ColorAI Launcher] Note: Could not auto-open browser ({e}). Please navigate manually.")
    
    t = threading.Thread(target=_open, daemon=True)
    t.start()

def main():
    port = int(os.environ.get("PORT", 0)) or find_available_port(5000)
    url = f"http://127.0.0.1:{port}"

    print("==================================================================")
    print("  ColorAI — Deep Image Colorization Studio")
    print("  College Mini Project: U-Net in CIE Lab Color Space")
    print("==================================================================")
    print(f"  * Web Application URL : {url}")
    print(f"  * Project Root         : {PROJECT_ROOT}")
    print("  * Machine Learning    : PyTorch (Black-Box ML Integration)")
    print("==================================================================")
    print("  Press CTRL+C to terminate the server.\n")

    # Start browser opener
    open_browser_later(url)

    # Import Flask app and start server
    from backend.app import app
    app.run(host="0.0.0.0", port=port, debug=False)

if __name__ == "__main__":
    main()
