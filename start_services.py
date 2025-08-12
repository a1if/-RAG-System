import subprocess
import socket
import os
import time
import requests
from pathlib import Path

OVERALL_START_TIME = time.time()
# The frontend will create this file in the project root
FRONTEND_READY_FILE = Path(".frontend_ready") 

def find_free_port(start=8000, max_tries=20):
    for port in range(start, start + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    raise RuntimeError("No free port found in range.")

def write_port_file(port):
    with open(".port", "w") as f:
        f.write(str(port))

def wait_for_backend(port):
    for _ in range(60):  # wait up to ~10s
        try:
            r = requests.get(f"http://localhost:{port}/docs", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def wait_for_frontend_ready():
    print("⏳ Waiting for frontend to become ready...")
    for _ in range(180): # Wait up to ~30s (60 * 0.5s) for frontend to signal readiness
        if FRONTEND_READY_FILE.exists():
            print("✅ Frontend readiness signal received.")
            try:
                with open(FRONTEND_READY_FILE, "r") as f:
                    timestamp_str = f.read().strip()
                    if timestamp_str:
                        return float(timestamp_str)
            except Exception as e:
                print(f"WARNING: Could not read frontend ready timestamp: {e}")
            return time.time() # Fallback to current time if file content is bad
        time.sleep(0.5)
    return None 

def clean_up_files():
    if Path(".port").exists():
        os.remove(".port")
        print("Cleaned up .port file.")
    if FRONTEND_READY_FILE.exists():
        os.remove(FRONTEND_READY_FILE)
        print(f"Cleaned up {FRONTEND_READY_FILE} file.")

def launch_backend(port):
    env = os.environ.copy()
    env["BACKEND_PORT"] = str(port)
    return subprocess.Popen([
        "uvicorn", "app.main:app",
        "--port", str(port),
        "--host", "0.0.0.0"
    ], env=env, cwd=".", stdout=None, stderr=None)  # This will show output in the console
       #stdout=subprocess.PIPE, stderr=subprocess.PIPE

def launch_streamlit():
    streamlit_exe = "streamlit.cmd" if os.name == "nt" else "streamlit"
    return subprocess.Popen([
        streamlit_exe,
        "run",
        "frontend/streamlit_app.py"
    ], cwd=".", # This ensures Streamlit is run from the project root
       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

if __name__ == "__main__":
    clean_up_files()
    
    backend_port = find_free_port()
    write_port_file(backend_port)

    print(f"🚀 Starting backend on port {backend_port}...")
    backend_proc = launch_backend(backend_port)

    time.sleep(1) # Give backend a moment to start outputting logs

    if not wait_for_backend(backend_port):
        print("❌ Backend failed to start in time or is not responsive. Check backend logs for errors.")
        backend_proc.terminate()
        clean_up_files()
        exit(1)

    print("🚀 Launching Streamlit frontend...")
    frontend_proc = launch_streamlit()

    frontend_ready_timestamp = wait_for_frontend_ready()
    
    if frontend_ready_timestamp:
        total_load_time = frontend_ready_timestamp - OVERALL_START_TIME
        print(f"\n🎉 **Full Application Load Time (Backend startup to Frontend ready): {total_load_time:.2f} seconds** 🎉\n")
    else:
        print("\n⚠️ Frontend did not signal readiness in time (timeout). Full load time could not be calculated. Check Streamlit logs for errors.\n")

    try:
        while True:
            backend_retcode = backend_proc.poll()
            frontend_retcode = frontend_proc.poll()

            if backend_retcode is not None:
                print(f"❗ Backend process exited with code {backend_retcode}. Restarting...")
                backend_proc = launch_backend(backend_port)
                if not wait_for_backend(backend_port):
                    print("❌ Restarted backend failed to start.")
                    clean_up_files()
                    exit(1)
            
            if frontend_retcode is not None:
                print(f"❗ Frontend process exited with code {frontend_retcode}. Restarting...")
                frontend_proc = launch_streamlit()

            time.sleep(2)
    except KeyboardInterrupt:
        print("🔻 Shutting down processes...")
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        clean_up_files()
        print("✅ All services shut down and files cleaned.")

