# Perception Engine: 3-Week MVP Roadmap

## Project Overview
**Timeline**: August 6, 2025 - August 31, 2025 (3 weeks + weekend buffer)  
**Team Size**: 3.5 developers (Howard + Zikang + Kehan + Zihan (partial))  
**Delivery**: Tool implementations with documentation for MCP/Gemini integration

## Tool Inventory & Technical Implementation

### 1. Screenshot/Screen Capture Tool
**Function**: `capture_screen`  
**Type**: Synchronous  
**Parameters**:
- `device_type` (str): "windows", "android_phone", "android_tablet"
- `monitor` (int): Monitor index for Windows (default: 0)
- `format` (str): Output format - "png" or "jpeg" (default: "png")
- `quality` (int): JPEG quality 1-100 (default: 85)

**Technical Solution by Platform**:

**Windows PC**:
- **Library**: `python-mss` for high-performance screen capture
- **Implementation**: Direct Windows API access via ctypes
- **Performance**: 3-8 FPS for full screen capture

**Android (Motorola/Lenovo)**:
- **Library**: `adb` (Android Debug Bridge) with Python wrapper
- **Implementation**: MediaProjection API via ADB commands
- **Requirements**: USB debugging enabled or wireless ADB
- **Note**: Android 11+ has built-in screen recording

```python
# Unified implementation approach
import platform
import subprocess
import mss
from PIL import Image
import base64
from io import BytesIO

class ScreenCaptureAdapter:
    def __init__(self):
        self.platform = self._detect_platform()
    
    def capture_screen(self, device_type="auto", format="png", quality=85):
        if device_type == "auto":
            device_type = self.platform
            
        if device_type == "windows":
            return self._capture_windows(format, quality)
        elif device_type.startswith("android"):
            return self._capture_android(format, quality)
    
    def _capture_windows(self, format, quality):
        with mss.mss() as sct:
            monitor = sct.monitors[0]
            screenshot = sct.grab(monitor)
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            return self._encode_image(img, format, quality)
    
    def _capture_android(self, format, quality):
        # Use ADB to capture screenshot
        try:
            # Execute screencap command
            subprocess.run(["adb", "exec-out", "screencap", "-p"], 
                         stdout=open("temp_screen.png", "wb"))
            
            # Load and process image
            img = Image.open("temp_screen.png")
            return self._encode_image(img, format, quality)
            
        except Exception as e:
            # Fallback to MediaProjection API if available
            return self._capture_android_media_projection(format, quality)
    
    def _capture_android_media_projection(self, format, quality):
        # Requires Android app with MediaProjection permission
        # This would communicate with a companion Android service
        pass
    
    def _encode_image(self, img, format, quality):
        buffer = BytesIO()
        img.save(buffer, format=format.upper(), quality=quality, optimize=True)
        
        # Check size limit
        if buffer.tell() > 5 * 1024 * 1024:  # 5MB limit
            # Reduce quality or resize
            img.thumbnail((1920, 1080), Image.LANCZOS)
            buffer = BytesIO()
            img.save(buffer, format=format.upper(), quality=quality-10, optimize=True)
            
        return {
            "status": "success",
            "data": {
                "image": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                "format": format,
                "dimensions": {"width": img.width, "height": img.height},
                "platform": self.platform
            }
        }
```

### 2. Camera Capture Tool
**Function**: `capture_camera_photo`  
**Type**: Synchronous  
**Parameters**:
- `device_type` (str): "windows", "android_phone", "android_tablet"
- `device_id` (int): Camera device index (default: 0)
- `camera_facing` (str): For Android - "front" or "back" (default: "back")
- `format` (str): Output format - "png" or "jpeg" (default: "jpeg")
- `resolution` (dict, optional): {"width", "height"} for custom resolution

**Technical Solution by Platform**:

**Windows PC**:
- **Library**: `opencv-python` (cv2) for camera access
- **Implementation**: DirectShow backend on Windows
- **Device Discovery**: Enumerate available cameras up to index 10

**Android (Motorola/Lenovo)**:
- **Library**: `adb` with Camera2 API commands or companion app
- **Implementation**: Intent-based capture via ADB or Camera2 API
- **Front/Back Selection**: Camera facing parameter support

```python
# Unified implementation approach
import cv2
import subprocess
import base64
from io import BytesIO
from PIL import Image

class CameraCaptureAdapter:
    def capture_camera_photo(self, device_type="auto", device_id=0, 
                            camera_facing="back", format="jpeg", resolution=None):
        if device_type == "windows":
            return self._capture_windows_camera(device_id, format, resolution)
        elif device_type.startswith("android"):
            return self._capture_android_camera(camera_facing, format, resolution)
    
    def _capture_windows_camera(self, device_id, format, resolution):
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            return {
                "status": "error",
                "error": {"code": "CAMERA_NOT_FOUND", "message": f"Camera {device_id} not available"}
            }
        
        # Set resolution if specified
        if resolution:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution["width"])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution["height"])
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {
                "status": "error",
                "error": {"code": "CAPTURE_FAILED", "message": "Failed to capture frame"}
            }
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        return self._encode_camera_image(img, format, device_id)
    
    def _capture_android_camera(self, camera_facing, format, resolution):
        # Method 1: Using ADB to trigger camera intent
        try:
            camera_id = "0" if camera_facing == "back" else "1"
            
            # Launch camera and capture
            subprocess.run([
                "adb", "shell", "am", "start", 
                "-a", "android.media.action.IMAGE_CAPTURE",
                "--ei", "android.intent.extras.CAMERA_FACING", camera_id
            ])
            
            # Wait for capture and retrieve latest photo
            # This requires additional logic to monitor media store
            
            # Alternative: Use companion app with Camera2 API
            result = subprocess.run([
                "adb", "shell", "am", "broadcast",
                "-a", "com.quantum.perception.CAPTURE_PHOTO",
                "--es", "camera", camera_facing,
                "--es", "format", format
            ], capture_output=True, text=True)
            
            # Parse result and get image data
            # Image would be transmitted via ADB or network
            
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "ANDROID_CAMERA_ERROR", "message": str(e)}
            }
    
    def _encode_camera_image(self, img, format, device_id):
        buffer = BytesIO()
        img.save(buffer, format=format.upper(), quality=85, optimize=True)
        
        return {
            "status": "success",
            "data": {
                "image": base64.b64encode(buffer.getvalue()).decode('utf-8'),
                "format": format,
                "device_id": device_id,
                "dimensions": {"width": img.width, "height": img.height}
            }
        }
```

### 3. Audio Stream Tool
**Function**: `capture_audio_stream`  
**Type**: Asynchronous  
**Parameters**:
- `device_type` (str): "windows", "android_phone", "android_tablet"
- `duration` (float): Seconds of audio to return (default: 5, max: 30)
- `sample_rate` (int): Sample rate in Hz (default: 44100)
- `format` (str): Output format - "wav" or "mp3" (default: "wav")
- `source` (str): For Android - "mic" or "system" (default: "mic")

**Technical Solution by Platform**:

**Windows PC**:
- **Library**: `python-sounddevice` for audio capture
- **Buffer**: 30-second rolling circular buffer
- **Implementation**: Callback-based streaming with queue

**Android (Motorola/Lenovo)**:
- **Library**: `adb` with AudioRecord API or MediaRecorder
- **Implementation**: Background service with audio streaming
- **Permissions**: RECORD_AUDIO permission required

```python
# Unified implementation approach
import sounddevice as sd
import numpy as np
import queue
import threading
from scipy.io import wavfile
import io
import base64
import subprocess

class AudioStreamAdapter:
    def __init__(self, sample_rate=44100, buffer_seconds=30):
        self.sample_rate = sample_rate
        self.buffer_size = sample_rate * buffer_seconds
        self.platform = self._detect_platform()
        
        if self.platform == "windows":
            self._init_windows_audio()
        elif self.platform.startswith("android"):
            self._init_android_audio()
    
    def _init_windows_audio(self):
        self.audio_buffer = np.zeros((self.buffer_size, 1))
        self.write_index = 0
        self.lock = threading.Lock()
        
        # Start continuous recording for Windows
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=self._audio_callback_windows
        )
        self.stream.start()
    
    def _audio_callback_windows(self, indata, frames, time, status):
        """Callback for continuous audio streaming on Windows"""
        with self.lock:
            chunk_size = len(indata)
            if self.write_index + chunk_size <= self.buffer_size:
                self.audio_buffer[self.write_index:self.write_index + chunk_size] = indata
            else:
                # Wrap around
                overflow = (self.write_index + chunk_size) - self.buffer_size
                self.audio_buffer[self.write_index:] = indata[:-overflow]
                self.audio_buffer[:overflow] = indata[-overflow:]
            
            self.write_index = (self.write_index + chunk_size) % self.buffer_size
    
    def _init_android_audio(self):
        # Initialize Android audio capture via companion app
        # The app runs a background service with AudioRecord
        try:
            subprocess.run([
                "adb", "shell", "am", "startservice",
                "-n", "com.quantum.perception/.AudioCaptureService"
            ])
        except:
            pass
    
    async def capture_audio_stream(self, device_type="auto", duration=5, source="mic"):
        """Extract audio from buffer based on platform"""
        if device_type == "windows" or (device_type == "auto" and self.platform == "windows"):
            return await self._capture_windows_audio(duration)
        elif device_type.startswith("android") or (device_type == "auto" and self.platform.startswith("android")):
            return await self._capture_android_audio(duration, source)
    
    async def _capture_windows_audio(self, duration):
        """Extract last N seconds from Windows circular buffer"""
        samples_needed = int(self.sample_rate * min(duration, 30))
        
        with self.lock:
            # Get last N seconds from circular buffer
            if self.write_index >= samples_needed:
                audio_data = self.audio_buffer[self.write_index - samples_needed:self.write_index]
            else:
                # Wrap around case
                audio_data = np.concatenate([
                    self.audio_buffer[-(samples_needed - self.write_index):],
                    self.audio_buffer[:self.write_index]
                ])
        
        return self._encode_audio(audio_data, duration)
    
    async def _capture_android_audio(self, duration, source):
        """Capture audio from Android device"""
        try:
            # Request audio data from companion service
            result = subprocess.run([
                "adb", "shell", "am", "broadcast",
                "-a", "com.quantum.perception.GET_AUDIO",
                "--ei", "duration", str(int(duration * 1000)),  # milliseconds
                "--es", "source", source
            ], capture_output=True, text=True)
            
            # Retrieve audio file via ADB pull
            subprocess.run(["adb", "pull", "/sdcard/quantum_audio.wav", "temp_audio.wav"])
            
            # Load and encode
            rate, audio_data = wavfile.read("temp_audio.wav")
            return self._encode_audio(audio_data, duration)
            
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "ANDROID_AUDIO_ERROR", "message": str(e)}
            }
    
    def _encode_audio(self, audio_data, duration):
        """Encode audio data to WAV format"""
        wav_buffer = io.BytesIO()
        wavfile.write(wav_buffer, self.sample_rate, audio_data)
        wav_buffer.seek(0)
        
        return {
            "status": "success",
            "data": {
                "audio": base64.b64encode(wav_buffer.read()).decode('utf-8'),
                "format": "wav",
                "duration": duration,
                "sample_rate": self.sample_rate,
                "platform": self.platform
            }
        }
```

### 4. System Status Tool
**Function**: `get_system_status`  
**Type**: Synchronous  
**Parameters**:
- `device_type` (str): "windows", "android_phone", "android_tablet"
- `metrics` (list): Specific metrics to return (default: ["cpu", "memory", "battery", "network"])
- `include_processes` (bool): Include top processes - Windows only (default: False)
- `process_count` (int): Number of top processes (default: 5)

**Technical Solution by Platform**:

**Windows PC**:
- **Library**: `psutil` for comprehensive system monitoring
- **Metrics**: Full CPU, memory, battery, network, process information
- **Performance**: <100ms response time

**Android (Motorola/Lenovo)**:
- **Library**: `adb` with Android system commands
- **Metrics**: Limited to battery, memory, basic CPU info
- **Limitations**: Process info restricted due to Android privacy

```python
# Unified implementation approach
import psutil
import subprocess
import json
from datetime import datetime

class SystemMonitorAdapter:
    def __init__(self):
        self.platform = self._detect_platform()
        self.cache = {}
        self.cache_ttl = 5  # seconds
    
    def get_system_status(self, device_type="auto", metrics=None, 
                         include_processes=False, process_count=5):
        if device_type == "auto":
            device_type = self.platform
            
        if metrics is None:
            metrics = ["cpu", "memory", "battery", "network"]
        
        if device_type == "windows":
            return self._get_windows_status(metrics, include_processes, process_count)
        elif device_type.startswith("android"):
            return self._get_android_status(metrics)
    
    def _get_windows_status(self, metrics, include_processes, process_count):
        """Get comprehensive Windows system status"""
        result = {"status": "success", "data": {}}
        
        if "cpu" in metrics:
            result["data"]["cpu"] = {
                "usage_percent": psutil.cpu_percent(interval=1),
                "core_count": psutil.cpu_count(logical=True),
                "frequency_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
            }
        
        if "memory" in metrics:
            mem = psutil.virtual_memory()
            result["data"]["memory"] = {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "used_gb": mem.used / (1024**3),
                "percent": mem.percent
            }
        
        if "battery" in metrics:
            battery = psutil.sensors_battery()
            if battery:
                result["data"]["battery"] = {
                    "percent": battery.percent,
                    "plugged": battery.power_plugged,
                    "time_remaining": battery.secsleft if battery.secsleft != -1 else None
                }
        
        if "network" in metrics:
            net = psutil.net_io_counters()
            result["data"]["network"] = {
                "bytes_sent": net.bytes_sent,
                "bytes_recv": net.bytes_recv,
                "packets_sent": net.packets_sent,
                "packets_recv": net.packets_recv
            }
        
        if include_processes:
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            processes.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
            result["data"]["top_processes"] = processes[:process_count]
        
        result["data"]["timestamp"] = datetime.utcnow().isoformat()
        result["data"]["platform"] = "windows"
        return result
    
    def _get_android_status(self, metrics):
        """Get Android system status via ADB"""
        result = {"status": "success", "data": {}}
        
        try:
            if "battery" in metrics:
                # Get battery status
                battery_output = subprocess.run(
                    ["adb", "shell", "dumpsys", "battery"],
                    capture_output=True, text=True
                ).stdout
                
                battery_info = {}
                for line in battery_output.split('\n'):
                    if 'level:' in line:
                        battery_info['percent'] = int(line.split(':')[1].strip())
                    elif 'AC powered:' in line:
                        battery_info['ac_powered'] = line.split(':')[1].strip() == 'true'
                    elif 'USB powered:' in line:
                        battery_info['usb_powered'] = line.split(':')[1].strip() == 'true'
                
                battery_info['plugged'] = battery_info.get('ac_powered', False) or battery_info.get('usb_powered', False)
                result["data"]["battery"] = battery_info
            
            if "memory" in metrics:
                # Get memory info
                meminfo = subprocess.run(
                    ["adb", "shell", "cat", "/proc/meminfo"],
                    capture_output=True, text=True
                ).stdout
                
                mem_total = 0
                mem_free = 0
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        mem_total = int(line.split()[1]) / (1024 * 1024)  # Convert to GB
                    elif 'MemAvailable:' in line:
                        mem_free = int(line.split()[1]) / (1024 * 1024)
                
                result["data"]["memory"] = {
                    "total_gb": mem_total,
                    "available_gb": mem_free,
                    "used_gb": mem_total - mem_free,
                    "percent": ((mem_total - mem_free) / mem_total * 100) if mem_total > 0 else 0
                }
            
            if "cpu" in metrics:
                # Get CPU info
                cpu_info = subprocess.run(
                    ["adb", "shell", "cat", "/proc/cpuinfo"],
                    capture_output=True, text=True
                ).stdout
                
                core_count = cpu_info.count("processor")
                
                # Get current CPU usage
                top_output = subprocess.run(
                    ["adb", "shell", "top", "-n", "1"],
                    capture_output=True, text=True
                ).stdout
                
                # Parse CPU usage from top command
                cpu_usage = 0
                for line in top_output.split('\n'):
                    if '%cpu' in line.lower():
                        # Extract total CPU usage
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if '%cpu' in part.lower() and i > 0:
                                try:
                                    cpu_usage = float(parts[i-1])
                                except:
                                    pass
                
                result["data"]["cpu"] = {
                    "usage_percent": cpu_usage,
                    "core_count": core_count
                }
            
            if "network" in metrics:
                # Get network statistics
                net_stats = subprocess.run(
                    ["adb", "shell", "cat", "/proc/net/dev"],
                    capture_output=True, text=True
                ).stdout
                
                # Parse network stats (simplified)
                total_recv = 0
                total_sent = 0
                for line in net_stats.split('\n'):
                    if ':' in line and ('wlan' in line or 'rmnet' in line):
                        parts = line.split()
                        if len(parts) >= 10:
                            total_recv += int(parts[1])
                            total_sent += int(parts[9])
                
                result["data"]["network"] = {
                    "bytes_recv": total_recv,
                    "bytes_sent": total_sent
                }
            
            result["data"]["timestamp"] = datetime.utcnow().isoformat()
            result["data"]["platform"] = "android"
            
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "ANDROID_STATUS_ERROR", "message": str(e)}
            }
        
        return result
```

### 5. Device Location Tool
**Function**: `get_device_location`  
**Type**: Synchronous  
**Parameters**:
- `device_type` (str): "windows", "android_phone", "android_tablet"
- `method` (str): Location method - "ip", "wifi", "gps" (default: "ip")
- `accuracy` (str): Accuracy level - "city", "region", "country" (default: "city")

**Technical Solution by Platform**:

**Windows PC**:
- **Library**: `geocoder` for IP-based location, `wmi` for WiFi triangulation
- **Fallback**: IP geolocation when GPS unavailable
- **Privacy**: Configurable accuracy levels

**Android (Motorola/Lenovo)**:
- **Library**: `adb` with Location Manager commands
- **Methods**: GPS, Network, or IP-based location
- **Permissions**: ACCESS_FINE_LOCATION required for GPS

```python
# Unified implementation approach
import geocoder
import subprocess
import json

class LocationAdapter:
    def get_device_location(self, device_type="auto", method="ip", accuracy="city"):
        if device_type == "windows" or (device_type == "auto" and self._is_windows()):
            return self._get_windows_location(method, accuracy)
        elif device_type.startswith("android") or (device_type == "auto" and self._is_android()):
            return self._get_android_location(method, accuracy)
    
    def _get_windows_location(self, method, accuracy):
        """Get location on Windows PC"""
        try:
            if method == "ip":
                # IP-based geolocation
                g = geocoder.ip('me')
                
                if g.ok:
                    location_data = {
                        "latitude": g.latlng[0],
                        "longitude": g.latlng[1],
                        "accuracy": accuracy,
                        "method": "ip"
                    }
                    
                    if accuracy in ["city", "region", "country"]:
                        location_data["city"] = g.city
                    if accuracy in ["region", "country"]:
                        location_data["region"] = g.state
                    if accuracy == "country":
                        location_data["country"] = g.country
                    
                    return {
                        "status": "success",
                        "data": location_data
                    }
            
            elif method == "wifi":
                # WiFi-based location using Windows Location API
                # Requires additional setup with Windows Location Services
                try:
                    import winrt.windows.devices.geolocation as geolocation
                    locator = geolocation.Geolocator()
                    pos = locator.get_geoposition()
                    
                    return {
                        "status": "success",
                        "data": {
                            "latitude": pos.coordinate.latitude,
                            "longitude": pos.coordinate.longitude,
                            "accuracy": pos.coordinate.accuracy,
                            "method": "wifi"
                        }
                    }
                except:
                    # Fallback to IP
                    return self._get_windows_location("ip", accuracy)
                    
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "LOCATION_ERROR", "message": str(e)}
            }
    
    def _get_android_location(self, method, accuracy):
        """Get location on Android device"""
        try:
            if method == "gps":
                # Get GPS location via ADB
                location_output = subprocess.run(
                    ["adb", "shell", "dumpsys", "location"],
                    capture_output=True, text=True
                ).stdout
                
                # Parse last known location
                for line in location_output.split('\n'):
                    if 'last location=' in line.lower():
                        # Extract coordinates from string like "Location[gps 37.123,-122.456 ...]"
                        import re
                        coords = re.findall(r'[-]?\d+\.\d+', line)
                        if len(coords) >= 2:
                            return {
                                "status": "success",
                                "data": {
                                    "latitude": float(coords[0]),
                                    "longitude": float(coords[1]),
                                    "method": "gps",
                                    "accuracy": accuracy,
                                    "platform": "android"
                                }
                            }
            
            elif method == "network":
                # Get network-based location
                # This requires companion app or Google Play Services
                result = subprocess.run([
                    "adb", "shell", "am", "broadcast",
                    "-a", "com.quantum.perception.GET_LOCATION",
                    "--es", "method", "network"
                ], capture_output=True, text=True)
                
                # Parse response
                if "latitude" in result.stdout:
                    import re
                    lat = re.search(r'latitude[:\s]+([-]?\d+\.\d+)', result.stdout)
                    lon = re.search(r'longitude[:\s]+([-]?\d+\.\d+)', result.stdout)
                    
                    if lat and lon:
                        return {
                            "status": "success",
                            "data": {
                                "latitude": float(lat.group(1)),
                                "longitude": float(lon.group(1)),
                                "method": "network",
                                "accuracy": accuracy,
                                "platform": "android"
                            }
                        }
            
            # Fallback to IP-based location
            if method == "ip" or True:  # Always fallback
                # Get device IP and use geocoding
                ip_output = subprocess.run(
                    ["adb", "shell", "ip", "addr", "show"],
                    capture_output=True, text=True
                ).stdout
                
                # Use geocoder with device's public IP
                g = geocoder.ip('me')  # This would get PC's IP, need device's public IP
                
                if g.ok:
                    return {
                        "status": "success",
                        "data": {
                            "latitude": g.latlng[0],
                            "longitude": g.latlng[1],
                            "method": "ip",
                            "accuracy": accuracy,
                            "city": g.city,
                            "country": g.country,
                            "platform": "android"
                        }
                    }
                    
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "ANDROID_LOCATION_ERROR", "message": str(e)}
            }
```

### 6. Application Context Tool
**Function**: `get_application_context`  
**Type**: Synchronous  
**Parameters**:
- `device_type` (str): "windows", "android_phone", "android_tablet"
- `include_all` (bool): Include all processes or just user applications (default: False)
- `sort_by` (str): Sort criteria - "cpu", "memory", "name" (default: "cpu")
- `limit` (int): Maximum number of applications to return (default: 10)

**Technical Solution by Platform**:

**Windows PC**:
- **Library**: `psutil` for comprehensive process monitoring
- **Features**: Full process list with CPU, memory, running time
- **Context Hints**: Detect work vs leisure mode

**Android (Motorola/Lenovo)**:
- **Library**: `adb` with ActivityManager commands
- **Limitations**: Only foreground app and recent tasks accessible
- **Privacy**: Cannot access other apps' CPU/memory usage since Android 7.0

```python
# Unified implementation approach
import psutil
import subprocess
import time
from datetime import datetime

class ApplicationContextAdapter:
    def get_application_context(self, device_type="auto", include_all=False, 
                               sort_by="cpu", limit=10):
        if device_type == "windows" or (device_type == "auto" and self._is_windows()):
            return self._get_windows_context(include_all, sort_by, limit)
        elif device_type.startswith("android") or (device_type == "auto" and self._is_android()):
            return self._get_android_context(limit)
    
    def _get_windows_context(self, include_all, sort_by, limit):
        """Get Windows application context"""
        try:
            applications = []
            
            # Get all running processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 
                                           'memory_percent', 'create_time', 
                                           'exe', 'status']):
                try:
                    pinfo = proc.info
                    
                    # Filter system processes unless include_all
                    if not include_all:
                        # Skip system/background processes
                        if pinfo['name'] in ['System', 'Registry', 'svchost.exe']:
                            continue
                        # Skip processes without window
                        if pinfo.get('status') == 'zombie':
                            continue
                    
                    # Calculate running time
                    create_time = pinfo.get('create_time', 0)
                    if create_time:
                        running_time = time.time() - create_time
                    else:
                        running_time = 0
                    
                    app_info = {
                        "name": pinfo['name'],
                        "pid": pinfo['pid'],
                        "cpu_percent": pinfo.get('cpu_percent', 0),
                        "memory_percent": pinfo.get('memory_percent', 0),
                        "running_time_seconds": running_time,
                        "status": pinfo.get('status', 'unknown')
                    }
                    
                    # Try to get executable path
                    if pinfo.get('exe'):
                        app_info["path"] = pinfo['exe']
                    
                    applications.append(app_info)
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Sort applications
            if sort_by == "cpu":
                applications.sort(key=lambda x: x['cpu_percent'], reverse=True)
            elif sort_by == "memory":
                applications.sort(key=lambda x: x['memory_percent'], reverse=True)
            elif sort_by == "name":
                applications.sort(key=lambda x: x['name'].lower())
            
            # Apply limit
            applications = applications[:limit]
            
            # Determine user context (work vs leisure)
            context_hints = {
                "development": any(app['name'].lower() in ['code.exe', 'devenv.exe', 'pycharm.exe', 'idea.exe'] 
                                 for app in applications[:5]),
                "office_work": any(app['name'].lower() in ['winword.exe', 'excel.exe', 'powerpnt.exe', 'outlook.exe'] 
                                 for app in applications[:5]),
                "communication": any(app['name'].lower() in ['teams.exe', 'slack.exe', 'zoom.exe', 'discord.exe'] 
                                   for app in applications[:5]),
                "browsing": any(app['name'].lower() in ['chrome.exe', 'firefox.exe', 'edge.exe', 'brave.exe'] 
                              for app in applications[:5]),
                "media": any(app['name'].lower() in ['spotify.exe', 'vlc.exe', 'netflix.exe', 'youtube.exe'] 
                           for app in applications[:5])
            }
            
            return {
                "status": "success",
                "data": {
                    "applications": applications,
                    "total_count": len(applications),
                    "context_hints": context_hints,
                    "platform": "windows",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "WINDOWS_CONTEXT_ERROR", "message": str(e)}
            }
    
    def _get_android_context(self, limit):
        """Get Android application context"""
        try:
            # Get current foreground app
            foreground = subprocess.run(
                ["adb", "shell", "dumpsys", "window", "windows", "|", 
                 "grep", "-E", "mCurrentFocus|mFocusedApp"],
                capture_output=True, text=True, shell=True
            ).stdout
            
            current_app = "Unknown"
            for line in foreground.split('\n'):
                if 'mCurrentFocus' in line or 'mFocusedApp' in line:
                    # Extract package name
                    import re
                    match = re.search(r'(\w+\.\w+\.\w+)/', line)
                    if match:
                        current_app = match.group(1)
                        break
            
            # Get recent tasks
            recent_tasks = subprocess.run(
                ["adb", "shell", "dumpsys", "activity", "recents"],
                capture_output=True, text=True
            ).stdout
            
            applications = []
            
            # Parse recent tasks
            task_lines = recent_tasks.split('\n')
            for i, line in enumerate(task_lines):
                if 'Recent #' in line:
                    # Extract app info from next few lines
                    app_info = {"name": "Unknown", "package": ""}
                    
                    for j in range(i, min(i+5, len(task_lines))):
                        if 'intent=' in task_lines[j]:
                            import re
                            match = re.search(r'cmp=([^/\s]+)', task_lines[j])
                            if match:
                                app_info["package"] = match.group(1)
                                app_info["name"] = match.group(1).split('.')[-1]
                        elif 'realActivity=' in task_lines[j]:
                            import re
                            match = re.search(r'realActivity=([^/\s]+)', task_lines[j])
                            if match:
                                app_info["package"] = match.group(1)
                                app_info["name"] = match.group(1).split('.')[-1]
                    
                    if app_info["package"]:
                        app_info["is_foreground"] = app_info["package"] == current_app
                        applications.append(app_info)
                    
                    if len(applications) >= limit:
                        break
            
            # Get running services (limited info)
            services = subprocess.run(
                ["adb", "shell", "dumpsys", "activity", "services"],
                capture_output=True, text=True
            ).stdout
            
            # Count services by package
            service_counts = {}
            for line in services.split('\n'):
                if 'ServiceRecord' in line:
                    import re
                    match = re.search(r'{[^}]*([a-z]+\.[a-z]+\.[a-z]+)', line)
                    if match:
                        pkg = match.group(1)
                        service_counts[pkg] = service_counts.get(pkg, 0) + 1
            
            # Add service count to applications
            for app in applications:
                app["service_count"] = service_counts.get(app["package"], 0)
            
            # Determine context hints based on package names
            context_hints = {
                "communication": any('whatsapp' in app["package"].lower() or 
                                   'telegram' in app["package"].lower() or
                                   'messenger' in app["package"].lower() 
                                   for app in applications),
                "browsing": any('chrome' in app["package"].lower() or 
                              'firefox' in app["package"].lower() or
                              'browser' in app["package"].lower() 
                              for app in applications),
                "media": any('youtube' in app["package"].lower() or 
                           'spotify' in app["package"].lower() or
                           'netflix' in app["package"].lower() 
                           for app in applications),
                "social": any('facebook' in app["package"].lower() or 
                            'instagram' in app["package"].lower() or
                            'twitter' in app["package"].lower() 
                            for app in applications),
                "gaming": any('game' in app["package"].lower() or 
                            'play' in app["package"].lower() 
                            for app in applications)
            }
            
            return {
                "status": "success",
                "data": {
                    "current_app": current_app,
                    "recent_applications": applications,
                    "total_count": len(applications),
                    "context_hints": context_hints,
                    "platform": "android",
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": {"code": "ANDROID_CONTEXT_ERROR", "message": str(e)}
            }
```

## Project Timeline: August 6-31, 2025

### Planning Phase (Aug 6-8)

**Day 1 (Aug 6): Kickoff**
- Team kickoff meeting
- Environment setup and dependency installation
- Review tool specifications and requirements
- Set up development infrastructure (Git, CI/CD, testing framework)
- **Deliverable**: Development environment ready

**Day 2-3 (Aug 7-8): Architecture & Game Plan**
- Finalize tool interface design and base classes
- Design platform adapter pattern for Windows/Android
- Create error handling patterns and response schemas
- Set up testing harness and mock framework
- Establish coding standards and review process
- **Deliverable**: Solidified architecture and development plan

### Week 1: Core Development (Aug 11-15)

**Day 4-5 (Aug 11-12): Screenshot & Camera Tools**
- Implement screenshot capture for Windows (mss library)
- Implement screenshot capture for Android (ADB)
- Develop camera tool for Windows (OpenCV)
- Develop camera tool for Android (ADB intents)
- Create image encoding and size limit handling
- **Deliverable**: Working screenshot and camera tools

**Day 6-7 (Aug 13-14): Audio & System Tools**
- Build audio streaming with 30-second circular buffer (Windows)
- Implement Android audio capture via ADB
- Create system monitoring tool for both platforms
- Implement cross-platform adapter pattern
- **Deliverable**: Audio and system monitoring tools

**Day 8 (Aug 15): Location & Application Context**
- Develop location tool (IP-based and platform-specific)
- Build application context tool for Windows (psutil)
- Build application context tool for Android (ADB)
- Integration testing of all tools
- **Deliverable**: All 6 tools with basic functionality

### Week 2: Robustness & Optimization (Aug 18-22)

**Day 9-10 (Aug 18-19): Error Handling & Edge Cases**
- Implement comprehensive error handling
- Handle permission denied scenarios
- Address hardware not available cases
- Implement network timeouts and retries
- Ensure proper resource cleanup
- **Deliverable**: Robust error handling across all tools

**Day 11-12 (Aug 20-21): Performance Optimization**
- Profile memory usage and optimize buffers
- Optimize response times (<2 second target)
- Implement caching strategies
- Reduce ADB overhead for Android
- Performance benchmarking
- **Deliverable**: Optimized tools meeting performance targets

**Day 13 (Aug 22): Cross-Platform Testing**
- Test all tools on Windows PC
- Test all tools with Android devices (Motorola phone, Lenovo tablet)
- Validate cross-platform adapter pattern
- Document platform-specific limitations
- **Deliverable**: Fully tested cross-platform tools

### Week 3: Integration & Polish (Aug 25-29)

**Day 14-15 (Aug 25-26): Integration Package**
- Create unified package structure
- Develop MCP adapter examples
- Create Gemini function calling examples
- Build comprehensive test suite
- Write deployment scripts
- **Deliverable**: Integration-ready package

**Day 16-17 (Aug 27-28): Documentation & Polish**
- Complete API documentation with schemas
- Write integration guides for MCP team
- Create usage examples and sample code
- Prepare troubleshooting guide
- Final performance validation
- **Deliverable**: Complete documentation package

**Day 18 (Aug 29): Final Testing & Handoff Prep**
- End-to-end testing of complete system
- Prepare handoff materials
- Create deployment instructions
- Final bug fixes
- Package all deliverables
- **Deliverable**: Production-ready tools

### Buffer Weekend (Aug 30-31)

**Saturday (Aug 30)**
- Emergency bug fixes if needed
- Documentation updates
- Final testing round

**Sunday (Aug 31)**
- Final review
- Prepare Monday morning delivery
- **Deliverable**: Complete package ready for handoff

## Key Milestones

- **Aug 8**: Architecture finalized
- **Aug 15**: All 6 tools functional
- **Aug 22**: Performance optimized and tested
- **Aug 29**: Production-ready
- **Aug 31**: Final delivery

## Development Approach

- **Pair Programming**: Windows + Android implementation in parallel
- **Daily Standups**: Quick sync on progress and blockers
- **Continuous Integration**: Test on both platforms daily
- **Iterative Development**: Basic functionality first, then optimize

## Success Criteria

- All 6 tools working on Windows PC
- All 6 tools working on Android devices (via ADB or companion app)
- Response times <2 seconds
- Memory usage <100MB
- 80% test coverage
- Complete documentation and integration guides
  
## Technical Stack

### Core Dependencies
```python
# requirements.txt
# Windows-specific libraries
mss>=9.0.1               # Screenshot capture (Windows)
opencv-python>=4.9.0     # Camera access (Windows)
sounddevice>=0.4.6       # Audio streaming (Windows)
psutil>=5.9.8           # System monitoring (Cross-platform)
pywin32>=306            # Windows API access (Windows only)
pillow>=10.2.0          # Image processing
scipy>=1.12.0           # Audio processing
geocoder>=1.38.1        # Location services
numpy>=1.26.3           # Array operations
pydantic>=2.5.3         # Data validation

# Android interaction libraries
adb-shell>=0.4.3        # ADB communication
pure-python-adb>=0.3.0  # Alternative ADB library

# Optional for enhanced Android support
pyjnius>=1.4.2          # Java bridge for Android (if using Kivy)
kivy>=2.2.1             # Cross-platform framework (optional)
```

### Development Dependencies
```python
# requirements-dev.txt
pytest>=8.0.0        # Testing framework
pytest-asyncio>=0.23.0  # Async test support
pytest-cov>=4.1.0    # Coverage reporting
black>=24.1.0        # Code formatting
mypy>=1.8.0          # Type checking
sphinx>=7.2.0        # Documentation
```

## Delivery Package Structure

```
perception-tools/
├── src/
│   ├── __init__.py
│   ├── base.py              # Base classes and interfaces
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── screenshot.py
│   │   ├── camera.py
│   │   ├── audio.py
│   │   ├── system.py
│   │   ├── location.py
│   │   └── window.py
│   ├── utils/
│   │   ├── encoding.py      # Base64, compression
│   │   ├── validation.py    # Input validation
│   │   └── errors.py        # Error definitions
│   └── schemas/
│       └── tool_schemas.json  # JSON schemas for all tools
├── tests/
│   ├── unit/                # Unit tests per tool
│   ├── integration/         # Integration tests
│   └── fixtures/            # Test data
├── docs/
│   ├── api/                 # API documentation
│   ├── integration/         # MCP/Gemini integration guides
│   └── examples/            # Usage examples
├── examples/
│   ├── test_harness.py      # CLI testing tool
│   ├── mcp_adapter.py       # MCP integration example
│   └── gemini_adapter.py    # Gemini integration example
├── requirements.txt
├── setup.py
└── README.md
```

## Success Metrics

### Success Metrics
- Screenshot capture: <500ms for full screen
- Camera capture: <1 second including initialization
- Audio retrieval: <200ms for 5-second chunk
- System status: <100ms for basic metrics
- Memory usage: <100MB during normal operation
- Test coverage: >80% for core functionality

## Risk Mitigation

### Technical Risks
1. **Audio buffer overflow**: Implement ring buffer with automatic wraparound
2. **Camera permission denied**: Graceful fallback with clear error messages
3. **Screenshot of secure windows**: Handle black screen scenarios
4. **Memory leaks**: Implement proper resource cleanup in all tools

### Schedule Risks
1. **Windows API complexity**: Front-load Windows-specific development
2. **Integration delays**: Provide early tool stubs for MCP team testing
3. **Performance issues**: Daily performance testing from Week 2

### Mitigation Strategies
- Daily standups for early issue detection
- Pair programming for complex Windows API integration
- Continuous integration with automated testing
- Regular sync with MCP team (twice weekly)

## Handoff Checklist

### Code Deliverables
- [ ] All 6 tools implemented and tested
- [ ] Unit tests with >90% coverage
- [ ] Integration test suite
- [ ] Performance benchmarks documented

### Documentation Deliverables
- [ ] API reference documentation
- [ ] JSON schemas for all tools
- [ ] MCP integration guide
- [ ] Gemini function calling examples
- [ ] Troubleshooting guide

### Technical Deliverables
- [ ] Python package (pip installable)
- [ ] Requirements.txt with pinned versions
- [ ] Docker container (optional)
- [ ] CI/CD pipeline configuration

### Communication
- [ ] Final demo to QT Core Orchestration team
- [ ] Handoff meeting scheduled
- [ ] Support contact established
- [ ] Known issues documented
