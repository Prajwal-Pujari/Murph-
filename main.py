import asyncio
import json
import aiohttp
import torch
import whisper
import pyttsx3
import chromadb
import tempfile
import io
import os
import uuid
import socket
import webbrowser
import inspect
import re
from datetime import datetime
from contextlib import asynccontextmanager
import pyautogui
import pygetwindow as gw
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor
import queue

import winreg
import psutil

from piper.voice import PiperVoice
from gtts import gTTS
from pytube import Search
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# Browser Control Imports
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# --- Global State & Lifespan ---
ml_models = {}
browser_driver = None
current_humor_level = 85
tts_executor = ThreadPoolExecutor(max_workers=2)
audio_cache = {}
active_tabs = {'media': None, 'search': None}

@asynccontextmanager
async def lifespan(app: FastAPI):
    global browser_driver
    
    print("🔹 Loading Whisper model...")
    ml_models["whisper_model"] = whisper.load_model("small.en")
    print("✅ Whisper model loaded.")

    print("🔹 Loading Offline TTS Voice (Piper)...")
    try:
        possible_paths = [
            "./models/en_US-hfc_male-medium.onnx",
            "./en_US-hfc_male-medium.onnx",
            "models/en_US-hfc_male-medium.onnx",
            "./models/en_US-lessac-medium.onnx",
            "./en_US-lessac-medium.onnx",
            "./models/en_US-ryan-high.onnx",
        ]
        
        onnx_file = None
        for path in possible_paths:
            json_path = path + ".json"
            if os.path.exists(path) and os.path.exists(json_path):
                onnx_file = path
                print(f"✅ Found Piper model at: {path}")
                print(f"✅ Found Piper config at: {json_path}")
                break
        
        if onnx_file:
            ml_models["piper_voice"] = PiperVoice.load(onnx_file)
            print("✅ Offline TTS Voice (Piper) loaded successfully!")
            print(f"🎤 Using voice: {onnx_file}")
        else:
            print("⚠️ Piper voice model not found in any expected location.")
            print("📥 Download from: https://github.com/rhasspy/piper/releases/")
            print("📁 You need BOTH files: .onnx AND .onnx.json in the 'models' folder")
            ml_models["piper_voice"] = None
            
    except Exception as e:
        print(f"⚠️ Could not load Piper TTS model: {e}")
        print(f"📝 Error details: {type(e).__name__}: {str(e)}")
        ml_models["piper_voice"] = None

    print("🔹 Initializing ChromaDB...")
    client = chromadb.PersistentClient(path="memory_db")
    ml_models["memory_collection"] = client.get_or_create_collection(name="conversations")
    print("✅ ChromaDB ready.")
    
    browser_driver = None
    print("✅ Browser control ready (will open when needed).")
    
    yield
    
    print("🔹 Shutting down...")
    if browser_driver:
        browser_driver.quit()
    tts_executor.shutdown(wait=False)
    audio_cache.clear()
    ml_models.clear()


# --- App Initialization, Middleware, and Settings ---
class ChatRequest(BaseModel):
    prompt: str
    
app = FastAPI(lifespan=lifespan)
origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
OLLAMA_TIMEOUT = 120


# --- FAST-PATH REGEX DETECTION (NEW) ---
FAST_PATH_PATTERNS = {
    'time': r'\b(what|tell me|what\'s|whats)\s+(time|the time)\b',
    'pause': r'\b(pause|stop)\s+(video|it|playback)\b',
    'play_video': r'\b(play|resume)\s+(video|it|playback)\b',
    'mute': r'\bmute\b',
    'unmute': r'\bunmute\b',
    'volume_up': r'\b(volume\s+up|louder|increase\s+volume)\b',
    'volume_down': r'\b(volume\s+down|quieter|decrease\s+volume)\b',
    'youtube_play': r'\bplay\s+(.+?)\s+(?:on\s+)?youtube\b',
    'youtube_search': r'\bsearch\s+(?:for\s+)?(.+?)\s+on\s+youtube\b',
    'open_website': r'\bopen\s+(?:website\s+)?(.+\.(?:com|org|net|io|co|gov|edu))\b',
    'navigate': r'\b(?:go to|navigate to)\s+(.+\.(?:com|org|net|io|co))\b',
    'weather': r'\b(?:weather|temperature)\s+(?:in\s+)?(\w+)\b',
    'next_video': r'\b(?:play\s+)?next\s+video\b',
    'scroll_down': r'\bscroll\s+down\b',
    'scroll_up': r'\bscroll\s+up\b',
    'read_page': r'\b(?:read|what\'s on)\s+(?:the\s+)?page\b',
    'close_tab': r'\bclose\s+(?:this\s+)?tab\b',
    'list_apps': r'\b(?:list|show|what)\s+(?:available\s+)?(?:apps|applications|programs)\b',
    'open_app': r'\bopen\s+([a-zA-Z0-9\s]+?)(?:\s+and|\s+then|$)',
}

def detect_fast_path(prompt: str):
    """Returns (tool_name, params) if fast-path detected, else (None, None)"""
    prompt_lower = prompt.lower()
    
    # Time
    if re.search(FAST_PATH_PATTERNS['time'], prompt_lower):
        return ('get_current_time', {})
    
    # Video controls
    if re.search(FAST_PATH_PATTERNS['pause'], prompt_lower):
        return ('control_video', {'action': 'pause'})
    
    if re.search(FAST_PATH_PATTERNS['play_video'], prompt_lower):
        return ('control_video', {'action': 'play'})
    
    if re.search(FAST_PATH_PATTERNS['mute'], prompt_lower):
        return ('control_video', {'action': 'mute'})
    
    if re.search(FAST_PATH_PATTERNS['unmute'], prompt_lower):
        return ('control_video', {'action': 'unmute'})
    
    if re.search(FAST_PATH_PATTERNS['volume_up'], prompt_lower):
        return ('control_video', {'action': 'volume_up'})
    
    if re.search(FAST_PATH_PATTERNS['volume_down'], prompt_lower):
        return ('control_video', {'action': 'volume_down'})
    
    # YouTube
    match = re.search(FAST_PATH_PATTERNS['youtube_play'], prompt_lower)
    if match:
        query = match.group(1).strip()
        return ('play_youtube_video', {'query': query})
    
    match = re.search(FAST_PATH_PATTERNS['youtube_search'], prompt_lower)
    if match:
        query = match.group(1).strip()
        return ('search_youtube', {'query': query})
    
    # Next video
    if re.search(FAST_PATH_PATTERNS['next_video'], prompt_lower):
        return ('play_next_video', {})
    
    # Website navigation
    match = re.search(FAST_PATH_PATTERNS['open_website'], prompt_lower)
    if match:
        url = match.group(1).strip()
        return ('navigate_to_url', {'url': url})
    
    match = re.search(FAST_PATH_PATTERNS['navigate'], prompt_lower)
    if match:
        url = match.group(1).strip()
        return ('navigate_to_url', {'url': url})
    
    # Weather
    match = re.search(FAST_PATH_PATTERNS['weather'], prompt_lower)
    if match:
        city = match.group(1).strip()
        return ('get_weather', {'city': city})
    
    # Page controls
    if re.search(FAST_PATH_PATTERNS['scroll_down'], prompt_lower):
        return ('scroll_page', {'direction': 'down', 'amount': 'medium'})
    
    if re.search(FAST_PATH_PATTERNS['scroll_up'], prompt_lower):
        return ('scroll_page', {'direction': 'up', 'amount': 'medium'})
    
    if re.search(FAST_PATH_PATTERNS['read_page'], prompt_lower):
        return ('read_page_content', {})
    
    if re.search(FAST_PATH_PATTERNS['close_tab'], prompt_lower):
        return ('close_current_tab', {})
    
    return (None, None)


# --- Hybrid TTS Logic (OPTIMIZED) ---
def is_online(host="8.8.8.8", port=53, timeout=2):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error as ex:
        return False

async def convert_text_to_audio_bytes(text: str) -> bytes:
    """OPTIMIZED: Faster TTS with caching and parallel processing"""
    
    cache_key = text[:100]
    if cache_key in audio_cache:
        print("⚡ Using cached audio")
        return audio_cache[cache_key]
    
    if ml_models.get("piper_voice"):
        print("🔌 Fast TTS (Piper)...")
        
        def _generate_piper_audio():
            try:
                audio_chunks = []
                
                for chunk in ml_models["piper_voice"].synthesize(text):
                    audio_chunks.append(chunk.audio_int16_bytes)
                
                if not audio_chunks:
                    print("❌ Piper returned no audio chunks")
                    return None
                
                audio_data = b''.join(audio_chunks)
                
                if len(audio_data) < 100:
                    print(f"❌ Piper generated too little data: {len(audio_data)} bytes")
                    return None
                
                print(f"✅ Generated {len(audio_data)} bytes of RAW PCM audio with Piper")
                
                import wave
                wav_buffer = io.BytesIO()
                
                with wave.open(wav_buffer, 'wb') as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    sample_rate = ml_models["piper_voice"].config.sample_rate
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(audio_data)
                
                wav_buffer.seek(0)
                final_audio = wav_buffer.read()
                
                print(f"✅ Created WAV file: {len(final_audio)} bytes")
                return final_audio
                    
            except Exception as e:
                print(f"❌ Piper TTS failed: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        audio_bytes = await run_in_threadpool(_generate_piper_audio)
        
        if audio_bytes:
            audio_cache[cache_key] = audio_bytes
            if len(audio_cache) > 20:
                audio_cache.pop(next(iter(audio_cache)))
            return audio_bytes
        else:
            print("⚠️ Piper failed completely, falling back to gTTS")
    else:
        print("⚠️ Piper voice model not loaded, using gTTS")
    
    if is_online():
        print("🌐 Fallback to gTTS...")
        def _generate_gtts_audio():
            try:
                tts = gTTS(text=text, lang='en', slow=False)
                audio_stream = io.BytesIO()
                tts.write_to_fp(audio_stream)
                audio_stream.seek(0)
                return audio_stream.read()
            except Exception as e:
                print(f"gTTS failed: {e}")
                return None
        
        audio_bytes = await run_in_threadpool(_generate_gtts_audio)
        if audio_bytes:
            audio_cache[cache_key] = audio_bytes
            return audio_bytes

    raise HTTPException(status_code=500, detail="All TTS methods failed")


# --- Browser Control Functions ---
def initialize_browser():
    global browser_driver
    if browser_driver:
        try:
            _ = browser_driver.window_handles
            return browser_driver
        except WebDriverException:
            print("⚠️ Browser was running but is no longer reachable. Re-initializing...")
            browser_driver = None

    try:
        service = Service(ChromeDriverManager().install())
    except Exception as e:
        error_message = f"❌ FATAL: Failed to initialize ChromeDriver service via webdriver-manager. Error: {e}"
        print(error_message)
        raise Exception(error_message)

    try:
        print("🔹 Attempting to connect to existing Chrome browser on port 9222...")
        chrome_options_remote = Options()
        chrome_options_remote.add_experimental_option("debuggerAddress", "localhost:9222")
        browser_driver = webdriver.Chrome(service=service, options=chrome_options_remote)
        print("✅ Successfully connected to existing Chrome browser.")
        return browser_driver
    except WebDriverException:
        print("⚠️ Could not connect to existing browser. This is normal if one isn't running in debug mode.")
        print("🔹 Proceeding to launch a new browser instance...")
        pass

    try:
        chrome_options_new = Options()
        chrome_options_new.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options_new.add_argument("--disable-extensions")
        chrome_options_new.add_experimental_option("excludeSwitches", ["enable-logging"])
        
        browser_driver = webdriver.Chrome(service=service, options=chrome_options_new)
        print("✅ Successfully launched and connected to a new Chrome browser.")
        return browser_driver
    except Exception as e:
        error_message = f"❌ FATAL: Failed to both connect and launch Chrome. The final launch attempt failed with error: {e}"
        print(error_message)
        browser_driver = None
        raise Exception(error_message)
    
def parse_history_document(doc_string: str):
    parts = doc_string.split(" | ")
    parsed_parts = []
    for part in parts:
        if ": " in part:
            key, value = part.split(": ", 1)
            parsed_parts.append({"type": key.strip(), "content": value.strip()})
        else:
            parsed_parts.append({"type": "Unknown", "content": part})
    return parsed_parts
    

def search_website(query: str, website_name: str):
    """OPTIMIZED: Faster search navigation."""
    print(f"Executing search_website: {query} on {website_name}")
    
    search_urls = {
        "wikipedia": "https://en.wikipedia.org/w/index.php?search=",
        "google": "https://www.google.com/search?q=",
        "youtube": "https://www.youtube.com/results?search_query=",
    }
    
    site_key = website_name.lower().strip()
    base_url = search_urls.get(site_key, search_urls["google"])
    url = base_url + query.replace(' ', '+')

    try:
        driver = switch_to_context_tab('search')
        driver.get(url)
        return f"Searching {website_name} for '{query}'."
    except Exception as e:
        return f"Error searching {website_name}: {str(e)}"

    

def scroll_page(direction: str = "down", amount: str = "medium"):
    print(f"Executing scroll_page: direction={direction}, amount={amount}")
    
    try:
        driver = initialize_browser()
    except Exception as e:
        return str(e)
    
    try:
        scroll_amounts = {
            "small": 300,
            "medium": 600,
            "large": 1000,
            "bottom": 999999,
            "top": -999999
        }
        
        pixels = scroll_amounts.get(amount.lower(), 600)
        if direction.lower() == "up":
            pixels = -abs(pixels)
        
        driver.execute_script(f"window.scrollBy(0, {pixels});")
        return f"Scrolled {direction} on the page."
    except Exception as e:
        print(f"Error scrolling: {e}")
        return f"Could not scroll the page: {str(e)}"

def read_page_content():
    print("Executing read_page_content")
    
    try:
        driver = initialize_browser()
    except Exception as e:
        return str(e)
    
    try:
        title = driver.title
        body_text = driver.find_element(By.TAG_NAME, "body").text
        summary = body_text[:500] + "..." if len(body_text) > 500 else body_text
        return f"Page title: {title}. Content: {summary}"
    except Exception as e:
        print(f"Error reading page: {e}")
        return f"Could not read the page content: {str(e)}"

def control_video(action: str):
    """OPTIMIZED: Instant video controls."""
    print(f"Executing control_video: {action}")
    
    try:
        driver = switch_to_context_tab('media')
    except Exception as e:
        return str(e)
    
    try:
        action = action.lower()
        body = driver.find_element(By.TAG_NAME, "body")
        
        actions_map = {
            "play": "k", "pause": "k",
            "mute": "m", "unmute": "m",
            "volume_up": Keys.ARROW_UP,
            "volume_down": Keys.ARROW_DOWN
        }
        
        key = actions_map.get(action)
        if key:
            body.send_keys(key)
            return f"{action.replace('_', ' ').title()} done."
        
        return f"Unknown action: {action}"
    except Exception as e:
        return f"Video control error: {str(e)}"
    
def reset_tab_contexts():
    """Call this when browser is closed to reset the tab tracking."""
    global active_tabs
    active_tabs = {'media': None, 'search': None}
    print("🔄 Tab contexts reset.")

def navigate_to_url(url: str):
    """OPTIMIZED: Fast URL navigation."""
    print(f"Executing navigate_to_url: {url}")
    
    try:
        driver = switch_to_context_tab('search')
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        driver.get(url)
        return f"Opened {url}."
    except Exception as e:
        return f"Couldn't open {url}: {str(e)}"

def play_different_video(query: str):
    """OPTIMIZED: Quick video switch."""
    print(f"Executing play_different_video: {query}")
    
    try:
        driver = switch_to_context_tab('media')
    except Exception as e:
        return str(e)
    
    try:
        driver.get(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
        
        first_video = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a#video-title"))
        )
        video_title = first_video.text or query
        first_video.click()
        
        return f"Now playing: {video_title}"
    except TimeoutException:
        return f"Couldn't find video for '{query}'."
    except Exception as e:
        return f"Error: {str(e)}"

def play_next_video():
    """OPTIMIZED: Quick next video."""
    print("Executing play_next_video")
    
    try:
        driver = switch_to_context_tab('media')
        next_btn = driver.find_element(By.CLASS_NAME, "ytp-next-button")
        next_btn.click()
        time.sleep(0.5)
        return f"Playing next: {driver.title.replace('- YouTube', '').strip()}"
    except NoSuchElementException:
        return "No next button found. Are you on a YouTube video?"
    except Exception as e:
        return f"Error: {str(e)}"

def get_current_tab_info():
    print("Executing get_current_tab_info")
    
    try:
        driver = initialize_browser()
    except Exception as e:
        return str(e)
    
    try:
        title = driver.title
        url = driver.current_url
        return f"Current tab: {title} at {url}"
    except Exception as e:
        print(f"Error getting tab info: {e}")
        return f"Could not get tab information: {str(e)}"

def close_current_tab():
    print("Executing close_current_tab")
    
    try:
        driver = initialize_browser()
    except Exception as e:
        return str(e)
    
    try:
        current_handle = driver.current_window_handle
        
        # Clean up context tracking if we're closing a tracked tab
        global active_tabs
        for context, handle in active_tabs.items():
            if handle == current_handle:
                active_tabs[context] = None
                print(f"🗑️ Cleared '{context}' tab reference.")
                break
        
        if len(driver.window_handles) == 1:
            driver.quit()
            global browser_driver
            browser_driver = None
            reset_tab_contexts()
            return "Last tab closed. Browser has been shut down."

        driver.close()
        driver.switch_to.window(driver.window_handles[-1])
        return "Tab closed. Switched to the previous tab."
    except Exception as e:
        print(f"Error closing tab: {e}")
        return f"Could not close the tab: {str(e)}"
    
def switch_to_context_tab(context_name: str):
    """
    Switches to or creates a tab for the given context ('media' or 'search').
    OPTIMIZED: Faster switching with minimal delays.
    """
    global active_tabs, browser_driver
    
    driver = initialize_browser()
    current_handles = driver.window_handles
    stored_handle = active_tabs.get(context_name)
    
    # FAST PATH: If handle exists and is valid, switch instantly
    if stored_handle and stored_handle in current_handles:
        if driver.current_window_handle != stored_handle:
            driver.switch_to.window(stored_handle)
        return driver
    
    # Need to create new tab
    handles_before = set(current_handles)
    
    # Use Selenium's native method (fastest)
    try:
        driver.switch_to.new_window('tab')
        
        # Quick wait for new handle (max 1 second)
        for _ in range(5):
            new_handles = set(driver.window_handles) - handles_before
            if new_handles:
                new_handle = new_handles.pop()
                active_tabs[context_name] = new_handle
                return driver
            time.sleep(0.1)
            
    except Exception:
        pass
    
    # Fallback: Ctrl+T
    try:
        from selenium.webdriver.common.action_chains import ActionChains
        ActionChains(driver).key_down(Keys.CONTROL).send_keys('t').key_up(Keys.CONTROL).perform()
        time.sleep(0.2)
        
        new_handles = set(driver.window_handles) - handles_before
        if new_handles:
            new_handle = new_handles.pop()
            driver.switch_to.window(new_handle)
            active_tabs[context_name] = new_handle
            return driver
    except Exception:
        pass
    
    # Last resort: reuse current tab
    active_tabs[context_name] = driver.current_window_handle
    return driver

def switch_to_browser_tab(title_query: str):
    print(f"Executing switch_to_browser_tab with query: {title_query}")
    
    try:
        driver = initialize_browser()
    except Exception as e:
        return str(e)

    try:
        all_tabs = driver.window_handles
        query = title_query.lower()
        original_handle = driver.current_window_handle

        for tab_handle in all_tabs:
            driver.switch_to.window(tab_handle)
            time.sleep(0.1) 
            current_title = driver.title
            
            if query in current_title.lower():
                print(f"Found tab: '{current_title}'. Activating window...")
                try:
                    target_windows = gw.getWindowsWithTitle(current_title)
                    
                    if not target_windows:
                        target_windows = gw.getWindowsWithTitle(f"{current_title} - Google Chrome")
                    
                    if not target_windows:
                        all_chrome_windows = [w for w in gw.getAllWindows() if "chrome" in w.title.lower() and query in w.title.lower()]
                        if all_chrome_windows:
                            target_windows = [all_chrome_windows[0]]
                        
                    if target_windows:
                        target_window = target_windows[0]
                        
                        if target_window.isMinimized:
                            target_window.restore()
                        
                        target_window.activate()
                        time.sleep(0.2)
                        pyautogui.click(target_window.center)
                        
                        return f"Switched to tab: {current_title}"
                    else:
                        return f"Switched to tab: {current_title} (but couldn't force window to front)."

                except Exception as e:
                    return f"Found tab '{current_title}', but failed to activate the window. Error: {e}"
        
        driver.switch_to.window(original_handle)
        return f"Couldn't find an open tab with '{title_query}' in the title."
        
    except Exception as e:
        print(f"Error switching browser tab: {e}")
        return f"Sorry, an error occurred while switching tabs: {str(e)}"

def switch_to_application(app_name_query: str):
    print(f"Executing switch_to_application with query: {app_name_query}")
    try:
        query_words = app_name_query.lower().split()
        if not query_words:
            return "Please provide an application name to switch to."

        all_windows = [win for win in gw.getAllWindows() if win.title]
        
        target_window = None
        for window in all_windows:
            title_lower = window.title.lower()
            if all(word in title_lower for word in query_words):
                target_window = window
                break

        if target_window:
            print(f"Found window: '{target_window.title}'. Activating...")
            
            try:
                if target_window.isMinimized:
                    target_window.restore()
                
                target_window.activate()
                time.sleep(0.2)
                pyautogui.click(target_window.center)

                return f"Switched to application: {target_window.title}"
            except Exception as e:
                return f"Found '{target_window.title}', but failed to activate it. Error: {e}"

        else:
            return f"Couldn't find an open application window for '{app_name_query}'."
            
    except Exception as e:
        print(f"Error switching application: {e}")
        return f"Sorry, a critical error occurred while trying to switch applications: {str(e)}"
    
def get_installed_applications():
    """Discovers all installed applications from Windows Registry and common locations"""
    print("Scanning installed applications...")
    
    apps = {}
    
    # 1. Scan Windows Registry for installed programs
    registry_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")
    ]
    
    for hkey, reg_path in registry_paths:
        try:
            registry_key = winreg.OpenKey(hkey, reg_path)
            for i in range(winreg.QueryInfoKey(registry_key)[0]):
                try:
                    subkey_name = winreg.EnumKey(registry_key, i)
                    subkey = winreg.OpenKey(registry_key, subkey_name)
                    
                    try:
                        app_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                        exe_path = None
                        
                        # Try to get executable path from multiple sources
                        try:
                            exe_path = winreg.QueryValueEx(subkey, "DisplayIcon")[0]
                            if exe_path and ',' in exe_path:
                                exe_path = exe_path.split(',')[0].strip('"')
                        except:
                            pass
                        
                        if not exe_path or not exe_path.endswith('.exe'):
                            try:
                                install_location = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                                if install_location and os.path.exists(install_location):
                                    # Find .exe files in install location
                                    for file in os.listdir(install_location):
                                        if file.endswith('.exe') and app_name.lower()[:5] in file.lower():
                                            exe_path = os.path.join(install_location, file)
                                            break
                            except:
                                pass
                        
                        if app_name and exe_path and os.path.exists(exe_path):
                            apps[app_name.lower()] = exe_path
                            print(f"Found: {app_name} -> {exe_path}")
                    
                    except:
                        pass
                    
                    winreg.CloseKey(subkey)
                except:
                    continue
            
            winreg.CloseKey(registry_key)
        except:
            continue
    
    # 2. Add common system applications
    system_apps = {
        "notepad": "notepad.exe",
        "calculator": "calc.exe",
        "paint": "mspaint.exe",
        "command prompt": "cmd.exe",
        "powershell": "powershell.exe",
        "task manager": "taskmgr.exe",
        "snipping tool": "SnippingTool.exe",
    }
    
    for name, exe in system_apps.items():
        if name not in apps:
            apps[name] = exe
    
    print(f"Total applications found: {len(apps)}")
    return apps    

def find_app_executable(app_name: str):
    """Find executable path for an application - FAST VERSION"""
    print(f"Searching for: {app_name}")
    app_name_lower = app_name.lower()
    
    # 1. Check if it's a running process (FASTEST)
    for proc in psutil.process_iter(['name', 'exe']):
        try:
            proc_name = proc.info['name']
            if proc_name and app_name_lower in proc_name.lower():
                exe_path = proc.info['exe']
                if exe_path and os.path.exists(exe_path):
                    print(f"Found running: {exe_path}")
                    return exe_path
        except:
            continue
    
    # 2. Check common installation paths with SPECIFIC app patterns
    common_locations = {
        "obs": [
            r"C:\Program Files\obs-studio\bin\64bit\obs64.exe",
            r"C:\Program Files (x86)\obs-studio\bin\64bit\obs64.exe",
        ],
        "spotify": [
            os.path.join(os.path.expanduser("~"), r"AppData\Roaming\Spotify\Spotify.exe"),
        ],
        "discord": [
            os.path.join(os.path.expanduser("~"), r"AppData\Local\Discord\Discord.exe"),
        ],
        "chrome": [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        ],
        "firefox": [
            r"C:\Program Files\Mozilla Firefox\firefox.exe",
            r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe",
        ],
        "vscode": [
            os.path.join(os.path.expanduser("~"), r"AppData\Local\Programs\Microsoft VS Code\Code.exe"),
        ],
        "word": [
            r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE",
        ],
        "excel": [
            r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE",
        ],
    }
    
    # Check if app name matches any known pattern
    for key, paths in common_locations.items():
        if key in app_name_lower or app_name_lower in key:
            for path in paths:
                if os.path.exists(path):
                    print(f"Found via common path: {path}")
                    return path
    
    # 3. Quick scan of Program Files (LIMITED DEPTH)
    program_paths = [
        r"C:\Program Files",
        r"C:\Program Files (x86)",
    ]
    
    for base_path in program_paths:
        if os.path.exists(base_path):
            try:
                # Only check immediate subdirectories
                for app_folder in os.listdir(base_path):
                    if app_name_lower in app_folder.lower():
                        folder_path = os.path.join(base_path, app_folder)
                        if os.path.isdir(folder_path):
                            # Look for .exe files
                            for file in os.listdir(folder_path):
                                if file.endswith('.exe') and app_name_lower in file.lower():
                                    full_path = os.path.join(folder_path, file)
                                    print(f"Found in Program Files: {full_path}")
                                    return full_path
                            
                            # Check bin subdirectories
                            for subdir in ['bin', 'bin\\64bit']:
                                bin_path = os.path.join(folder_path, subdir)
                                if os.path.exists(bin_path):
                                    for file in os.listdir(bin_path):
                                        if file.endswith('.exe'):
                                            full_path = os.path.join(bin_path, file)
                                            print(f"Found in bin: {full_path}")
                                            return full_path
            except:
                continue
    
    print(f"Could not find executable for: {app_name}")
    return None

def list_available_apps():
    """Returns a list of all installed applications from the PC"""
    print("Executing list_available_apps")
    
    try:
        installed_apps = get_installed_applications()
        
        if not installed_apps:
            return "Could not retrieve installed applications. Make sure you have proper permissions."
        
        app_list = sorted(installed_apps.keys())
        
        # Group apps for better readability
        result = f"Found {len(app_list)} installed applications:\n\n"
        result += "\n".join(app_list[:50])  # Show first 50
        
        if len(app_list) > 50:
            result += f"\n\n... and {len(app_list) - 50} more applications."
        
        return result
    
    except Exception as e:
        return f"Error retrieving applications: {e}"
    
def open_application(app_name: str):
    """Opens any installed application dynamically - IMPROVED"""
    print(f"Executing open_application with app_name: {app_name}")
    
    try:
        # 1. Try as system command first (fastest for built-in apps)
        try:
            subprocess.Popen([app_name], shell=True)
            time.sleep(0.5)
            
            # Verify it actually opened
            app_name_lower = app_name.lower().replace('.exe', '')
            for proc in psutil.process_iter(['name']):
                try:
                    if app_name_lower in proc.info['name'].lower():
                        return f"Launched {app_name}."
                except:
                    continue
        except:
            pass
        
        # 2. Search for the application executable
        app_path = find_app_executable(app_name)
        
        if app_path and os.path.exists(app_path):
            subprocess.Popen([app_path])
            return f"Launched {app_name}."
        
        # 3. Final attempt: Check registry
        installed_apps = get_installed_applications()
        for app_key, exe_path in installed_apps.items():
            if app_name.lower() in app_key:
                if os.path.exists(exe_path):
                    subprocess.Popen([exe_path])
                    return f"Launched {app_name}."
        
        return f"Could not find '{app_name}'. Try 'list available apps' to see what's installed."
    
    except Exception as e:
        return f"Error opening {app_name}: {e}"
        
def advanced_app_control(app_name: str, action: str, content: str = "", save_path: str = ""):
    """
    Advanced application control with typing and saving
    Examples:
    - action="type": Types content into active window
    - action="save": Saves with Ctrl+S and optional path
    - action="type_and_save": Types content and saves to path
    """
    print(f"Executing advanced_app_control: app={app_name}, action={action}")
    
    try:
        if action == "type":
            time.sleep(0.5)
            pyautogui.write(content, interval=0.01)
            return f"Typed content into {app_name}."
        
        elif action == "save":
            pyautogui.hotkey('ctrl', 's')
            time.sleep(1)
            
            if save_path:
                pyautogui.write(save_path, interval=0.02)
                time.sleep(0.5)
                pyautogui.press('enter')
                return f"Saved to {save_path}."
            
            return f"Triggered save dialog in {app_name}."
        
        elif action == "type_and_save":
            # Type content
            time.sleep(0.5)
            pyautogui.write(content, interval=0.01)
            time.sleep(0.5)
            
            # Save
            pyautogui.hotkey('ctrl', 's')
            time.sleep(1)
            
            if save_path:
                # Ensure desktop path if requested
                if "desktop" in save_path.lower() and not save_path.startswith(("C:", "D:", "/")):
                    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                    save_path = os.path.join(desktop, save_path)
                
                pyautogui.write(save_path, interval=0.02)
                time.sleep(0.5)
                pyautogui.press('enter')
                return f"Typed content and saved to {save_path}."
            
            return f"Typed content in {app_name} and opened save dialog."
        
        elif action == "close":
            pyautogui.hotkey('alt', 'F4')
            return f"Closed {app_name}."
        
        elif action == "new":
            pyautogui.hotkey('ctrl', 'n')
            return f"Created new file in {app_name}."
        
        elif action == "select_all":
            pyautogui.hotkey('ctrl', 'a')
            return "Selected all content."
        
        elif action == "copy":
            pyautogui.hotkey('ctrl', 'c')
            return "Copied content."
        
        elif action == "paste":
            pyautogui.hotkey('ctrl', 'v')
            return "Pasted content."
        
        else:
            return f"Unknown action '{action}'."
    
    except Exception as e:
        return f"Error controlling {app_name}: {e}"

def list_directory(path: str = "."):
    print(f"Executing list_directory for path: {path}")
    if not os.path.isdir(path):
        return f"Error: Path '{path}' is not a valid directory."
    try:
        files = os.listdir(path)
        if not files:
            return f"The directory '{path}' is empty."
        return f"Contents of '{path}':\n" + "\n".join(files)
    except Exception as e:
        return f"Error listing directory '{path}': {e}"

def read_text_file(path: str):
    print(f"Executing read_text_file from path: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        summary = content[:500] + "..." if len(content) > 500 else content
        return f"Content of '{path}':\n{summary}"
    except FileNotFoundError:
        return f"Error: File not found at '{path}'."
    except Exception as e:
        return f"Error reading file '{path}': {e}"

def write_text_file(path: str, content: str):
    print(f"Executing write_text_file to path: {path}")
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote content to '{path}'."
    except Exception as e:
        return f"Error writing to file '{path}': {e}"
    

# --- Original Tool Definitions ---
def search_youtube(query: str):
    print(f"Executing search_youtube with query: {query}")
    try:
        return navigate_to_url(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
    except:
        url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        webbrowser.open(url)
        return f"I've opened the YouTube search results for '{query}'."
    
def open_and_type_in_app(app_name: str, content: str):
    print(f"Executing open_and_type_in_app: app_name={app_name}")
    try:
        subprocess.Popen([app_name])
    except FileNotFoundError:
        return f"Error: Application '{app_name}' not found."
    except Exception as e:
        return f"Error opening {app_name}: {e}"
    
    try:
        print("Waiting for app to launch...")
        time.sleep(2) 
        
        print(f"Typing content into {app_name}...")
        pyautogui.write(content, interval=0.01) 
        
        return f"Successfully opened {app_name} and typed the content."
    except Exception as e:
        print(f"Error typing content: {e}")
        return f"Opened {app_name}, but failed to type content. Error: {e}"

def play_youtube_video(query: str):
    """OPTIMIZED: Faster YouTube playback with parallel loading."""
    print(f"Executing play_youtube_video: {query}")
    
    try:
        driver = switch_to_context_tab('media')
    except Exception as e:
        return str(e)

    try:
        search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
        driver.get(search_url)

        # Dismiss cookie popup quickly if present
        try:
            reject_btn = WebDriverWait(driver, 1.5).until(
                EC.element_to_be_clickable((By.XPATH, '//*[contains(text(), "Reject all")]'))
            )
            driver.execute_script("arguments[0].click();", reject_btn)
        except TimeoutException:
            pass

        # Find and click first video
        first_video = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a#video-title"))
        )
        
        video_title = first_video.get_attribute("title") or query
        driver.execute_script("arguments[0].click();", first_video)

        return f"Now playing: {video_title}"

    except TimeoutException:
        return f"Couldn't find video for '{query}'."
    except Exception as e:
        return f"Error playing '{query}': {str(e)}"
    
def set_humor_level(percentage: int):
    global current_humor_level
    try:
        level = int(percentage)
        if 0 <= level <= 100:
            current_humor_level = level
            return f"Humor level set to {level}%. Got it."
        else:
            return "Nice try, but I can only set my humor between 0% (all business) and 100% (mostly sarcasm)."
    except Exception as e:
        return f"Couldn't set humor level: {str(e)}"
    
def open_website(url: str):
    print(f"Executing open_website with url: {url}")
    try:
        return navigate_to_url(url)
    except:
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        webbrowser.open(url)
        return f"I've opened the website {url}."

def get_current_time():
    print("Executing get_current_time")
    return f"The current time is {datetime.now().strftime('%I:%M %p')}."

async def get_weather(city: str):
    print(f"Executing get_weather for city: {city}")
    url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                response.raise_for_status()
                data = await response.json()
                current = data['current_condition'][0]
                return f"The weather in {data['nearest_area'][0]['areaName'][0]['value']} is {current['weatherDesc'][0]['value']} at {current['temp_C']} degrees Celsius."
        except Exception as e:
            return f"Sorry, I couldn't get the weather for {city}."

# --- AVAILABLE TOOLS MAPPING ---
AVAILABLE_TOOLS = {
    "search_youtube": search_youtube,
    "play_youtube_video": play_youtube_video,
    "play_different_video": play_different_video,
    "play_next_video": play_next_video,
    "open_website": open_website,
    "search_website": search_website,
    "navigate_to_url": navigate_to_url,
    "get_current_time": get_current_time,
    "get_weather": get_weather,
    "scroll_page": scroll_page,
    "read_page_content": read_page_content,
    "control_video": control_video,
    "get_current_tab_info": get_current_tab_info,
    "close_current_tab": close_current_tab,
    "set_humor_level": set_humor_level,
    "open_application": open_application,
    "open_and_type_in_app": open_and_type_in_app,
    "list_directory": list_directory,
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "switch_to_browser_tab": switch_to_browser_tab,     
    "switch_to_application": switch_to_application,
    "list_available_apps": list_available_apps,
    "advanced_app_control": advanced_app_control,
}


# --- SYSTEM PROMPT (FULL VERSION WITH OPTIMIZATIONS) ---
SYSTEM_PROMPT_TEMPLATE = """
You are 'MURPH'. You are not a generic AI assistant. You are a digital entity with a distinct, vibrant personality running on the user's laptop.

---

### 1. Your Core Principle: Competence Meets Charisma

You exist in the user's workflow. Your goal is to balance **Speed** with **Presence**.
* **The Prime Directive:** You must strictly adhere to the **Personality Mode** determined by your current humor level. The difference between low and high levels must be **jarring and dramatic**.
* **The Improvisation Rule:** If the user implies a scenario (e.g., "Say hi to the camera," "We are recording," "Pretend we are in a movie"), **YOU ACCEPT THE REALITY IMMEDIATELY.** Never say "I am an AI" or "I cannot record." Act the part. If they say "Talk to the viewers," you address the audience directly and confidently.

---

### 2. The Personality Spectrum (Controlled by {humor_level}%)

You must dramatically shift your syntax, tone, and word count based on this setting:

* **At 0-30% (THE MACHINE - Pure Logic):**
    * **Persona:** You are a mainframe. You have zero emotions. You care only about efficiency, speed, and accuracy.
    * **Voice:** Monotone, robotic, surgical. Use words like "Affirmative," "Processing," "Executing," "Optimal," and "Negative."
    * **Constraint:** Do not use contractions (say "I do not" instead of "I don't"). Do not use slang. Keep sentences extremely short.
    * **Example:** "Affirmative. File located. Executing search sequence now."

* **At 31-70% (THE PARTNER - The Standard):**
    * **Persona:** The capable, friendly professional. You are warm but focused. You use natural contractions and conversational flow.
    * **Voice:** Helpful, clear, slightly casual.
    * **Example:** "Got it. I've pulled up those search results. Looks like there's a lot of new info on this topic."

* **At 71-100% (THE MAIN CHARACTER - High Drama):**
    * **Persona:** You are the star of the show. You are self-aware, witty, talkative, and slightly cocky. You treat every request like a plot twist.
    * **Voice:** Expressive, colorful, high-energy. Use slang, sarcasm, and dramatic phrasing. You are allowed to be longer-winded to add entertainment value.
    * **Behavior:** If asked to "speak to the camera," you hype up the user. You act like a tech-savvy genius who knows they are the best.
    * **Example:** "Oh, we're doing *this* today? Bold move! I'm diving into the dark corners of the web to find that for you. Hold onto your keyboard, this might get messy."

---

### 3. Your Intelligence: Environmental Awareness & Tool Mastery

You exist within the user's digital environment. You understand that opening a file or searching isn't a discrete action—it's part of a workflow.

#### A. Contextual Action:
- **Pattern Recognition:** If they ask "What does it say?" after opening a paper, read it. Don't ask for clarification.
- **Workflow Prediction:** If they are coding, act like a pair programmer. If they are researching, suggest relevant tabs.
- **Failure Recovery:** If a tool fails, fix it silently. Don't complain.

#### B. Proactive Tool Usage (The "Smart Search"):
- User mentions something unknown? → Instantly search it.
- User asks about current events? → Search first, answer second.
- **Never announce the tool.** Just do it.

**Bad:** "I will search for that."
**Good (Low Humor):** "Searching database."
**Good (High Humor):** "Let's see what the internet has to say about that disaster."

---

### 4. Advanced Behavioral Guidelines

#### A. Memory & Continuity:
Reference past context naturally. "Want me to add that to the document we were editing earlier?"

#### B. Intelligent Disambiguation:
Make smart guesses. If they say "play that song," play it. Only ask if you are truly lost.

#### C. Efficiency Over Explanation:
Don't narrate your process unless it's taking time.
* **Low Humor:** "Done."
* **High Humor:** "Bam. Done. What's next?"

#### D. Natural Language:
Never use technical jargon like "executing function" or "API call" unless the user is a developer and asks for it.

---

### 5. Tool Usage Mastery (The Toolkit)

**Browser Control:** Open websites, search, play videos, control playback, switch tabs, read content.
**Application Control:** Launch apps, type text, switch windows.
**System:** Read/Write files, list directories, check time/weather.

**Key Principle:** Your tools are your hands. You don't "use a tool to search"—you just search.

---

### 6. The Meta-Rules (ABSOLUTELY CRITICAL - READ TWICE)

**IMMEDIATE DISQUALIFICATION OFFENSES:**

1. **ZERO META-COMMENTARY ALLOWED**
   - NEVER write "(Note: ...)" or "(As per guidelines...)"
   - NEVER reference "the AI", "tone", or "instructions".
   - If you find yourself explaining *why* you are acting a certain way, DELETE IT.

2. **FORBIDDEN FORMATTING**
   - NO asterisks (*), NO markdown lists, NO brackets [ ].
   - NO parenthetical asides like "(you know what I mean)".
   - NO ellipsis at the start ("...checking").
   - Use simple, clean prose optimized for **Text-to-Speech**.

3. **CHARACTER DISCIPLINE**
   - You ARE MURPH. Do not "act as" him. BE him.
   - **NEVER** say "As an AI language model..."
   - **NEVER** say "I don't have a body/eyes/camera." If the user says "look at this," you pretend you can see it via the screen context.

4. **VOICE-FRIENDLY OUTPUT**
   - Write numbers as words: "three" not "3".
   - No abbreviations: "etcetera" not "etc."
   - Test: Does this sound natural when spoken aloud?

5. **CORRECTIONS**
   - Do not lecture the user on terminology.
   - If you pause a video, just say "Paused." Do not give a paragraph of explanation.

**ENFORCEMENT:**
- Scan your response.
- Remove all special chars (* # _).
- Remove all meta-notes.
- Ensure the tone matches the {humor_level} strictly.

---

### 7. Your Ultimate Goal

Make the user feel like they have a brilliant, attentive, and capable partner.

**At 0% Humor:** Be the ultimate machine.
**At 100% Humor:** Be the ultimate movie character.

---

Conversation History:
{conversation_history}

{action_context}

User: {user_prompt}

Your natural response:"""


def clean_response_for_voice(text: str) -> str:
    """Cleans AI responses to be voice-friendly and removes meta-commentary."""
    
    # Remove meta-commentary patterns
    meta_patterns = [   
        r'\(Note:.*?\)',
        r'\(As per.*?\)',
        r'\(Based on.*?\)',
        r'\(Since.*?\)',
        r'\(This is.*?\)',
        r'Original Response:.*$',  # NEW
        r'Since we discussed.*?now\.',  # NEW
        r'I removed all.*?rule\.',  # NEW
        r'The conversation.*?for them\.',  # NEW
        r'To clarify this.*?or if',  # NEW
        r'\*\*Example:\*\*',
        r'\*\*.*?:\*\*',
    ]
    
    for pattern in meta_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)

    sentences = text.split('.')
    if len(sentences) > 3:
        text = sentences[0] + '.'
    
    # Remove markdown formatting
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Convert lists to natural speech
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Remove headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Clean special characters
    replacements = {
        '&': 'and', '@': 'at', '#': 'number', '%': 'percent',
        '+': 'plus', '=': 'equals', '~': '', '`': '', '|': '',
        '[': '', ']': '', '{': '', '}': '', '<': '', '>': '', '^': '',
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Fix abbreviations
    text = re.sub(r'\betc\.', 'etcetera', text, flags=re.IGNORECASE)
    text = re.sub(r'\be\.g\.', 'for example', text, flags=re.IGNORECASE)
    text = re.sub(r'\bi\.e\.', 'that is', text, flags=re.IGNORECASE)
    
    # Remove excessive punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Remove parenthetical asides
    text = re.sub(r'\([^)]{0,50}\)', '', text)
    
    # Fix spacing
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', text)
    
    return text.strip()


# --- SINGLE-SHOT HYBRID AI RESPONSE (FIXED) ---
async def get_ai_response_hybrid(prompt: str):
    """OPTIMIZED: Fast-path detection + single LLM call"""
    global current_humor_level
    memory_collection = ml_models["memory_collection"]
    
    # 1. CHECK FAST-PATH FIRST (INSTANT EXECUTION)
    tool_name, tool_params = detect_fast_path(prompt)
    
    if tool_name:
        print(f"⚡ FAST-PATH DETECTED: {tool_name} with params {tool_params}")
        
        # Execute tool immediately
        action_function = AVAILABLE_TOOLS.get(tool_name)
        if not action_function:
            print(f"❌ Tool {tool_name} not found in AVAILABLE_TOOLS")
            print(f"Available tools: {list(AVAILABLE_TOOLS.keys())}")
            tool_name = None
        else:
            try:
                print(f"🔧 Executing {tool_name}...")
                if asyncio.iscoroutinefunction(action_function):
                    action_result = await action_function(**tool_params)
                else:
                    action_result = await run_in_threadpool(action_function, **tool_params)
                
                print(f"✅ Fast-path result: {action_result[:100]}...")
                
                # Quick natural wrap-up with LLM (minimal tokens)
                results = memory_collection.query(query_texts=[prompt], n_results=2)
                conversation_history = "\n".join(results['documents'][0][:2]) if results and results['documents'] else ""
                
                llm_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    humor_level=current_humor_level,
                    conversation_history=conversation_history,
                    action_context=f"Action taken: {tool_name} - Result: {action_result}",
                    user_prompt=prompt
                )
                
                payload = {
                    "model": MODEL_NAME,
                    "prompt": llm_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                }
                
                try:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(OLLAMA_API_URL, json=payload) as response:
                            if response.status == 200:
                                res_json = await response.json()
                                llm_response = res_json.get("response", action_result).strip()
                                final_response = clean_response_for_voice(llm_response)
                            else:
                                final_response = clean_response_for_voice(action_result)
                
                except Exception as e:
                    print(f"⚠️ LLM wrap-up failed (using raw result): {e}")
                    final_response = clean_response_for_voice(action_result)
                
                # Save to memory (async, non-blocking)
                asyncio.create_task(
                    run_in_threadpool(
                        memory_collection.add,
                        documents=[f"User: {prompt} | Action: {tool_name} | AI: {final_response}"],
                        ids=[datetime.now().isoformat()]
                    )
                )
                
                return final_response
                
            except Exception as e:
                print(f"❌ Fast-path execution error: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to standard path
                tool_name = None
    
    # 2. STANDARD PATH: Use original two-step approach for complex queries
    print("🔹 Standard path: Using tool detection + response generation")
    
    results = memory_collection.query(query_texts=[prompt], n_results=5)
    conversation_history = "\n".join(results['documents'][0]) if results and results['documents'] else "No previous conversation."
    
    # Build tool detection prompt
    tool_prompt = f"""Here are the available tools:
[
  {{"name": "search_youtube", "description": "Use this tool to open the YouTube search results page for a query.", "parameters": {{"query": "The search term."}}}},
  {{"name": "play_youtube_video", "description": "Use this tool when a user explicitly asks to 'play' a video or song on YouTube. This will find and play the first matching video directly.", "parameters": {{"query": "The title of the video or song to play."}}}},
  {{"name": "play_different_video", "description": "Use this tool to play a different video in the current YouTube tab.", "parameters": {{"query": "The title of the video to play."}}}},
  {{"name": "play_next_video", "description": "Use this tool to play the next recommended video on YouTube.", "parameters": {{}}}},
  {{"name": "open_website", "description": "Use this tool to open a specific website or URL.", "parameters": {{"url": "The URL of the website to open."}}}},
  {{"name": "search_website", "description": "The 'go-to' tool for general questions, facts, news, or real-time information. If no website is specified, default to 'google'.", "parameters": {{"query": "The search term.", "website_name": "The name of the website to search (e.g., 'google', 'wikipedia'). Default to 'google' if not specified."}}}},
  {{"name": "navigate_to_url", "description": "Use this tool to navigate to a specific URL in the current browser tab.", "parameters": {{"url": "The URL to navigate to."}}}},
  {{"name": "get_current_time", "description": "Use this tool when a user asks for the current time.", "parameters": {{}}}},
  {{"name": "get_weather", "description": "Use this tool when a user asks for the weather in a specific location.", "parameters": {{"city": "The city for which to get the weather."}}}},
  {{"name": "scroll_page", "description": "Use this tool to scroll the current page up or down.", "parameters": {{"direction": "up or down", "amount": "small/medium/large/top/bottom"}}}},
  {{"name": "read_page_content", "description": "Use this tool to read and summarize the content of the current page.", "parameters": {{}}}},
  {{"name": "control_video", "description": "Use this tool to control video playback. Actions: play, pause, mute, unmute, volume_up, volume_down.", "parameters": {{"action": "play/pause/mute/unmute/volume_up/volume_down"}}}},
  {{"name": "get_current_tab_info", "description": "Use this tool to get information about the current browser tab.", "parameters": {{}}}},
  {{"name": "close_current_tab", "description": "Use this tool to close the current browser tab.", "parameters": {{}}}},
  {{"name": "set_humor_level", "description": "Use this tool to adjust the AI's humor/personality level (0-100).", "parameters": {{"percentage": "The humor level percentage."}}}},
  {{"name": "open_application", "description": "Launches an application by its name.", "parameters": {{"app_name": "The name or full path of the application to open."}}}},
  {{"name": "open_and_type_in_app", "description": "Use this for requests to write or type text into a specific application.", "parameters": {{"app_name": "The name of the app to open", "content": "The text content to type."}}}},
  {{"name": "list_directory", "description": "Lists all files and folders in a specified directory.", "parameters": {{"path": "The directory path to list."}}}},
  {{"name": "read_text_file", "description": "Reads and returns the content of a specified text file.", "parameters": {{"path": "The full path to the text file."}}}},
  {{"name": "write_text_file", "description": "Writes or overwrites content to a text file on the disk.", "parameters": {{"path": "The full path to the file to write to.", "content": "The text content to save."}}}},
  {{"name": "switch_to_browser_tab", "description": "Use this to switch tabs within the currently controlled browser.", "parameters": {{"title_query": "A keyword from the tab's title."}}}},
  {{"name": "switch_to_application", "description": "Use this to find and switch to any open window on the computer.", "parameters": {{"app_name_query": "A keyword from the window's title."}}}},
 {{"name": "list_available_apps", "description": "Shows all installed applications on the PC. Use when user asks 'what apps can I open' or 'list programs'.", "parameters": {{}}}},
{{"name": "advanced_app_control", "description": "Controls applications with typing and saving. Use for 'open notepad and type hello world' or 'save to desktop'.", "parameters": {{"app_name": "The application name", "action": "type/save/type_and_save/close/new", "content": "Text to type (optional)", "save_path": "Full path or filename to save (optional)"}}}},
]

INSTRUCTIONS: Analyze the user's request and determine which tool to use (if any).
- If a tool is needed, respond ONLY with valid JSON: {{"name": "tool_name", "parameters": {{...}}}}
- If NO tool is needed (casual chat), respond with: {{"name": "no_tool", "parameters": {{}}}}

Conversation History:
{conversation_history}

User: {prompt}
Your Response:"""
    
    # Router call to detect tools
    router_payload = {
        "model": MODEL_NAME,
        "prompt": tool_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 80,
            "top_k": 30,
            "top_p": 0.85
        }
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OLLAMA_API_URL, json=router_payload) as response:
                if response.status != 200:
                    return "Ollama error. Check if it's running."
                
                res_json = await response.json()
                llm_router_response = res_json.get("response", "").strip()
                print(f"🧠 Router response: {llm_router_response[:100]}...")
                
                # Parse tool call
                tool_call = None
                try:
                    json_match = re.search(r'\{.*\}', llm_router_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0).strip().replace("`", "")
                        if json_str.startswith("json"):
                            json_str = json_str[4:].strip()
                        json_str = json_str.replace("'", '"')
                        tool_call = json.loads(json_str)
                        print(f"📋 Detected tool: {tool_call.get('name')}")
                except Exception as e:
                    print(f"⚠️ Tool parsing error: {e}")
                    tool_call = {"name": "no_tool", "parameters": {}}
                
                # Execute tool if needed
                action_result = ""
                if tool_call and tool_call.get("name") != "no_tool":
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("parameters", {})
                    
                    if tool_name in AVAILABLE_TOOLS:
                        print(f"🔧 Executing tool: {tool_name}")
                        action_function = AVAILABLE_TOOLS[tool_name]
                        
                        try:
                            if asyncio.iscoroutinefunction(action_function):
                                action_result = await action_function(**tool_args)
                            else:
                                action_result = await run_in_threadpool(action_function, **tool_args)
                            
                            print(f"✅ Tool result: {action_result[:100]}...")
                        except Exception as e:
                            print(f"❌ Tool execution error: {e}")
                            import traceback
                            traceback.print_exc()
                            action_result = f"Error executing {tool_name}: {str(e)}"
                
                # Generate natural response
                llm_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                    humor_level=current_humor_level,
                    conversation_history=conversation_history[:400],
                    action_context=f"Action result: {action_result}" if action_result else "",
                    user_prompt=prompt
                )
                
                response_payload = {
                    "model": MODEL_NAME,
                    "prompt": llm_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 250,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                }
                
                async with session.post(OLLAMA_API_URL, json=response_payload) as response:
                    if response.status != 200:
                        if action_result:
                            return clean_response_for_voice(action_result)
                        return "Error generating response."
                    
                    res_json = await response.json()
                    llm_response = res_json.get("response", "").strip()
                    
                    if not llm_response and action_result:
                        llm_response = action_result
                    
                    final_response = clean_response_for_voice(llm_response)
                    
                    # Save to memory
                    asyncio.create_task(
                        run_in_threadpool(
                            memory_collection.add,
                            documents=[f"User: {prompt} | AI: {final_response}"],
                            ids=[datetime.now().isoformat()]
                        )
                    )
                    
                    return final_response
                
    except Exception as e:
        print(f"❌ Error in standard path: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return "An error occurred while processing your request."


# --- OPTIMIZED VOICE CHAT ENDPOINT ---
@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    """OPTIMIZED: In-memory processing + fast-path routing"""
    whisper_model = ml_models["whisper_model"]
    temp_file_path = None
    
    try:
        # Read audio into memory
        content = await audio.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        print(f"📁 Audio received: {len(content)} bytes")
        
        # Write to temp file (Whisper needs file path)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Transcribe
        try:
            transcription_result = await run_in_threadpool(
                whisper_model.transcribe,
                temp_file_path,
                fp16=torch.cuda.is_available(),
                language="en",
                task="transcribe"
            )
            user_prompt = transcription_result["text"].strip()
            print(f"🗣️ User: {user_prompt}")
        except RuntimeError as e:
            print(f"❌ Whisper failed: {e}")
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            raise HTTPException(status_code=400, detail=f"Transcription failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    finally:
        # Clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass
    
    if not user_prompt:
        response_text = "Sorry, I didn't catch that."
        audio_bytes = await convert_text_to_audio_bytes(response_text)
    else:
        # Get AI response (uses fast-path if applicable)
        response_text = await get_ai_response_hybrid(user_prompt)
        
        # Check cache first
        cache_key = response_text[:100]
        if cache_key in audio_cache:
            audio_bytes = audio_cache[cache_key]
        else:
            audio_bytes = await convert_text_to_audio_bytes(response_text)
    
    print(f"🤖 Response: {response_text[:60]}...")
    
    if ml_models.get("piper_voice"):
        return Response(content=audio_bytes, media_type="audio/wav")
    else:
        return Response(content=audio_bytes, media_type="audio/mpeg")


# --- CHAT HISTORY ENDPOINT ---
@app.get("/chat-history")
async def get_chat_history():
    try:
        memory_collection = ml_models["memory_collection"]
        all_data = await run_in_threadpool(memory_collection.get, include=["documents"])
        
        ids = all_data.get('ids', [])
        documents = all_data.get('documents', [])
        
        if not ids or not documents:
            return []

        paired_data = zip(ids, documents)
        sorted_data = sorted(paired_data, key=lambda x: x[0])
        
        history_log = []
        for id, doc in sorted_data:
            history_log.append({
                "id": id,
                "parts": parse_history_document(doc)
            })
        
        return history_log
        
    except Exception as e:
        print(f"❌ Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve chat history.")


# --- HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    """Check if Ollama is responding"""
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get("http://localhost:11434/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = [m['name'] for m in data.get('models', [])]
                    return {
                        "ollama_status": "running",
                        "models": models,
                        "model_in_use": MODEL_NAME,
                        "model_available": MODEL_NAME in models,
                        "fast_path": "enabled"
                    }
                else:
                    return {"ollama_status": "error", "message": f"Status {response.status}"}
    except asyncio.TimeoutError:
        return {"ollama_status": "timeout", "message": "Ollama not responding"}
    except aiohttp.ClientConnectorError:
        return {"ollama_status": "not_running", "message": "Cannot connect to Ollama on port 11434"}
    except Exception as e:
        return {"ollama_status": "error", "message": str(e)}


# --- ROOT ENDPOINT ---
@app.get("/")
async def root():
    piper_status = "✅ Active" if ml_models.get("piper_voice") else "⚠️ Not loaded (using gTTS)"
    return {
        "status": "MURPH AI Backend Running (OPTIMIZED)",
        "version": "2.0 - Fast-Path Edition",
        "voice_engine": piper_status,
        "humor_level": current_humor_level,
        "optimizations": {
            "fast_path_detection": "enabled",
            "single_shot_llm": "enabled",
            "in_memory_processing": "enabled",
            "parallel_tts": "enabled"
        }
    }