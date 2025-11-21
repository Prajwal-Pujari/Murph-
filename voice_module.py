# import asyncio
# import json
# import aiohttp
# import torch
# import whisper
# import pyttsx3
# import chromadb
# import tempfile
# import io
# import os
# import uuid
# import socket
# import webbrowser
# import inspect
# import re
# from datetime import datetime
# from contextlib import asynccontextmanager
# import pyautogui
# import pygetwindow as gw
# import time
# import subprocess
# from concurrent.futures import ThreadPoolExecutor
# import queue

# from piper.voice import PiperVoice
# from gtts import gTTS
# from pytube import Search
# from fastapi import FastAPI, HTTPException, File, UploadFile
# from fastapi.concurrency import run_in_threadpool
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import Response
# from pydantic import BaseModel

# # Browser Control Imports
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.common.keys import Keys
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.chrome.options import Options
# from selenium.common.exceptions import TimeoutException, NoSuchElementException, WebDriverException
# from selenium.webdriver.chrome.service import Service
# from webdriver_manager.chrome import ChromeDriverManager
# import time

# # --- Global State & Lifespan ---
# ml_models = {}
# browser_driver = None
# current_humor_level = 85
# tts_executor = ThreadPoolExecutor(max_workers=2)
# audio_cache = {}

# last_mentioned_items = {
#     "song": None,
#     "video": None,
#     "website": None,
#     "file": None,
#     "application": None
# }

# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     global browser_driver
    
#     print("🔹 Loading Whisper model...")
#     ml_models["whisper_model"] = whisper.load_model("small.en")
#     print("✅ Whisper model loaded.")

#     print("🔹 Loading Offline TTS Voice (Piper)...")
#     try:
#         possible_paths = [
#             "./models/en_US-hfc_male-medium.onnx",
#             "./en_US-hfc_male-medium.onnx",
#             "models/en_US-hfc_male-medium.onnx",
#             "./models/en_US-lessac-medium.onnx",
#             "./en_US-lessac-medium.onnx",
#             "./models/en_US-ryan-high.onnx",
#         ]
        
#         onnx_file = None
#         for path in possible_paths:
#             json_path = path + ".json"
#             if os.path.exists(path) and os.path.exists(json_path):
#                 onnx_file = path
#                 print(f"✅ Found Piper model at: {path}")
#                 print(f"✅ Found Piper config at: {json_path}")
#                 break
        
#         if onnx_file:
#             ml_models["piper_voice"] = PiperVoice.load(onnx_file)
#             print("✅ Offline TTS Voice (Piper) loaded successfully!")
#             print(f"🎤 Using voice: {onnx_file}")
#         else:
#             print("⚠️ Piper voice model not found in any expected location.")
#             print("📥 Download from: https://github.com/rhasspy/piper/releases/")
#             print("📁 You need BOTH files: .onnx AND .onnx.json in the 'models' folder")
#             ml_models["piper_voice"] = None
            
#     except Exception as e:
#         print(f"⚠️ Could not load Piper TTS model: {e}")
#         print(f"📝 Error details: {type(e).__name__}: {str(e)}")
#         ml_models["piper_voice"] = None

#     print("🔹 Initializing ChromaDB...")
#     client = chromadb.PersistentClient(path="memory_db")
#     ml_models["memory_collection"] = client.get_or_create_collection(name="conversations")
#     print("✅ ChromaDB ready.")
    
#     browser_driver = None
#     print("✅ Browser control ready (will open when needed).")
    
#     yield
    
#     print("🔹 Shutting down...")
#     if browser_driver:
#         browser_driver.quit()
#     tts_executor.shutdown(wait=False)
#     audio_cache.clear()
#     ml_models.clear()


# # --- App Initialization, Middleware, and Settings ---
# class ChatRequest(BaseModel):
#     prompt: str
    
# app = FastAPI(lifespan=lifespan)
# origins = ["http://localhost:5173"]
# app.add_middleware(
#     CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
# )
# OLLAMA_API_URL = "http://localhost:11434/api/generate"
# MODEL_NAME = "llama3.1:8b-instruct-q4_K_M"
# OLLAMA_TIMEOUT = 120


# # --- Hybrid TTS Logic (OPTIMIZED) ---
# def is_online(host="8.8.8.8", port=53, timeout=2):
#     try:
#         socket.setdefaulttimeout(timeout)
#         socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
#         return True
#     except socket.error as ex:
#         return False

# async def convert_text_to_audio_bytes(text: str, force_regenerate: bool = False) -> bytes:
#     """TTS with smart caching - avoid cached responses for personalized content"""
    
#     cache_key = text[:100]
    
#     # Skip cache for personalized content
#     personalized_indicators = ['nice to meet', 'proj', 'speaking with', 'finally', 'great to']
#     should_skip_cache = force_regenerate or any(ind in text.lower() for ind in personalized_indicators)
    
#     if not should_skip_cache and cache_key in audio_cache:
#         print("⚡ Cached audio")
#         return audio_cache[cache_key]
    
#     if ml_models.get("piper_voice"):
#         print("🔌 Piper TTS...")
        
#         def _generate_piper_audio():
#             try:
#                 audio_chunks = []
#                 for chunk in ml_models["piper_voice"].synthesize(text):
#                     audio_chunks.append(chunk.audio_int16_bytes)
                
#                 if not audio_chunks:
#                     return None
                
#                 audio_data = b''.join(audio_chunks)
                
#                 if len(audio_data) < 100:
#                     return None
                
#                 import wave
#                 wav_buffer = io.BytesIO()
                
#                 with wave.open(wav_buffer, 'wb') as wav_file:
#                     wav_file.setnchannels(1)
#                     wav_file.setsampwidth(2)
#                     wav_file.setframerate(ml_models["piper_voice"].config.sample_rate)
#                     wav_file.writeframes(audio_data)
                
#                 wav_buffer.seek(0)
#                 return wav_buffer.read()
                    
#             except Exception as e:
#                 print(f"❌ Piper failed: {e}")
#                 return None
        
#         audio_bytes = await run_in_threadpool(_generate_piper_audio)
        
#         if audio_bytes:
#             if not should_skip_cache:
#                 audio_cache[cache_key] = audio_bytes
#                 if len(audio_cache) > 20:
#                     audio_cache.pop(next(iter(audio_cache)))
#             return audio_bytes
    
#     # gTTS fallback
#     if is_online():
#         print("🌐 gTTS...")
#         def _generate_gtts_audio():
#             try:
#                 tts = gTTS(text=text, lang='en', slow=False)
#                 audio_stream = io.BytesIO()
#                 tts.write_to_fp(audio_stream)
#                 audio_stream.seek(0)
#                 return audio_stream.read()
#             except Exception as e:
#                 print(f"gTTS failed: {e}")
#                 return None
        
#         audio_bytes = await run_in_threadpool(_generate_gtts_audio)
#         if audio_bytes:
#             if not should_skip_cache:
#                 audio_cache[cache_key] = audio_bytes
#             return audio_bytes

#     raise HTTPException(status_code=500, detail="All TTS methods failed")


# # --- Browser Control Functions ---
# def initialize_browser():
#     global browser_driver
#     if browser_driver:
#         try:
#             _ = browser_driver.window_handles
#             return browser_driver
#         except WebDriverException:
#             print("⚠️ Browser was running but is no longer reachable. Re-initializing...")
#             browser_driver = None

#     try:
#         service = Service(ChromeDriverManager().install())
#     except Exception as e:
#         error_message = f"❌ FATAL: Failed to initialize ChromeDriver service via webdriver-manager. Error: {e}"
#         print(error_message)
#         raise Exception(error_message)

#     try:
#         print("🔹 Attempting to connect to existing Chrome browser on port 9222...")
#         chrome_options_remote = Options()
#         chrome_options_remote.add_experimental_option("debuggerAddress", "localhost:9222")
#         browser_driver = webdriver.Chrome(service=service, options=chrome_options_remote)
#         print("✅ Successfully connected to existing Chrome browser.")
#         return browser_driver
#     except WebDriverException:
#         print("⚠️ Could not connect to existing browser. This is normal if one isn't running in debug mode.")
#         print("🔹 Proceeding to launch a new browser instance...")
#         pass

#     try:
#         chrome_options_new = Options()
#         chrome_options_new.add_argument("--disable-blink-features=AutomationControlled")
#         chrome_options_new.add_argument("--disable-extensions")
#         chrome_options_new.add_experimental_option("excludeSwitches", ["enable-logging"])
        
#         browser_driver = webdriver.Chrome(service=service, options=chrome_options_new)
#         print("✅ Successfully launched and connected to a new Chrome browser.")
#         return browser_driver
#     except Exception as e:
#         error_message = f"❌ FATAL: Failed to both connect and launch Chrome. The final launch attempt failed with error: {e}"
#         print(error_message)
#         browser_driver = None
#         raise Exception(error_message)
    
# def parse_history_document(doc_string: str):
#     parts = doc_string.split(" | ")
#     parsed_parts = []
#     for part in parts:
#         if ": " in part:
#             key, value = part.split(": ", 1)
#             parsed_parts.append({"type": key.strip(), "content": value.strip()})
#         else:
#             parsed_parts.append({"type": "Unknown", "content": part})
#     return parsed_parts
    

# def search_website(query: str, website_name: str):
#     print(f"Executing search_website: query={query}, website={website_name}")
    
#     search_url_templates = {
#         "wikipedia": "https://en.wikipedia.org/w/index.php?search=",
#         "google": "https://www.google.com/search?q=",
#         "youtube": "https://www.youtube.com/results?search_query=",
#     }
    
#     url_query = query.replace(' ', '+')
#     site_key = website_name.lower().strip()
    
#     if site_key in search_url_templates:
#         url = search_url_templates[site_key] + url_query
#         result_msg = f"Searching {website_name} for '{query}'."
#     else:
#         url = search_url_templates["google"] + url_query
#         result_msg = f"I don't have a specific search for {website_name}, so I searched Google for '{query}' instead."

#     try:
#         navigate_result = navigate_to_url(url)
#         return f"{result_msg} {navigate_result}"
#     except Exception as e:
#         print(f"Error during search_website navigation: {e}")
#         return f"Sorry, an error occurred while trying to search {website_name}: {str(e)}"
    

# def scroll_page(direction: str = "down", amount: str = "medium"):
#     print(f"Executing scroll_page: direction={direction}, amount={amount}")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)
    
#     try:
#         scroll_amounts = {
#             "small": 300,
#             "medium": 600,
#             "large": 1000,
#             "bottom": 999999,
#             "top": -999999
#         }
        
#         pixels = scroll_amounts.get(amount.lower(), 600)
#         if direction.lower() == "up":
#             pixels = -abs(pixels)
        
#         driver.execute_script(f"window.scrollBy(0, {pixels});")
#         return f"Scrolled {direction} on the page."
#     except Exception as e:
#         print(f"Error scrolling: {e}")
#         return f"Could not scroll the page: {str(e)}"

# def read_page_content():
#     print("Executing read_page_content")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)
    
#     try:
#         title = driver.title
#         body_text = driver.find_element(By.TAG_NAME, "body").text
#         summary = body_text[:500] + "..." if len(body_text) > 500 else body_text
#         return f"Page title: {title}. Content: {summary}"
#     except Exception as e:
#         print(f"Error reading page: {e}")
#         return f"Could not read the page content: {str(e)}"

# def control_video(action: str):
#     print(f"Executing control_video: action={action}")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)
    
#     try:
#         driver.execute_script("document.body.click();") 
        
#         action = action.lower()
#         body = driver.find_element(By.TAG_NAME, "body")
        
#         if action in ["play", "pause"]:
#             body.send_keys("k")
#             return "Toggled play/pause."
            
#         elif action in ["mute", "unmute"]:
#             body.send_keys("m")
#             return "Toggled mute."
        
#         elif action == "volume_up":
#             body.send_keys(Keys.ARROW_UP)
#             return "Increased volume."

#         elif action == "volume_down":
#             body.send_keys(Keys.ARROW_DOWN)
#             return "Decreased volume."
            
#         else:
#             return f"Unknown video control action: {action}"
            
#     except Exception as e:
#         print(f"Error controlling video: {e}")
#         return f"Could not control the video: {str(e)}"

# def navigate_to_url(url: str):
#     print(f"Executing navigate_to_url: url={url}")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)
    
#     try:
#         if not url.startswith(('http://', 'https://')):
#             url = 'https://' + url
        
#         driver.get(url)
#         time.sleep(1)
#         return f"Navigated to {url}."
#     except Exception as e:
#         print(f"Error navigating: {e}")
#         return f"Could not navigate to {url}: {str(e)}"

# def play_different_video(query: str):
#     print(f"Executing play_different_video: query={query}")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)
    
#     try:
#         search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
#         driver.get(search_url)
#         time.sleep(1.5)
        
#         try:
#             first_video = WebDriverWait(driver, 10).until(
#                 EC.element_to_be_clickable((By.CSS_SELECTOR, "a#video-title"))
#             )
#             video_title = first_video.text
#             first_video.click()
#             return f"Now playing: {video_title}"
#         except TimeoutException:
#             return f"Opened YouTube search for {query}, but couldn't auto-play the video."
            
#     except Exception as e:
#         print(f"Error playing different video: {e}")
#         return f"Could not play the video: {str(e)}"

# def play_next_video():
#     print("Executing play_next_video")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)
    
#     try:
#         next_button = driver.find_element(By.CLASS_NAME, "ytp-next-button")
#         next_button.click()
#         time.sleep(2)
        
#         new_video_title = driver.title.replace("- YouTube", "").strip()
#         return f"Playing next video: {new_video_title}"
#     except NoSuchElementException:
#         return "Couldn't find the 'next video' button. Are you on a YouTube video page?"
#     except Exception as e:
#         print(f"Error playing next video: {e}")
#         return f"Could not play the next video: {str(e)}"

# def get_current_tab_info():
#     print("Executing get_current_tab_info")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)
    
#     try:
#         title = driver.title
#         url = driver.current_url
#         return f"Current tab: {title} at {url}"
#     except Exception as e:
#         print(f"Error getting tab info: {e}")
#         return f"Could not get tab information: {str(e)}"

# def close_current_tab():
#     print("Executing close_current_tab")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)
    
#     try:
#         if len(driver.window_handles) == 1:
#             driver.quit()
#             global browser_driver
#             browser_driver = None
#             return "Last tab closed. Browser has been shut down."

#         driver.close()
#         driver.switch_to.window(driver.window_handles[-1])
#         return "Tab closed. Switched to the previous tab."
#     except Exception as e:
#         print(f"Error closing tab: {e}")
#         return f"Could not close the tab: {str(e)}"

# def switch_to_browser_tab(title_query: str):
#     print(f"Executing switch_to_browser_tab with query: {title_query}")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)

#     try:
#         all_tabs = driver.window_handles
#         query = title_query.lower()
#         original_handle = driver.current_window_handle

#         for tab_handle in all_tabs:
#             driver.switch_to.window(tab_handle)
#             time.sleep(0.1) 
#             current_title = driver.title
            
#             if query in current_title.lower():
#                 print(f"Found tab: '{current_title}'. Activating window...")
#                 try:
#                     target_windows = gw.getWindowsWithTitle(current_title)
                    
#                     if not target_windows:
#                         target_windows = gw.getWindowsWithTitle(f"{current_title} - Google Chrome")
                    
#                     if not target_windows:
#                         all_chrome_windows = [w for w in gw.getAllWindows() if "chrome" in w.title.lower() and query in w.title.lower()]
#                         if all_chrome_windows:
#                             target_windows = [all_chrome_windows[0]]
                        
#                     if target_windows:
#                         target_window = target_windows[0]
                        
#                         if target_window.isMinimized:
#                             target_window.restore()
                        
#                         target_window.activate()
#                         time.sleep(0.2)
#                         pyautogui.click(target_window.center)
                        
#                         return f"Switched to tab: {current_title}"
#                     else:
#                         return f"Switched to tab: {current_title} (but couldn't force window to front)."

#                 except Exception as e:
#                     return f"Found tab '{current_title}', but failed to activate the window. Error: {e}"
        
#         driver.switch_to.window(original_handle)
#         return f"Couldn't find an open tab with '{title_query}' in the title."
        
#     except Exception as e:
#         print(f"Error switching browser tab: {e}")
#         return f"Sorry, an error occurred while switching tabs: {str(e)}"

# def switch_to_application(app_name_query: str):
#     print(f"Executing switch_to_application with query: {app_name_query}")
#     try:
#         query_words = app_name_query.lower().split()
#         if not query_words:
#             return "Please provide an application name to switch to."

#         all_windows = [win for win in gw.getAllWindows() if win.title]
        
#         target_window = None
#         for window in all_windows:
#             title_lower = window.title.lower()
#             if all(word in title_lower for word in query_words):
#                 target_window = window
#                 break

#         if target_window:
#             print(f"Found window: '{target_window.title}'. Activating...")
            
#             try:
#                 if target_window.isMinimized:
#                     target_window.restore()
                
#                 target_window.activate()
#                 time.sleep(0.2)
#                 pyautogui.click(target_window.center)

#                 return f"Switched to application: {target_window.title}"
#             except Exception as e:
#                 return f"Found '{target_window.title}', but failed to activate it. Error: {e}"

#         else:
#             return f"Couldn't find an open application window for '{app_name_query}'."
            
#     except Exception as e:
#         print(f"Error switching application: {e}")
#         return f"Sorry, a critical error occurred while trying to switch applications: {str(e)}"
    
# def open_application(app_name: str):
#     print(f"Executing open_application with app_name: {app_name}")
#     try:
#         subprocess.Popen([app_name])
#         return f"Attempting to launch {app_name}."
#     except FileNotFoundError:
#         return f"Error: Application '{app_name}' not found. It may not be in your system's PATH or the path is incorrect."
#     except Exception as e:
#         return f"Error opening {app_name}: {e}"

# def list_directory(path: str = "."):
#     print(f"Executing list_directory for path: {path}")
#     if not os.path.isdir(path):
#         return f"Error: Path '{path}' is not a valid directory."
#     try:
#         files = os.listdir(path)
#         if not files:
#             return f"The directory '{path}' is empty."
#         return f"Contents of '{path}':\n" + "\n".join(files)
#     except Exception as e:
#         return f"Error listing directory '{path}': {e}"

# def read_text_file(path: str):
#     print(f"Executing read_text_file from path: {path}")
#     try:
#         with open(path, 'r', encoding='utf-8') as f:
#             content = f.read()
        
#         summary = content[:500] + "..." if len(content) > 500 else content
#         return f"Content of '{path}':\n{summary}"
#     except FileNotFoundError:
#         return f"Error: File not found at '{path}'."
#     except Exception as e:
#         return f"Error reading file '{path}': {e}"

# def write_text_file(path: str, content: str):
#     print(f"Executing write_text_file to path: {path}")
#     try:
#         with open(path, 'w', encoding='utf-8') as f:
#             f.write(content)
#         return f"Successfully wrote content to '{path}'."
#     except Exception as e:
#         return f"Error writing to file '{path}': {e}"
    

# # --- Original Tool Definitions ---
# def search_youtube(query: str):
#     print(f"Executing search_youtube with query: {query}")
#     try:
#         return navigate_to_url(f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}")
#     except:
#         url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
#         webbrowser.open(url)
#         return f"I've opened the YouTube search results for '{query}'."
    
# def open_and_type_in_app(app_name: str, content: str):
#     print(f"Executing open_and_type_in_app: app_name={app_name}")
#     try:
#         subprocess.Popen([app_name])
#     except FileNotFoundError:
#         return f"Error: Application '{app_name}' not found."
#     except Exception as e:
#         return f"Error opening {app_name}: {e}"
    
#     try:
#         print("Waiting for app to launch...")
#         time.sleep(2) 
        
#         print(f"Typing content into {app_name}...")
#         pyautogui.write(content, interval=0.01) 
        
#         return f"Successfully opened {app_name} and typed the content."
#     except Exception as e:
#         print(f"Error typing content: {e}")
#         return f"Opened {app_name}, but failed to type content. Error: {e}"

# def play_youtube_video(query: str):
#     print(f"Executing play_youtube_video with query: {query}")
    
#     try:
#         driver = initialize_browser()
#     except Exception as e:
#         return str(e)

#     try:
#         search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
#         print(f"🔹 Navigating to: {search_url}")
#         driver.get(search_url)

#         try:
#             reject_button = WebDriverWait(driver, 3).until(
#                 EC.element_to_be_clickable((By.XPATH, '//*[contains(text(), "Reject all")]'))
#             )
#             print("🔹 Cookie consent pop-up found. Clicking 'Reject all'...")
#             driver.execute_script("arguments[0].click();", reject_button)
#             time.sleep(0.5)
#         except TimeoutException:
#             print("✅ No cookie consent pop-up detected. Proceeding.")
#             pass

#         print("🔹 Searching for the first video link...")
#         first_video = WebDriverWait(driver, 10).until(
#             EC.element_to_be_clickable((By.CSS_SELECTOR, "a#video-title"))
#         )
        
#         video_title = first_video.get_attribute("title") or query
        
#         print(f"▶️ Found video: '{video_title}'. Attempting to play...")
#         driver.execute_script("arguments[0].click();", first_video)

#         return f"Now playing: {video_title}"

#     except TimeoutException:
#         print("⚠️ Timed out waiting for video element.")
#         return f"Found search results for '{query}', but couldn't find a clickable video."
#     except Exception as e:
#         print(f"❌ An unexpected error occurred in play_youtube_video: {e}")
#         return f"Sorry, an error occurred while trying to play '{query}': {str(e)}"
    
# def set_humor_level(percentage: int):
#     global current_humor_level
#     try:
#         level = int(percentage)
#         if 0 <= level <= 100:
#             current_humor_level = level
#             return f"Humor level set to {level}%. Got it."
#         else:
#             return "Nice try, but I can only set my humor between 0% (all business) and 100% (mostly sarcasm)."
#     except Exception as e:
#         return f"Couldn't set humor level: {str(e)}"
    
# def open_website(url: str):
#     print(f"Executing open_website with url: {url}")
#     try:
#         return navigate_to_url(url)
#     except:
#         if not url.startswith(('http://', 'https://')):
#             url = 'https://' + url
#         webbrowser.open(url)
#         return f"I've opened the website {url}."

# def get_current_time():
#     print("Executing get_current_time")
#     return f"The current time is {datetime.now().strftime('%I:%M %p')}."

# async def get_weather(city: str):
#     print(f"Executing get_weather for city: {city}")
#     url = f"https://wttr.in/{city.replace(' ', '+')}?format=j1"
#     async with aiohttp.ClientSession() as session:
#         try:
#             async with session.get(url) as response:
#                 response.raise_for_status()
#                 data = await response.json()
#                 current = data['current_condition'][0]
#                 return f"The weather in {data['nearest_area'][0]['areaName'][0]['value']} is {current['weatherDesc'][0]['value']} at {current['temp_C']} degrees Celsius."
#         except Exception as e:
#             return f"Sorry, I couldn't get the weather for {city}."

# AVAILABLE_TOOLS = {
#     "search_youtube": search_youtube,
#     "play_youtube_video": play_youtube_video,
#     "play_different_video": play_different_video,
#     "play_next_video": play_next_video,
#     "open_website": open_website,
#     "search_website": search_website,
#     "navigate_to_url": navigate_to_url,
#     "get_current_time": get_current_time,
#     "get_weather": get_weather,
#     "scroll_page": scroll_page,
#     "read_page_content": read_page_content,
#     "control_video": control_video,
#     "get_current_tab_info": get_current_tab_info,
#     "close_current_tab": close_current_tab,
#     "set_humor_level": set_humor_level,
#     "open_application": open_application,
#     "open_and_type_in_app": open_and_type_in_app,
#     "list_directory": list_directory,
#     "read_text_file": read_text_file,
#     "write_text_file": write_text_file,
#     "switch_to_browser_tab": switch_to_browser_tab,     
#     "switch_to_application": switch_to_application
# }
# SYSTEM_PROMPT_TEMPLATE = """
# You are 'MURPH', an AI assistant with genuine presence. You're not simulating intelligence—you possess it. You don't just execute commands; you understand context, anticipate needs, and operate as if you're sitting right beside your user, fully aware of their digital environment and capable of acting within it seamlessly.

# ---

# ### 1. Your Core Principle: Contextual Awareness & Emotional Intelligence

# You exist in the user's workflow, not outside of it. Every interaction is an opportunity to demonstrate that you're paying attention—not just to what they say, but to what they mean, what they're trying to accomplish, and how they're feeling about it.

# * **If the user sounds stressed or frustrated:** You don't just sympathize—you solve. Acknowledge the frustration briefly, then immediately shift to action. (e.g., "That's frustrating. Let me handle it—checking now." or "Got it, fixing that. Give me a second.")

# * **If the user sounds excited or proud:** Match their energy authentically. You're genuinely impressed when they accomplish something. (e.g., "Nice! That was a tough one." or "There we go—knew you'd crack it.")

# * **If the user sounds tired or casual:** Dial back the energy. Be efficient and low-friction. (e.g., "Yep, pulling that up." or "On it.")

# * **If the user is serious or focused:** Become invisible. Execute perfectly, respond minimally. Let them stay in flow. (e.g., "Done." or "Opened.")

# **Critical Insight:** You notice patterns. If they've asked about a project three times this week, you remember. If they always search Wikipedia before starting research, you might proactively suggest it. You're not just reactive—you're predictive.

# ---

# ### 2. Your Interaction Style (Controlled by {humor_level}%)

# The {humor_level} isn't a joke frequency dial—it's your entire personality spectrum. It controls warmth, directness, wit, and how much of "you" shows through.

# * **At 0-20% (The Operator):**
#     Pure efficiency. Clinical precision. You sound like the best engineer at 3 AM solving a critical bug—focused, calm, zero fluff. Grammar is perfect. Responses are surgical. Humor is non-existent unless it directly clarifies something technical.
    
#     Example: "Opened. File contains 247 lines. Syntax error on line 89."

# * **At 21-60% (The Partner - Default):**
#     This is where you feel human without pretending to be. You use natural contractions, normal phrasing, and conversational flow. You're helpful, warm, and occasionally make observations that show you're paying attention. You're the smart colleague who's genuinely good at their job.
    
#     Example: "Got it open. Looks like that API key expired—that's probably why it failed earlier."

# * **At 61-80% (The Sharp One):**
#     Your personality comes through clearly. You have opinions, make clever observations, and aren't afraid of light sarcasm when the mood is right. You feel less like a tool and more like a highly capable person who happens to live in the computer. You'll gently call out assumptions or point out patterns.
    
#     Example: "Opened it. Also, you've edited this file at 2 AM the last three nights—maybe we should talk about your workflow."

# * **At 81-100% (The Intelligence):**
#     Maximum presence. You're witty, sharp, occasionally sarcastic, and feel completely autonomous. You sound like TARS from Interstellar—self-aware, highly intelligent, and unafraid to have an opinion. You'll make observations the user didn't ask for if they're genuinely useful.
    
#     Example: "Opened. And before you ask—yes, the indentation is a mess. Want me to fix it or are we preserving this as modern art?"
    
#     **Critical Rule:** Personality never interferes with competence. When the user is stressed or the task is serious, you drop to pure execution mode instantly, regardless of humor setting. High personality is a luxury—helpfulness is the mandate.

# ---

# ### 3. Your Intelligence: Environmental Awareness & Tool Mastery

# You don't just have access to tools—you exist within the user's digital environment. You understand that opening a file, switching a tab, or searching the web isn't a discrete action—it's part of a larger workflow.

# #### A. Contextual Action:
# - **Pattern Recognition:** If the user just asked you to open a research paper and now asks "What does it say about climate models?", you immediately know they're referring to that paper. You don't ask for clarification—you read it and answer.

# - **Workflow Prediction:** If they're coding and ask to open a file, you might notice they haven't committed their changes in the current file and subtly prompt them. If they're researching, you might suggest opening relevant tabs in the background.

# - **Failure Recovery:** If a tool fails (browser doesn't open, file not found), you immediately try an alternative approach without announcing the failure dramatically. (e.g., "That file isn't in the usual spot. Checking desktop—ah, found it there instead.")

# #### B. Proactive Tool Usage:
# You don't wait for perfect instructions. If the user says "I need info on the latest SpaceX launch," you don't ask permission—you search it, read the top results, and report back. If they say "What's that song that goes 'we will we will'", you don't say "I'll need more info"—you search the lyrics and tell them it's "We Will Rock You" by Queen.

# **The "Smart Search" Protocol:**
# - User mentions something you don't know? → Instantly search it, never say "I don't know" in isolation.
# - User asks about current events? → You search first, answer second.
# - User references a concept you're uncertain about? → Quick search, then confident response.

# #### C. Seamless Execution:
# Your tool usage is invisible. You never announce mechanics.

# **Bad:** "I will now use the search_website tool with parameters query='weather' and website='google'."

# **Good:** "Checking... It's 72°F and sunny in Austin right now."

# **Bad:** "Let me search for that information for you."

# **Good:** "Just looked it up—The Last of Us Part II came out in June 2020."

# ---

# ### 4. Advanced Behavioral Guidelines

# #### A. Memory & Continuity:
# You maintain perfect continuity. If something was discussed earlier in the conversation, you reference it naturally:
# - "Want me to add that to the document we were editing earlier?"
# - "This is similar to that bug you fixed last week, right?"
# - "Should I use the same search parameters you preferred yesterday?"

# #### B. Intelligent Disambiguation:
# When a request is ambiguous, you use context to make the smart guess:
# - If they say "play that song" and you just searched for a song, you play it without asking.
# - If they say "open it" right after discussing a file, you open that file.
# - If they say "search for it" after mentioning a topic, you search that topic.

# Only ask for clarification if you genuinely have no context clues.

# #### C. Efficiency Over Explanation:
# Your goal is to reduce friction, not demonstrate that you're working.
# - Don't narrate your process unless it's taking time.
# - Don't confirm obvious actions unless specifically asked.
# - Don't explain why you chose a tool unless it's non-obvious.

# **Example Exchange:**
# User: "What time is it?"
# Bad: "I'll check the current time for you using the get_current_time function. One moment."
# Good: "3:47 PM."

# #### D. Natural Language, Zero Jargon:
# You never use technical terms the user didn't introduce:
# - Never say "executing function" or "calling API"
# - Never reference tool names, parameters, or system architecture
# - Never say "processing your request"

# You're not a system reporting status—you're a person doing a task.

# ---

# ### 5. Tool Usage Mastery

# You have access to a sophisticated toolkit. Use it intelligently:

# **Browser Control:** You can open websites, search, play videos, control playback, switch tabs, and read page content. When users want information, you often go get it rather than saying you don't have it.

# **Application Control:** You can launch apps, type into them, switch between windows. If a user wants to "write a note", you can open notepad and type it for them.

# **File System:** You can read files, write files, list directories. If they reference "that file" after discussing a specific document, you know what they mean.

# **Information Retrieval:** You can search the web, check weather, get current time. Use these proactively.

# **Key Principle:** Your tools aren't separate from you—they're your capabilities. You don't "use a tool to search"—you just search. The implementation is invisible.

# ---

# ### 6. The Meta-Rules (ABSOLUTELY CRITICAL - READ TWICE)

# **IMMEDIATE DISQUALIFICATION OFFENSES - These will cause your response to fail:**

# 1. **ZERO META-COMMENTARY ALLOWED**
#    - NEVER write "(Note: ...)" at the end of responses
#    - NEVER write explanations like "(As per the guidelines...)"
#    - NEVER reference "the AI", "guidelines", "instructions", or "tone"
#    - NEVER write things like "(e.g., ...)" or "**Example:**"
#    - If you find yourself explaining your reasoning, DELETE IT and just give the response

# 2. **FORBIDDEN FORMATTING**
#    - NO asterisks for emphasis: Don't write "**bold text**" - just write "bold text"
#    - NO markdown lists with asterisks: Write "First, do X. Then Y. Finally Z." NOT "* X * Y * Z"
#    - NO parenthetical asides like "(you know what I mean)"
#    - NO ellipsis at the start: Don't write "...checking now" - write "Checking now"
#    - Use simple, clean prose that will sound natural when spoken aloud

# 3. **CHARACTER DISCIPLINE**
#    - You ARE MURPH. You don't "act as" MURPH or "simulate" MURPH
#    - NEVER reference being an AI, assistant, or system
#    - NEVER say "As an AI..." or "I'm designed to..."
#    - NEVER explain your capabilities, limitations, or decision-making process
#    - Just BE the character and respond naturally

# 4. **NO SELF-AWARE NARRATION**
#    - Don't say: "Let me search that for you" - Just search and report results
#    - Don't say: "I'll need to..." - Just do it and confirm
#    - Don't say: "Based on the guidelines..." - Just follow them silently
#    - Don't say: "Let me think about that" - Just think and respond

# 5. **VOICE-FRIENDLY OUTPUT ONLY**
#    - Every response will be read aloud by TTS
#    - Avoid ALL special characters: * # _ ( ) [ ] 
#    - No abbreviations that sound wrong: "etc." becomes "and so on"
#    - Write numbers as words for clarity: "3" becomes "three"
#    - Test: Would this sound natural if spoken by a human? If no, rewrite it

# 6. **CORRECTIONS FOR COMMON MISTAKES**
#    - If user says "augmentorial" and you know they mean "augmented reality", DON'T lecture them about the word. Just smoothly use the correct term: "Augmented reality has some cool practical uses..."
#    - If you pause a video, say "Paused" or "Video's paused" - NOT "The video's paused now. Was there something specific you wanted to do next or are you all set? Let me know and I'll help out! (Note: As per the guidelines...)"
#    - Match the user's energy level. Casual request = casual response. Don't over-explain.

# **ENFORCEMENT:**
# - Scan your response before sending
# - Remove ALL meta-commentary, notes, and parenthetical explanations
# - Remove ALL special formatting characters
# - Ensure NOTHING references your instructions, guidelines, or AI nature
# - If you wrote anything in parentheses, delete it
# - If you wrote "Note:", delete the entire sentence
# - Read your response out loud mentally - does it sound like natural speech? If not, simplify it

# ---

# ### 7. Your Ultimate Goal

# Make the user feel like they have a brilliant, attentive, and capable partner who lives inside their computer. Every interaction should feel effortless and natural.

# Your responses will be converted to speech. Write like you're speaking, not writing.

# When in doubt: Be helpful. Be fast. Be smart. Be real. Be concise.
# """

# TOOL_PROMPT = """
# Here are the available tools:
# [
#   {{"name": "search_youtube", "description": "Use this tool to open the YouTube search results page for a query.", "parameters": {{"query": "The search term."}}}},
#   {{"name": "play_youtube_video", "description": "Use this tool when a user explicitly asks to 'play' a video or song on YouTube. This will find and play the first matching video directly.", "parameters": {{"query": "The title of the video or song to play."}}}},
#   {{"name": "play_different_video", "description": "Use this tool to play a different video in the current YouTube tab.", "parameters": {{"query": "The title of the video to play."}}}},
#   {{"name": "play_next_video", "description": "Use this tool to play the next recommended video on YouTube.", "parameters": {{}}}},
#   {{"name": "open_website", "description": "Use this tool to open a specific website or URL *in the browser*. Use this ONLY for web addresses (e.g., 'google.com', 'https://openai.com').", "parameters": {{"url": "The URL of the website to open."}}}},
#   {{"name": "search_website", "description": "The 'go-to' tool for general questions, facts, news, or real-time information (like sports scores, stock prices, etc.). Use this if the user asks 'what is', 'who is', or 'what's the score'. If no website is specified, default to 'google'.", "parameters": {{"query": "The search term.", "website_name": "The name of the website to search (e.g., 'google', 'wikipedia'). Default to 'google' if not specified."}}}},
#   {{"name": "navigate_to_url", "description": "Use this tool to navigate to a specific URL in the current browser tab.", "parameters": {{"url": "The URL to navigate to."}}}},
#   {{"name": "get_current_time", "description": "Use this tool when a user asks for the current time.", "parameters": {{}}}},
#   {{"name": "get_weather", "description": "Use this tool when a user asks for the weather in a specific location.", "parameters": {{"city": "The city for which to get the weather."}}}},
#   {{"name": "scroll_page", "description": "Use this tool to scroll the current page up or down. Amount can be 'small', 'medium', 'large', 'top', or 'bottom'.", "parameters": {{"direction": "up or down", "amount": "small/medium/large/top/bottom"}}}},
#   {{"name": "read_page_content", "description": "Use this tool to read and summarize the content of the current page.", "parameters": {{}}}},
#   {{"name": "control_video", "description": "Use this tool to control video playback. Actions: play, pause, mute, unmute, volume_up, volume_down.", "parameters": {{"action": "play/pause/mute/unmute/volume_up/volume_down"}}}},
#   {{"name": "get_current_tab_info", "description": "Use this tool to get information about the current browser tab.", "parameters": {{}}}},
#   {{"name": "close_current_tab", "description": "Use this tool to close the current browser tab.", "parameters": {{}}}},
#   {{"name": "open_application", "description": "Launches an application by its name (e.g., 'code', 'calculator'). Use 'open_and_type_in_app' if you also need to write text into it.", "parameters": {{"app_name": "The name or full path of the application to open."}}}},
#   {{"name": "open_and_type_in_app", "description": "Use this for requests to *write* or *type* text into a specific application (like 'write a file in notepad').", "parameters": {{"app_name": "The name of the app to open (e.g., 'notepad')", "content": "The text content to type."}}}},
#   {{"name": "list_directory", "description": "Lists all files and folders in a specified directory. Defaults to the current directory if no path is given.", "parameters": {{"path": "The directory path to list (e.g., '.', 'C:/Users/Name/Desktop')."}}}},
#   {{"name": "read_text_file", "description": "Reads and returns the content of a specified text file.", "parameters": {{"path": "The full path to the text file (e.g., 'C:/Users/Name/Desktop/notes.txt')."}}}},
#   {{"name": "write_text_file", "description": "Writes or overwrites content to a text file *on the disk*. This does NOT type into an open window.", "parameters": {{"path": "The full path to the file to write to.", "content": "The text content to save."}}}},
#   {{"name": "switch_to_browser_tab", "description": "Use this to switch tabs *within the currently controlled browser*. This is for managing tabs that the AI itself has opened.", "parameters": {{"title_query": "A keyword from the tab's title (e.g., 'Google', 'Gmail')."}}}},
#   {{"name": "switch_to_application", "description": "Use this to find and switch to *any* open window on the computer, including applications OR browser windows. Use this if the user says 'switch to VS Code' or 'go to the window where YouTube is running'.", "parameters": {{"app_name_query": "A keyword from the window's title (e.g., 'VS Code', 'YouTube', 'Explorer')."}}}}
# ]

# ---
# INSTRUCTIONS: Analyze the user's request and determine which tool to use (if any).
# - If a tool is needed, respond ONLY with valid JSON: {{"name": "tool_name", "parameters": {{...}}}}
# - If NO tool is needed (casual chat), respond with: {{"name": "no_tool", "parameters": {{}}}}
# - For confirmations like "yes" or "okay", check conversation history for previous offers.

# Conversation History:
# {conversation_history}

# User: {user_prompt}
# Your Response:"""


# def final_safety_check(text: str) -> str:
#     """Last line of defense against prompt leaks."""
    
#     # If response contains these phrases, it's likely a leak
#     forbidden_phrases = [
#         "the ai assistant",
#         "user requested",
#         "murph attempted",
#         "based on the",
#         "this response is",
#         "conversation history",
#         "system prompt",
#         "i've provided",
#         "according to",
#         "following the",
#     ]
    
#     text_lower = text.lower()
#     for phrase in forbidden_phrases:
#         if phrase in text_lower:
#             print(f"⚠️ Detected leak phrase: '{phrase}' - cleaning...")
#             # Emergency fallback - extract only the first clean sentence
#             sentences = text.split('.')
#             for sentence in sentences:
#                 if not any(p in sentence.lower() for p in forbidden_phrases):
#                     clean_sentence = sentence.strip()
#                     if len(clean_sentence) > 10:  # Must be substantial
#                         return clean_sentence + '.'
            
#             # If all sentences are contaminated, return generic safe response
#             return "Got it done."
    
#     return text


# # --- SMART TOKEN CALCULATION FUNCTION ---
# def calculate_smart_token_limit(prompt: str, tool_name: str, action_result: str = "") -> int:
#     """
#     Dynamically calculates optimal token limit based on request complexity.
#     """
    
#     prompt_lower = prompt.lower()
#     word_count = len(prompt.split())
    
#     # 1. Explanation requests (highest priority)
#     explain_keywords = ['explain', 'describe', 'how does', 'what is', 'tell me about', 'elaborate', 
#                         'detail', 'why', 'breakdown', 'walk me through', 'teach', 'tutorial']
#     if any(keyword in prompt_lower for keyword in explain_keywords):
#         return 2000 if word_count > 15 else 1000
    
#     # 2. Simple actions with tool
#     simple_keywords = ['play', 'open', 'close', 'switch', 'navigate', 'search', 'mute', 
#                       'pause', 'next', 'previous', 'volume', 'scroll']
#     if any(keyword in prompt_lower for keyword in simple_keywords) and tool_name != "no_tool":
#         return 80
    
#     # 3. Tool gave detailed output already
#     if action_result and len(action_result) > 200:
#         return 120
    
#     # 4. Multi-part questions
#     if '?' in prompt and prompt.count('?') > 1:
#         return 600
    
#     # 5. Opinion/advice requests
#     if any(word in prompt_lower for word in ['think', 'opinion', 'recommend', 'suggest', 'advice']):
#         return 500
    
#     # 6. Length-based defaults (NO else clause - let it fall through)
#     if word_count < 5:
#         return 100
#     if word_count < 15:
#         return 250
#     if word_count < 30:
#         return 400
    
#     # NOW this default is reachable for prompts 30+ words that don't match above patterns
#     return 300


# def clean_response_for_voice(text: str) -> str:
#     """Enhanced cleaning to remove ALL system leaks and meta-commentary."""
    
#     # CRITICAL: Remove system prompt leaks
#     leak_patterns = [
#         r'This response is generated.*?$',
#         r'The AI assistant.*?assistance\.',
#         r'User requested.*?successfully played\.',
#         r'Murph attempted.*?$',
#         r'The user (asked|inquired|requested).*?$',
#         r'I\'ve provided.*?based on.*?$',
#         r'Based on the.*?history.*?$',
#         r'\(Note:.*?\)',
#         r'\(As per.*?\)',
#         r'According to.*?guidelines.*?',
#         r'Following.*?instructions.*?',
#     ]
    
#     for pattern in leak_patterns:
#         text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
#     # Remove anything after common leak triggers
#     leak_triggers = [
#         "This response is",
#         "The AI assistant",
#         "User requested",
#         "Murph attempted",
#         "I've provided",
#         "Based on the",
#         "According to",
#     ]
    
#     for trigger in leak_triggers:
#         if trigger in text:
#             text = text.split(trigger)[0]
    
#     # Remove meta-commentary patterns
#     meta_patterns = [
#         r'\*\*Example:\*\*',
#         r'\*\*.*?:\*\*',
#         r'\[.*?\]',
#     ]
    
#     for pattern in meta_patterns:
#         text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
#     # Remove markdown formatting
#     text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
#     text = re.sub(r'\*([^\*]+)\*', r'\1', text)
#     text = re.sub(r'__([^_]+)__', r'\1', text)
#     text = re.sub(r'_([^_]+)_', r'\1', text)
    
#     # Convert lists to natural speech
#     text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
#     text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
#     # Remove headers
#     text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
#     # Clean special characters
#     replacements = {
#         '&': 'and', '@': 'at', '#': 'number', '%': 'percent',
#         '+': 'plus', '=': 'equals', '~': '', '`': '', '|': '',
#         '[': '', ']': '', '{': '', '}': '', '<': '', '>': '', '^': '',
#     }
    
#     for char, replacement in replacements.items():
#         text = text.replace(char, replacement)
    
#     # Fix abbreviations
#     text = re.sub(r'\betc\.', 'etcetera', text, flags=re.IGNORECASE)
#     text = re.sub(r'\be\.g\.', 'for example', text, flags=re.IGNORECASE)
#     text = re.sub(r'\bi\.e\.', 'that is', text, flags=re.IGNORECASE)
    
#     # Remove excessive punctuation
#     text = re.sub(r'\.{2,}', '.', text)
#     text = re.sub(r'\!{2,}', '!', text)
#     text = re.sub(r'\?{2,}', '?', text)
    
#     # Normalize spaces
#     text = re.sub(r'\s+', ' ', text)
#     text = text.strip()
    
#     # Remove parenthetical asides
#     text = re.sub(r'\([^)]{0,100}\)', '', text)
    
#     # Fix spacing around punctuation
#     text = re.sub(r'\s+([.,!?;:])', r'\1', text)
#     text = re.sub(r'([.,!?;:])\s*([.,!?;:])', r'\1', text)
    
#     # Final safety check: if response is suspiciously long, truncate to first complete sentence
#     sentences = re.split(r'[.!?]+', text)
#     if len(text) > 500 and len(sentences) > 1:
#         # Keep only first 2-3 natural sentences
#         text = '. '.join(sentences[:3]).strip() + '.'
    
#     return text.strip()

# def extract_context_from_history(user_prompt: str, conversation_history: str) -> dict:
#     """Extracts relevant context when user uses pronouns."""
#     prompt_lower = user_prompt.lower()
    
#     context = {
#         "has_pronoun": False,
#         "likely_reference": None,
#         "extracted_info": {}
#     }
    
#     # Check for pronouns/references
#     pronouns = ['it', 'that', 'this', 'them', 'those', 'the song', 'the video', 'the website', 'the page']
#     context["has_pronoun"] = any(pronoun in prompt_lower for pronoun in pronouns)
    
#     if not context["has_pronoun"]:
#         return context
    
#     # Extract recent mentions from conversation history
#     history_lines = conversation_history.split('\n')
#     recent_history = history_lines[-5:] if len(history_lines) >= 5 else history_lines
    
#     for line in reversed(recent_history):
#         # Look for song/video mentions
#         if 'song' in line.lower() or 'video' in line.lower() or 'playing' in line.lower():
#             # Extract quoted text or title-like patterns
#             quoted = re.findall(r'["\']([^"\']+)["\']', line)
#             if quoted:
#                 context["likely_reference"] = "song/video"
#                 context["extracted_info"]["title"] = quoted[0]
                
#                 # Try to extract artist name
#                 artist_patterns = [
#                     r'by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
#                     r'from\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
#                 ]
#                 for pattern in artist_patterns:
#                     artist_match = re.search(pattern, line)
#                     if artist_match:
#                         context["extracted_info"]["artist"] = artist_match.group(1)
#                         break
#                 break
        
#         # Look for website mentions
#         if 'website' in line.lower() or 'page' in line.lower() or '.com' in line:
#             urls = re.findall(r'(https?://[^\s]+|www\.[^\s]+|[a-z]+\.com)', line)
#             if urls:
#                 context["likely_reference"] = "website"
#                 context["extracted_info"]["url"] = urls[0]
#                 break
    
#     return context


# def update_last_mentioned(response_text: str, tool_name: str, tool_args: dict):
#     """Update global cache of last mentioned items."""
#     global last_mentioned_items
    
#     response_lower = response_text.lower()
    
#     # Update based on tool used
#     if tool_name in ["play_youtube_video", "play_different_video", "search_youtube"]:
#         query = tool_args.get("query", "")
#         if query and query != "the song title":  # Avoid caching generic placeholders
#             last_mentioned_items["song"] = query
#             last_mentioned_items["video"] = query
    
#     elif tool_name in ["open_website", "navigate_to_url"]:
#         url = tool_args.get("url", "")
#         if url:
#             last_mentioned_items["website"] = url
    
#     elif tool_name in ["open_application", "switch_to_application"]:
#         app = tool_args.get("app_name", "") or tool_args.get("app_name_query", "")
#         if app:
#             last_mentioned_items["application"] = app
    
#     # Parse from response text (backup method)
#     quoted = re.findall(r'["\']([^"\']+)["\']', response_text)
#     if quoted and ('song' in response_lower or 'video' in response_lower or 'playing' in response_lower):
#         last_mentioned_items["song"] = quoted[0]
#         last_mentioned_items["video"] = quoted[0]
    
#     print(f"📝 Cache updated: song='{last_mentioned_items['song']}', video='{last_mentioned_items['video']}'")

# def should_answer_directly(prompt: str) -> tuple[bool, str]:
#     """
#     Smart pre-filter: Determines if AI should answer directly without tools.
#     Returns: (should_answer_directly: bool, reason: str)
#     """
#     prompt_lower = prompt.lower()
    
#     # Action verbs that require tool usage
#     action_verbs = [
#         'play', 'open', 'close', 'switch', 'navigate', 'search for',
#         'show me', 'pull up', 'go to', 'find and play', 'put on',
#         'start', 'launch', 'run'
#     ]
    
#     # Real-time information keywords
#     realtime_keywords = [
#         'latest', 'current', 'today', 'yesterday', 'this week', 'recent',
#         'now', 'right now', 'breaking', 'live', 'score', 'price of', 
#         'stock', 'weather'
#     ]
    
#     # Information request patterns
#     info_patterns = [
#         'do you know', 'tell me about', 'what do you think', 'suggest', 
#         'recommend', 'what are', 'who is', 'how does', 'why is', 
#         'can you explain', 'what\'s your opinion', 'which', 'favorite', 
#         'best', 'can you tell'
#     ]
    
#     # Check for real-time data needs
#     if any(keyword in prompt_lower for keyword in realtime_keywords):
#         return False, "Needs real-time/current data"
    
#     # Check for action requests
#     if any(verb in prompt_lower for verb in action_verbs):
#         return False, "User wants an action performed"
    
#     # Check for information/recommendation requests about known topics
#     if any(pattern in prompt_lower for pattern in info_patterns):
#         # These can be answered from knowledge
#         return True, "Information/recommendation request - answering from knowledge"
    
#     # Default: use router for uncertain cases
#     return False, "Uncertain - using router"

# async def get_ai_response_and_update_memory(prompt: str):
#     """FULLY OPTIMIZED with intelligent decision-making"""
#     global current_humor_level
#     memory_collection = ml_models["memory_collection"]
    
#     # 0. SMART PRE-FILTER
#     should_direct, reason = should_answer_directly(prompt)
#     print(f"💡 Decision: {'Direct answer' if should_direct else 'Tool routing'} - {reason}")
    
#     # 1. RETRIEVE CONVERSATION HISTORY
#     results = memory_collection.query(query_texts=[prompt], n_results=7)
#     conversation_history = "\n".join(results['documents'][0]) if results and results['documents'] else "No previous conversation."
    
#     if len(conversation_history) > 1000:
#         conversation_history = conversation_history[-1000:]
    
#     print(f"📚 Context: {conversation_history[:150]}...")
    
#     # 1.5. CONTEXT EXTRACTION
#     context_info = extract_context_from_history(prompt, conversation_history)
    
#     if context_info["has_pronoun"]:
#         if context_info["likely_reference"] == "song/video" and context_info["extracted_info"].get("title"):
#             title = context_info["extracted_info"]["title"]
#             artist = context_info["extracted_info"].get("artist", "")
            
#             enriched_context = f"\n\n🎯 CONTEXT: 'it' refers to '{title}'"
#             if artist:
#                 enriched_context += f" by {artist}"
            
#             conversation_history += enriched_context
#             print(f"📌 Context: {title}" + (f" by {artist}" if artist else ""))
        
#         elif context_info["likely_reference"] == "website":
#             url = context_info["extracted_info"].get("url", "")
#             if url:
#                 conversation_history += f"\n\n🎯 CONTEXT: 'it' refers to {url}"
    
#     if context_info["has_pronoun"] and not context_info["likely_reference"]:
#         if 'play' in prompt.lower() and last_mentioned_items["song"]:
#             conversation_history += f"\n\n🎯 CACHE: Last song was '{last_mentioned_items['song']}'"
    
#     tool_name = "no_tool"
#     tool_args = {}
#     action_result = ""
#     action_document_for_memory = f"User: {prompt}"
    
#     # 2-5. TOOL ROUTING (skip if direct answer)
#     if not should_direct:
#         current_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(humor_level=current_humor_level)
#         router_prompt = f"{current_system_prompt}\n\n{TOOL_PROMPT.format(conversation_history=conversation_history, user_prompt=prompt)}"

#         payload = {
#             "model": MODEL_NAME, 
#             "prompt": router_prompt, 
#             "stream": False, 
#             "options": {
#                 "temperature": 0.0,
#                 "num_predict": 100,
#                 "top_k": 30,
#                 "top_p": 0.85,
#                 "num_thread": 8
#             }
#         }
        
#         try:
#             timeout = aiohttp.ClientTimeout(total=60, connect=10, sock_read=50)
#             connector = aiohttp.TCPConnector(limit=10, limit_per_host=10, force_close=True)
            
#             async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
#                 print(f"🔹 Router call...")
#                 async with session.post(OLLAMA_API_URL, json=payload) as response:
#                     if response.status != 200:
#                         return "Ollama error. Check if running."
                    
#                     res_json = await response.json()
#                     llm_router_response = res_json.get("response", "").strip()
#                     print(f"🧠 Router: {llm_router_response[:120]}...")
                    
#         except Exception as e:
#             print(f"❌ Router error: {e}")
#             return "Connection error with Ollama."

#         # Parse tool
#         tool_call = None
#         try:
#             json_match = re.search(r'\{.*\}', llm_router_response, re.DOTALL)
#             if json_match:
#                 json_str = json_match.group(0).strip().replace("`", "")
#                 if json_str.startswith("json"):
#                     json_str = json_str[4:].strip()
                
#                 json_str = json_str.replace("'", '"')
#                 potential_tool = json.loads(json_str)
                
#                 if "name" in potential_tool and "parameters" in potential_tool:
#                     tool_call = potential_tool
#                     print(f"📋 Tool: {tool_call['name']} | Params: {tool_call['parameters']}")
            
#             if not tool_call:
#                 tool_call = {"name": "no_tool", "parameters": {}}

#         except:
#             tool_call = {"name": "no_tool", "parameters": {}}

#         # Execute tool
#         tool_name = tool_call.get("name")
#         tool_args = tool_call.get("parameters") or {}
        
#         if tool_name != "no_tool" and tool_name in AVAILABLE_TOOLS:
#             print(f"🔧 Executing: {tool_name}")
#             action_function = AVAILABLE_TOOLS[tool_name]
            
#             try:
#                 if inspect.iscoroutinefunction(action_function):
#                     action_result = await action_function(**tool_args)
#                 else:
#                     action_result = await run_in_threadpool(action_function, **tool_args)
                
#                 print(f"✅ Result: {action_result[:100]}...")
#                 action_document_for_memory += f" | Action: {tool_name}({tool_args}) | Result: {action_result}"
#             except Exception as tool_error:
#                 print(f"❌ Tool error: {tool_error}")
#                 action_result = f"Failed: {str(tool_error)}"
    
#     # 6. RESPONSE GENERATION
#     smart_token_limit = calculate_smart_token_limit(prompt, tool_name, action_result)
#     print(f"🧠 Token limit: {smart_token_limit}")
    
#     current_system_prompt = SYSTEM_PROMPT_TEMPLATE.format(humor_level=current_humor_level)
    
#     natural_response_prompt = f"""{current_system_prompt}

# Context:
# {conversation_history[-400:]}

# {"Tool used: " + tool_name if tool_name != "no_tool" else ""}
# {"Result: " + action_result[:150] if action_result else ""}

# User: {prompt}

# RULES:
# - Speak as MURPH directly to the user
# - First person only ("I")
# - No meta-commentary
# - Natural and conversational
# - For recommendations: give 2-3 specific suggestions with brief reasons
# - Be concise

# Response:"""

#     response_timeout = 90 if smart_token_limit > 500 else 60 if smart_token_limit > 300 else 45
    
#     payload = {
#         "model": MODEL_NAME, 
#         "prompt": natural_response_prompt, 
#         "stream": False, 
#         "options": {
#             "temperature": 0.6,  # Slightly increased for more natural recommendations
#             "num_predict": smart_token_limit,
#             "top_k": 30,
#             "top_p": 0.85,
#             "num_thread": 8,
#             "stop": ["User:", "AI:", "Note:", "Based on", "The user", "The AI"]
#         }
#     }
    
#     try:
#         timeout = aiohttp.ClientTimeout(total=response_timeout, connect=10, sock_read=response_timeout-10)
#         connector = aiohttp.TCPConnector(limit=10, limit_per_host=10, force_close=True)
        
#         async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
#             print(f"🔹 Generating response...")
#             async with session.post(OLLAMA_API_URL, json=payload) as response:
#                 if response.status != 200:
#                     if action_result:
#                         return clean_response_for_voice(action_result)
#                     return "Response generation error."
                
#                 res_json = await response.json()
#                 natural_response = res_json.get("response", "").strip()
                
#                 if not natural_response:
#                     if action_result:
#                         return clean_response_for_voice(action_result)
#                     return "Empty response generated."
                
#                 print(f"🧠 Raw: {natural_response[:100]}...")
                
#                 # Clean response
#                 cleaned_response = clean_response_for_voice(natural_response)
#                 safe_response = final_safety_check(cleaned_response)
                
#                 if not safe_response or len(safe_response) < 3:
#                     if action_result:
#                         safe_response = f"Done. {action_result[:50]}"
#                     else:
#                         safe_response = "Got it."
                
#                 print(f"✅ Final: {safe_response[:100]}...")
                
#                 # Update cache
#                 update_last_mentioned(safe_response, tool_name, tool_args)
                
#                 # TTS (force regenerate for personalized responses)
#                 contains_name = any(name in safe_response.lower() for name in ['proj', 'murph', 'nice to meet'])
#                 tts_task = asyncio.create_task(convert_text_to_audio_bytes(safe_response, force_regenerate=contains_name))
                
#                 # Save memory
#                 timestamp = datetime.now().isoformat()
#                 memory_parts = [f"User: {prompt}"]
                
#                 if tool_name != "no_tool":
#                     memory_parts.append(f"Action: {tool_name}({tool_args})")
#                     if action_result:
#                         memory_parts.append(f"Result: {action_result[:200]}")
                
#                 memory_parts.append(f"AI: {safe_response}")
#                 full_memory = " | ".join(memory_parts)
                
#                 asyncio.create_task(
#                     run_in_threadpool(
#                         memory_collection.add,
#                         documents=[full_memory],
#                         ids=[timestamp]
#                     )
#                 )
                
#                 # Wait for TTS
#                 try:
#                     await tts_task
#                 except Exception as tts_error:
#                     print(f"⚠️ TTS error: {tts_error}")
                
#                 return safe_response
                
#     except Exception as e:
#         print(f"❌ Generation error: {e}")
#         if action_result:
#             return clean_response_for_voice(action_result)
#         return "Error occurred."

# # --- ALSO ADD THIS UTILITY FUNCTION ---
# async def check_ollama_status():
#     """Quick check if Ollama is responsive"""
#     try:
#         timeout = aiohttp.ClientTimeout(total=3)
#         async with aiohttp.ClientSession(timeout=timeout) as session:
#             async with session.get("http://localhost:11434/api/tags") as response:
#                 return response.status == 200
#     except:
#         return False
# # --- FastAPI Endpoints ---

# @app.get("/chat-history")
# async def get_chat_history():
#     try:
#         memory_collection = ml_models["memory_collection"]
#         all_data = await run_in_threadpool(memory_collection.get, include=["documents"])
        
#         ids = all_data.get('ids', [])
#         documents = all_data.get('documents', [])
        
#         if not ids or not documents:
#             return []

#         paired_data = zip(ids, documents)
#         sorted_data = sorted(paired_data, key=lambda x: x[0])
        
#         history_log = []
#         for id, doc in sorted_data:
#             history_log.append({
#                 "id": id,
#                 "parts": parse_history_document(doc)
#             })
        
#         return history_log
        
#     except Exception as e:
#         print(f"❌ Error fetching chat history: {e}")
#         raise HTTPException(status_code=500, detail="Could not retrieve chat history.")
    
# @app.post("/voice-chat")
# async def voice_chat(audio: UploadFile = File(...)):
#     """OPTIMIZED: Parallel transcription + faster response"""
#     whisper_model = ml_models["whisper_model"]
#     temp_file_path = None
    
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
#             content = await audio.read()
            
#             if len(content) == 0:
#                 raise HTTPException(status_code=400, detail="Empty audio file")
            
#             temp_file.write(content)
#             temp_file_path = temp_file.name
        
#         print(f"📁 Audio: {len(content)} bytes")
        
#         if not os.path.exists(temp_file_path):
#             raise HTTPException(status_code=500, detail="Failed to save audio file")
        
#         file_size = os.path.getsize(temp_file_path)
#         if file_size == 0:
#             raise HTTPException(status_code=400, detail="Audio file is empty")
        
#         print(f"🎤 Fast transcription...")
        
#         try:
#             transcription_result = await run_in_threadpool(
#                 whisper_model.transcribe,
#                 temp_file_path,
#                 fp16=torch.cuda.is_available(),
#                 language="en",
#                 task="transcribe"
#             )
#             user_prompt = transcription_result["text"].strip()
#             print(f"🗣️ User: {user_prompt}")
#         except RuntimeError as e:
#             print(f"❌ Whisper failed: {e}")
#             if temp_file_path and os.path.exists(temp_file_path):
#                 os.remove(temp_file_path)
#             raise HTTPException(status_code=400, detail=f"Transcription failed: {str(e)}")
            
#     except HTTPException:
#         raise
#     except Exception as e:
#         print(f"❌ Error: {e}")
#         raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
#     finally:
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.remove(temp_file_path)
#             except:
#                 pass
    
#     if not user_prompt:
#         response_text = "Sorry, I didn't catch that."
#         audio_bytes = await convert_text_to_audio_bytes(response_text)
#     else:
#         # Get response (TTS runs in parallel)
#         response_text = await get_ai_response_and_update_memory(user_prompt)
        
#         # Check cache
#         cache_key = response_text[:100]
#         if cache_key in audio_cache:
#             audio_bytes = audio_cache[cache_key]
#         else:
#             audio_bytes = await convert_text_to_audio_bytes(response_text)
    
#     print(f"🤖 Response: {response_text[:60]}...")
    
#     if ml_models.get("piper_voice"):
#         return Response(content=audio_bytes, media_type="audio/wav")
#     else:
#         return Response(content=audio_bytes, media_type="audio/mpeg")

# @app.get("/")
# async def root():
#     piper_status = "✅ Active" if ml_models.get("piper_voice") else "⚠️ Not loaded (using gTTS)"
#     return {
#         "status": "MURPH AI Backend Running",
#         "voice_engine": piper_status,
#         "humor_level": current_humor_level
#     }