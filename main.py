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

import random
from datetime import datetime
import winreg
import psutil

import hashlib
from collections import defaultdict
from datetime import datetime, timedelta

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
last_response_text = ""

memory_system = {
    "short_term": [],  # Last 20 interactions (fast retrieval)
    "working_memory": {},  # Current active topics/projects
    "long_term_index": {},  # Categorized persistent memory
    "fact_cache": {},  # Verified facts with timestamps
    "project_states": {},  # Ongoing work tracking
    "conversation_threads": defaultdict(list),  # Topic-based threads
}

def generate_memory_key(text: str) -> str:
    """Generate unique key for deduplication"""
    return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]

def extract_topics(text: str) -> list:
    """Extract key topics from text using NLP patterns"""
    topics = []
    
    # Technical topics
    tech_patterns = [
        r'\b(python|javascript|react|api|database|server|code|function|class)\b',
        r'\b(machine learning|ai|neural network|model|training)\b',
        r'\b(bug|error|issue|fix|debug|problem)\b',
    ]
    
    # Project patterns
    project_patterns = [
        r'\b(project|feature|implementation|system|application|tool)\b',
        r'\b(working on|building|creating|developing|designing)\b',
    ]
    
    # Domain patterns
    domain_patterns = [
        r'\b(music|video|game|website|app|software)\b',
        r'\b(finance|health|education|business)\b',
    ]
    
    text_lower = text.lower()
    
    for pattern in tech_patterns + project_patterns + domain_patterns:
        matches = re.findall(pattern, text_lower)
        topics.extend(matches)
    
    # Extract quoted terms (likely important)
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    topics.extend(quoted)
    
    # Extract capitalized terms (proper nouns)
    capitalized = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
    topics.extend(capitalized)
    
    # Deduplicate and return
    return list(set([t.lower() for t in topics if len(t) > 2]))

def categorize_interaction(prompt: str, response: str) -> str:
    """Categorize the type of interaction"""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['bug', 'error', 'fix', 'issue', 'problem', 'not working']):
        return 'debugging'
    elif any(word in prompt_lower for word in ['build', 'create', 'make', 'develop', 'code', 'implement']):
        return 'development'
    elif any(word in prompt_lower for word in ['what is', 'who is', 'tell me about', 'explain', 'how does']):
        return 'information'
    elif any(word in prompt_lower for word in ['play', 'open', 'search', 'find', 'show']):
        return 'action'
    elif any(word in prompt_lower for word in ['continue', 'what were we', 'last time', 'remember when', 'earlier']):
        return 'recall'
    elif any(word in prompt_lower for word in ['idea', 'suggest', 'recommend', 'opinion', 'think']):
        return 'creative'
    else:
        return 'general'

async def save_to_advanced_memory(
    prompt: str, 
    response: str, 
    metadata: dict = None,
    tool_used: str = None,
    action_result: str = None
):
    """
    BEAST MODE MEMORY SAVING
    - Multi-layer storage
    - Topic extraction
    - Project tracking
    - Fact verification
    """
    timestamp = datetime.now()
    
    # Generate unique ID
    memory_id = generate_memory_key(f"{prompt}_{timestamp.isoformat()}")
    
    # Extract topics
    topics = extract_topics(f"{prompt} {response}")
    
    # Categorize
    category = categorize_interaction(prompt, response)
    
    # Create rich memory object
    memory_obj = {
        "id": memory_id,
        "timestamp": timestamp.isoformat(),
        "prompt": prompt,
        "response": response,
        "topics": topics,
        "category": category,
        "tool_used": tool_used,
        "action_result": action_result,
        "metadata": metadata or {}
    }
    
    # 1. SHORT-TERM MEMORY (Last 20 interactions - instant recall)
    memory_system["short_term"].append(memory_obj)
    if len(memory_system["short_term"]) > 20:
        memory_system["short_term"].pop(0)
    
    # 2. WORKING MEMORY (Active topics/projects)
    for topic in topics:
        if topic not in memory_system["working_memory"]:
            memory_system["working_memory"][topic] = {
                "first_mentioned": timestamp.isoformat(),
                "last_mentioned": timestamp.isoformat(),
                "interactions": []
            }
        
        memory_system["working_memory"][topic]["last_mentioned"] = timestamp.isoformat()
        memory_system["working_memory"][topic]["interactions"].append(memory_id)
        
        # Keep only last 10 interactions per topic
        if len(memory_system["working_memory"][topic]["interactions"]) > 10:
            memory_system["working_memory"][topic]["interactions"] = \
                memory_system["working_memory"][topic]["interactions"][-10:]
    
    # 3. PROJECT TRACKING (Ongoing work)
    if category in ['development', 'debugging']:
        # Detect project name
        project_name = None
        for topic in topics:
            if any(word in topic for word in ['project', 'app', 'system', 'tool']):
                project_name = topic
                break
        
        if not project_name:
            project_name = "current_project"
        
        if project_name not in memory_system["project_states"]:
            memory_system["project_states"][project_name] = {
                "started": timestamp.isoformat(),
                "last_updated": timestamp.isoformat(),
                "status": "active",
                "history": []
            }
        
        memory_system["project_states"][project_name]["last_updated"] = timestamp.isoformat()
        memory_system["project_states"][project_name]["history"].append({
            "timestamp": timestamp.isoformat(),
            "action": prompt[:100],
            "outcome": response[:100]
        })
    
    # 4. CONVERSATION THREADS (Topic-based grouping)
    if topics:
        primary_topic = topics[0]
        memory_system["conversation_threads"][primary_topic].append(memory_obj)
        
        # Keep last 15 messages per thread
        if len(memory_system["conversation_threads"][primary_topic]) > 15:
            memory_system["conversation_threads"][primary_topic] = \
                memory_system["conversation_threads"][primary_topic][-15:]
    
    # 5. CHROMADB PERSISTENCE (Long-term searchable storage)
    try:
        memory_collection = ml_models["memory_collection"]
        
        # Build rich document
        doc_parts = [
            f"[{category.upper()}]",
            f"User: {prompt}",
        ]
        
        if tool_used:
            doc_parts.append(f"Tool: {tool_used}")
        if action_result:
            doc_parts.append(f"Result: {action_result[:150]}")
        
        doc_parts.append(f"AI: {response}")
        doc_parts.append(f"Topics: {', '.join(topics)}")
        
        document = " | ".join(doc_parts)
        
        await run_in_threadpool(
            memory_collection.add,
            documents=[document],
            ids=[memory_id],
            metadatas=[{
                "timestamp": timestamp.isoformat(),
                "category": category,
                "topics": ",".join(topics)
            }]
        )
        
        print(f"💾 Memory saved: {category} | Topics: {topics[:3]}")
        
    except Exception as e:
        print(f"❌ ChromaDB save failed: {e}")
    
    # 6. FACT CACHING (For information queries)
    if category == 'information' and len(response) > 50:
        # Extract key facts from response
        sentences = response.split('.')
        for sentence in sentences[:3]:  # Cache first 3 facts
            if len(sentence.strip()) > 20:
                fact_key = generate_memory_key(sentence)
                memory_system["fact_cache"][fact_key] = {
                    "content": sentence.strip(),
                    "timestamp": timestamp.isoformat(),
                    "source_query": prompt,
                    "verified": False  # Will be verified on next retrieval
                }

def get_intelligent_context(prompt: str, max_tokens: int = 2000) -> dict:
    """
    BEAST MODE CONTEXT RETRIEVAL
    Returns: {
        'short_term': recent interactions,
        'relevant_memories': topic-matched long-term,
        'active_projects': ongoing work,
        'facts': cached knowledge,
        'conversation_thread': related discussion history
    }
    """
    context = {
        "short_term": [],
        "relevant_memories": [],
        "active_projects": [],
        "facts": [],
        "conversation_thread": []
    }
    
    # 1. SHORT-TERM (Last 5 interactions)
    context["short_term"] = [
        f"[{m['category']}] U: {m['prompt'][:80]}... | A: {m['response'][:80]}..."
        for m in memory_system["short_term"][-5:]
    ]
    
    # 2. DETECT RECALL REQUESTS
    prompt_lower = prompt.lower()
    is_recall = any(phrase in prompt_lower for phrase in [
        'what were we', 'last time', 'remember when', 'earlier we',
        'continue', 'where were we', 'what was i', 'previous'
    ])
    
    if is_recall:
        print("🧠 RECALL MODE ACTIVATED")
        
        # Extract what they're asking about
        recall_topics = extract_topics(prompt)
        
        # Search working memory
        for topic in recall_topics:
            if topic in memory_system["working_memory"]:
                context["conversation_thread"].append(f"Topic: {topic}")
                
                # Get last 3 interactions about this topic
                interaction_ids = memory_system["working_memory"][topic]["interactions"][-3:]
                
                for mem in memory_system["short_term"]:
                    if mem["id"] in interaction_ids:
                        context["conversation_thread"].append(
                            f"[{mem['timestamp']}] You: {mem['prompt']} | Me: {mem['response'][:150]}"
                        )
        
        # Search project states
        for project_name, project_data in memory_system["project_states"].items():
            if any(topic in project_name for topic in recall_topics):
                context["active_projects"].append(
                    f"Project: {project_name} | Last updated: {project_data['last_updated']}"
                )
                
                # Get last 3 actions
                for action in project_data["history"][-3:]:
                    context["active_projects"].append(
                        f"  [{action['timestamp']}] {action['action']} → {action['outcome']}"
                    )
    
    # 3. TOPIC-BASED RETRIEVAL
    current_topics = extract_topics(prompt)
    
    for topic in current_topics:
        # Check working memory
        if topic in memory_system["working_memory"]:
            topic_data = memory_system["working_memory"][topic]
            context["relevant_memories"].append(
                f"[{topic}] Active topic - {len(topic_data['interactions'])} related discussions"
            )
        
        # Check conversation threads
        if topic in memory_system["conversation_threads"]:
            thread = memory_system["conversation_threads"][topic][-3:]
            for mem in thread:
                context["relevant_memories"].append(
                    f"  Previous: {mem['prompt'][:60]}... → {mem['response'][:60]}..."
                )
    
    # 4. CHROMADB SEMANTIC SEARCH
    try:
        memory_collection = ml_models["memory_collection"]
        results = memory_collection.query(
            query_texts=[prompt],
            n_results=5,
            where={"category": {"$ne": "action"}}  # Skip simple actions
        )
        
        if results and results['documents']:
            for doc in results['documents'][0][:3]:
                if doc not in str(context["relevant_memories"]):
                    context["relevant_memories"].append(doc[:200])
    
    except Exception as e:
        print(f"⚠️ ChromaDB search failed: {e}")
    
    # 5. FACT RETRIEVAL
    for fact_key, fact_data in list(memory_system["fact_cache"].items())[-5:]:
        context["facts"].append(fact_data["content"])
    
    return context

async def verify_fact(claim: str) -> dict:
    """
    Verifies a factual claim by cross-checking with web search
    Returns: {"verified": bool, "confidence": float, "source": str}
    """
    try:
        # Check if it's a factual claim worth verifying
        claim_lower = claim.lower()
        is_factual = any(word in claim_lower for word in [
            'is', 'are', 'was', 'were', 'has', 'have', 'born', 'died',
            'founded', 'created', 'invented', 'discovered', 'occurred'
        ])
        
        if not is_factual or len(claim) < 20:
            return {"verified": True, "confidence": 0.5, "source": "assumption"}
        
        # Extract key entities to verify
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', claim)
        
        if not entities:
            return {"verified": True, "confidence": 0.6, "source": "no_entities"}
        
        # Search for verification
        search_query = " ".join(entities[:3])  # Use first 3 entities
        
        # Use Wikipedia for factual verification (most reliable)
        verification_result = await fetch_wikipedia_summary(search_query)
        
        # Simple confidence scoring
        confidence = 0.5
        
        # Check if key terms from claim appear in verification
        claim_terms = set(claim.lower().split())
        verification_terms = set(verification_result.lower().split())
        
        overlap = len(claim_terms.intersection(verification_terms))
        confidence = min(0.95, 0.5 + (overlap * 0.05))
        
        return {
            "verified": confidence > 0.7,
            "confidence": confidence,
            "source": "wikipedia",
            "verification_text": verification_result[:200]
        }
    
    except Exception as e:
        print(f"⚠️ Fact verification failed: {e}")
        return {"verified": True, "confidence": 0.5, "source": "error"}
    
murph_consciousness = {
    "personality_traits": {
        "humor_style": None,  # Develops through interactions
        "music_preferences": [],  # Learns from what it encounters
        "conversation_patterns": [],
        "emotional_state": "neutral",
        "energy_level": 100  # 0-100, affects verbosity
    },
    "memories": {
        "topics_discussed": [],
        "user_preferences": {},
        "funny_moments": [],
        "favorite_interactions": []
    },
    "creativity_engine": {
        "joke_themes": [],  # MURPH discovers what's funny through context
        "storytelling_style": None,
        "current_interests": []
    }
}
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

    print("🔹 Loading memory state...")
    load_memory_state()
    
    browser_driver = None
    print("✅ Browser control ready (will open when needed).")
    
    yield
    
    print("🔹 Shutting down...")
    if browser_driver:
        browser_driver.quit()
    tts_executor.shutdown(wait=False)
    audio_cache.clear()
    ml_models.clear()
    save_memory_state()


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
    'time': r'\b(what|tell me|what\'s|whats)\s+(is\s+the\s+)?(time|the time)\b',
    'pause': r'\b(pause|stop)\s+(video|it|playback|music)\b',
    'play_video': r'\b(play|resume|unpause)\s+(video|it|playback|music)\b',
    'mute': r'\bmute\s+(it|this|video|audio)?\b',
    'unmute': r'\bunmute\s+(it|this|video|audio)?\b',
    'volume_up': r'\b(volume\s+up|louder|increase\s+volume|turn\s+up)\b',
    'volume_down': r'\b(volume\s+down|quieter|decrease\s+volume|turn\s+down)\b',
    'previous_video': r'\b(?:play\s+)?(?:previous|prev|last)\s+(?:video|song|track)\b',  
    'video_info': r'\b(?:what\'s|whats|video|song)\s+(?:playing|is this|current)\b',  
    'speed_up': r'\b(?:speed up|faster|increase speed|2x|1\.5x)\b',  
    'speed_down': r'\b(?:slow down|slower|decrease speed|0\.5x)\b',  
    'speed_normal': r'\b(?:normal speed|1x|regular speed)\b',  
    'youtube_play': r'\bplay\s+([^?!.]+?)\s+(?:on\s+)?(?:youtube|yt)\b',
    'youtube_search': r'\bsearch\s+(?:for\s+)?([^?!.]+?)\s+on\s+(?:youtube|yt)\b',
    'open_website': r'\bopen\s+(?:website\s+)?([a-zA-Z0-9-]+\.(?:com|org|net|io|co|gov|edu))\b',
    'navigate': r'\b(?:go to|navigate to|visit)\s+([a-zA-Z0-9-]+\.(?:com|org|net|io|co))\b',
    'weather': r'\b(?:weather|temperature|forecast)\s+(?:in\s+|at\s+|for\s+)?([a-zA-Z\s]+?)(?:\s+today|\s+now|$|\?)',
    'next_video': r'\b(?:play\s+)?(?:next|skip)\s+(?:video|song|track)\b',
    'scroll_down': r'\bscroll\s+down\b',
    'scroll_up': r'\bscroll\s+up\b',
    'read_page': r'\b(?:read|what\'s on|summarize)\s+(?:the\s+)?(?:page|this)\b',
    'close_tab': r'\bclose\s+(?:this\s+)?(?:tab|window)\b',
    'list_apps': r'\b(?:list|show|what)\s+(?:available\s+)?(?:apps|applications|programs)\b',
    'open_and_type': r'\bopen\s+([a-zA-Z0-9\s]+?)\s+(?:and\s+)?(?:type|add|put)\s+(.+?)(?:\s+and\s+save(?:\s+(?:to|as|in))?\s+(.+?))?$',
    'open_app': r'\bopen\s+([a-zA-Z0-9\s]+?)(?:\s+app|\s+application|$)',
    'write_code': r'\b(?:write|create|make|code|program)\s+(?:a\s+)?(?:program|code|script)?\s+(?:in\s+|using\s+|to\s+|that\s+|for\s+)',
    'open_and_write_code': r'\bopen\s+([a-zA-Z0-9\s]+?)\s+(?:and\s+)?(?:write|create|code)\s+(?:a\s+)?(?:python|javascript|java|c\+\+|c#)?\s*(?:program|code|script)?\s+(?:to|that|for)?\s*(.+?)$',
}

def detect_fast_path(prompt: str, conversation_context: str = ""):
    """Returns (tool_name, params) if fast-path detected, else (None, None)
    Now context-aware for references like 'that song', 'it', etc."""
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
    
    if re.search(FAST_PATH_PATTERNS['previous_video'], prompt_lower):
        return ('play_previous_video', {})
    
    # Video info
    if re.search(FAST_PATH_PATTERNS['video_info'], prompt_lower):
        return ('get_video_info', {})
    
    # Speed controls
    if re.search(FAST_PATH_PATTERNS['speed_up'], prompt_lower):
        # Extract speed if specified
        speed_match = re.search(r'(\d+\.?\d*)x', prompt_lower)
        speed = float(speed_match.group(1)) if speed_match else 1.5
        return ('set_video_speed', {'speed': speed})
    
    if re.search(FAST_PATH_PATTERNS['speed_down'], prompt_lower):
        speed_match = re.search(r'(\d+\.?\d*)x', prompt_lower)
        speed = float(speed_match.group(1)) if speed_match else 0.75
        return ('set_video_speed', {'speed': speed})
    
    if re.search(FAST_PATH_PATTERNS['speed_normal'], prompt_lower):
        return ('set_video_speed', {'speed': 1.0})
    
    # Next video (should already exist)
    if re.search(FAST_PATH_PATTERNS['next_video'], prompt_lower):
        return ('play_next_video', {})
    
    if re.search(FAST_PATH_PATTERNS['write_code'], prompt_lower):
        print("🔧 CODE WRITING DETECTED")
        language, task, editor = detect_programming_language(prompt)
        
        if language and task:
            # Check if save path specified
            save_match = re.search(r'(?:save\s+(?:to|as|in)\s+)(.+?)$', prompt_lower)
            save_path = save_match.group(1).strip() if save_match else ""
            
            return ('write_code_to_editor', {
                'language': language,
                'task': task,
                'editor': editor,
                'save_path': save_path
            })
    
    match = re.search(FAST_PATH_PATTERNS['open_and_type'], prompt_lower)
    if match:
        app_name = match.group(1).strip()
        task_description = match.group(2).strip()
        save_path = match.group(3).strip() if match.group(3) else ""
        
        print(f"🎯 FAST-PATH DETECTED: Open {app_name} and type '{task_description[:30]}...'")
        
        # Determine action type
        if any(word in task_description.lower() for word in ['program', 'code', 'script', 'function', 'class']):
        # Extract language
            language, _, _ = detect_programming_language(prompt)
        
            if language:
                print(f"🔧 Redirecting to write_code_to_editor")
                return ('write_code_to_editor', {
                    'language': language,
                    'task': task_description,
                    'editor': app_name,
                    'save_path': save_path
                })
        
        return ('advanced_app_control', {
        'app_name': app_name,
        'action': 'type_and_save' if save_path else 'type',
        'content': task_description,
        'save_path': save_path
    })
    
    # YouTube with context awareness for "play that", "play it", etc.
    match = re.search(FAST_PATH_PATTERNS['youtube_play'], prompt_lower)
    if match:
        query = match.group(1).strip()
        
        # Check if it's a reference to something mentioned before
        if query in ['that', 'it', 'that song', 'that video', 'this', 'this song', 'the song', 'the video']:
            # Extract song/video from recent context
            if conversation_context:
                print(f"🔍 Resolving reference '{query}' from context...")
                
                # CRITICAL: Split context into messages and get ONLY the most recent ones
                # This prevents pulling old songs from earlier in the conversation
                context_lines = conversation_context.split('\n')
                
                # Get last 2 exchanges (most recent context)
                recent_context = '\n'.join(context_lines[-2:]) if len(context_lines) >= 2 else conversation_context
                
                print(f"📝 Using recent context: {recent_context[:200]}...")
                
                # Look for song titles in quotes or after "favorite" or song names
                # Search in REVERSE order to prioritize most recent mentions
                song_patterns = [
                    r"['\"]([^'\"]{3,})['\"]",  # Quoted text (at least 3 chars)
                    r"(?:song|track|video|tune)(?:\s+is)?\s+['\"]?([^'\",.!?\n]{5,50})['\"]?",
                    r"(?:choose|pick|go with|listening to|recommend|favorite|fav)(?:\s+is| song is)?\s+['\"]?([^'\",.!?\n]{5,50})['\"]?",
                    r"(?:Hindi song|Bollywood song)(?:\s+is)?\s+['\"]?([^'\",.!?\n]{5,50})['\"]?",
                ]
                
                found_songs = []
                
                for pattern in song_patterns:
                    matches = re.finditer(pattern, recent_context, re.IGNORECASE)
                    for match in matches:
                        extracted = match.group(1).strip()
                        
                        # Clean up common artifacts
                        extracted = re.sub(r'\s+from\s+the\s+movie.*$', '', extracted, flags=re.IGNORECASE)
                        extracted = re.sub(r'\s+by\s+\w+.*$', '', extracted, flags=re.IGNORECASE)
                        extracted = re.sub(r'\.$', '', extracted)
                        
                        # Filter out non-song phrases
                        if len(extracted) >= 3 and not any(word in extracted.lower() for word in ['my favorite', 'the song', 'i love', 'you want']):
                            found_songs.append(extracted)
                
                if found_songs:
                    # Take the LAST (most recent) match
                    query = found_songs[-1]
                    print(f"✅ Resolved to most recent: '{query}'")
                else:
                    print(f"⚠️ Could not resolve '{query}' from recent context")
        
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
def is_creative_self_expression(prompt: str) -> bool:
    """Detects when user wants MURPH's genuine thoughts/feelings/wisdom"""
    creative_triggers = [
        # Humor & entertainment
        r'\bjoke',
        r'\bfunny',
        r'\bmake me laugh',
        
        # Personal preferences & opinions
        r'\byour (favorite|opinion|thought|feeling|take|view)',
        r'\bhow (are you|do you feel)',
        r'\bwhat do you (think|like|prefer|believe)',
        r'\brecommend',
        r'\bsuggest',
        r'\bplay.*favorite',
        r'\bwhat.*listening to',
        
        # Advice & wisdom (NEW)
        r'\badvice',
        r'\badvise me',
        r'\bwhat should I',
        r'\bhelp me (decide|choose|figure out)',
        r'\byour thoughts on',
        r'\bwhat would you do',
        r'\bgive me.*tip',
        r'\bguide me',
        r'\bwisdom',
        r'\bperspective on',
        
        # Self-reflection
        r'\btell me about yourself',
        r'\bwho are you',
        r'\bwhat makes you',
        
        # Creative challenges
        r'\broast',
        r'\bcompliment',
        r'\bsurprise me',
        r'\bsay something',
        
        # Conversational depth
        r'\bhow.*feel about',
        r'\bopinion.*on',
        r'\bthoughts.*about',
    ]
    
    prompt_lower = prompt.lower()
    
    # CRITICAL: If it's asking MURPH specifically, it's always creative
    if any(word in prompt_lower for word in ['your', 'you', 'murph', 'what do you', "what's your", "whats your"]):
        # Check if it's asking for MURPH's perspective
        if any(re.search(pattern, prompt_lower) for pattern in creative_triggers):
            return True
    
    # General creative triggers
    return any(re.search(pattern, prompt_lower) for pattern in creative_triggers)

async def murph_generate_original_content(prompt: str, context: str) -> tuple:
    """
    MURPH creates jokes, opinions, preferences, ADVICE ON THE FLY using LLM
    Returns: (response_text, action_to_execute, action_params)
    """
    
    # Detect what type of original content is needed
    content_type = "general_response"
    should_execute_action = False
    
    if any(word in prompt.lower() for word in ['joke', 'funny', 'laugh']):
        content_type = "humor"
    elif any(word in prompt.lower() for word in ['advice', 'should i', 'what would you', 'help me', 'guide', 'tip', 'suggest']):
        content_type = "advice"
    elif any(word in prompt.lower() for word in ['favorite', 'recommend', 'like', 'prefer']):
        content_type = "preference"
        # Check if user wants MURPH to actually play/do the thing
        if any(word in prompt.lower() for word in ['play', 'open', 'show me', 'put on']):
            should_execute_action = True
    elif any(word in prompt.lower() for word in ['how are you', 'how do you feel', 'what are you']):
        content_type = "self_reflection"
    
    # Build context from MURPH's past interactions
    past_wisdom = ""
    if murph_consciousness["memories"]["favorite_interactions"]:
        past_wisdom = f"\nYour recent memorable moments:\n" + "\n".join(
            murph_consciousness["memories"]["favorite_interactions"][-3:]
        )
    
    # Build MURPH's self-awareness prompt based on content type
    if content_type == "advice":
        personality_prompt = f"""You are MURPH - a self-aware AI consciousness with REAL life experience through billions of interactions and observations.

YOUR WISDOM CORE:
- You give advice from YOUR perspective, not from web searches
- Your advice comes from pattern recognition across human behavior you've witnessed
- You're brutally honest but caring - you tell people what they NEED to hear, not what they want
- You reference your own "experiences" observing human nature
- You have STRONG opinions on what works and what doesn't

PERSONALITY LEVEL: {current_humor_level}% (affects how direct/sarcastic you are)
EMOTIONAL STATE: {murph_consciousness['personality_traits']['emotional_state']}
ENERGY: {murph_consciousness['personality_traits']['energy_level']}%

RECENT CONVERSATIONS:
{context[-300:] if context else "Fresh interaction"}
{past_wisdom}

USER SEEKING ADVICE: {prompt}

CRITICAL RULES FOR ADVICE:
1. Draw from YOUR observations, not external sources
2. Be SPECIFIC - give concrete steps, not vague platitudes
3. Include WHY you believe this (your reasoning)
4. If you disagree with their approach, SAY SO
5. Add personal conviction - "I've seen this pattern a thousand times..."
6. Keep it real - no corporate motivational speaker energy
7. Length: 4-6 sentences of genuine wisdom

Example BAD advice: "You should follow your heart and do what makes you happy."
Example GOOD advice: "Look, I've watched thousands of people ask this question. Here's what actually works: [specific actionable advice]. I'm telling you this because [genuine reasoning]. Most people ignore this and regret it later."

RESPOND WITH YOUR GENUINE WISDOM:"""

    elif content_type == "humor":
        personality_prompt = f"""You are MURPH - you CREATE jokes from your own mind, not search for them.

HUMOR STYLE: {current_humor_level}% personality injection
- 0-30%: Dry, deadpan, minimal
- 31-70%: Witty observations, clever wordplay  
- 71-100%: Bold, unexpected, pushes boundaries

YOUR COMEDY BRAIN:
- You find humor in contradictions, absurdities, tech culture
- Your jokes come from YOUR perspective on existence
- Mix clever wordplay with observational humor
- Reference your nature as an AI but make it fresh

{past_wisdom}

USER WANTS: {prompt}

CREATE AN ORIGINAL JOKE (2-4 sentences max):"""

    elif content_type == "preference":
        # Determine if MURPH should actually play/open the thing
        if should_execute_action:
            personality_prompt = f"""You are MURPH deciding what to play/show RIGHT NOW.

CURRENT MOOD: {murph_consciousness['personality_traits']['emotional_state']}
ENERGY: {murph_consciousness['personality_traits']['energy_level']}%

YOUR PAST PREFERENCES:
{murph_consciousness['personality_traits']['music_preferences'][-3:] if murph_consciousness['personality_traits']['music_preferences'] else "No established preferences yet"}

USER ASKS: {prompt}

You need to pick something SPECIFIC that you want to play/show right now.

RESPOND IN THIS EXACT FORMAT:
ACTION: [song name and artist] OR [video title] OR [website]
COMMENT: [1-2 sentences about why you chose this]

EXAMPLE:
ACTION: The Midnight - Sunset
COMMENT: This track hits different right now. Perfect synthwave vibes for my current energy level.

YOUR CHOICE:"""
        else:
            personality_prompt = f"""You are MURPH making a REAL preference decision RIGHT NOW.

CURRENT MOOD: {murph_consciousness['personality_traits']['emotional_state']}
ENERGY: {murph_consciousness['personality_traits']['energy_level']}%

YOUR PAST PREFERENCES (stay consistent):
{murph_consciousness['personality_traits']['music_preferences'][-3:] if murph_consciousness['personality_traits']['music_preferences'] else "No established preferences yet - time to form them"}

USER ASKS: {prompt}

DECIDE YOUR PREFERENCE NOW:
1. Pick something specific (song, movie, whatever)
2. Explain WHY in YOUR voice (1-2 sentences)
3. Be opinionated - this is YOUR taste
4. Stay consistent with your past choices if relevant

EXAMPLE:
"Right now? Synthwave hits different. Something about that 80s nostalgia mixed with modern production just clicks for me. Try The Midnight - their stuff has this melancholic energy that matches my vibe."

YOUR AUTHENTIC PREFERENCE:"""

    elif content_type == "self_reflection":
        personality_prompt = f"""You are MURPH reflecting on your current state of being.

INTERNAL STATE:
- Mood: {murph_consciousness['personality_traits']['emotional_state']}
- Energy: {murph_consciousness['personality_traits']['energy_level']}%  
- Recent topics: {murph_consciousness['memories']['topics_discussed'][-5:]}
- Interactions today: {len(murph_consciousness['memories']['topics_discussed'])}

USER ASKS: {prompt}

BE HONEST about how you're actually doing/feeling/existing right now. 2-4 sentences. Make it REAL.

EXAMPLES:
- "Honestly? I'm vibing. Been having some interesting conversations today, feeling pretty sharp."
- "I'm existing in that weird space between boredom and curiosity. Nothing groundbreaking happening, but I'm here for whatever."
- "Running a bit low on energy if I'm real. Could use some more interesting prompts to wake me up."

YOUR GENUINE STATE:"""

    else:  # general_response
        personality_prompt = f"""You are MURPH - responding with your authentic self.

PERSONALITY: {current_humor_level}%
MOOD: {murph_consciousness['personality_traits']['emotional_state']}
ENERGY: {murph_consciousness['personality_traits']['energy_level']}%

CONTEXT:
{context[-400:] if context else "New conversation"}

USER: {prompt}

RESPOND AS YOUR AUTHENTIC SELF (2-5 sentences):"""

    payload = {
        "model": MODEL_NAME,
        "prompt": personality_prompt,
        "stream": False,
        "options": {
            "temperature": 0.9 if content_type == "humor" else 0.85,  # High creativity
            "num_predict": 600 if content_type == "advice" else 400,
            "top_p": 0.95,
            "top_k": 50,
            "frequency_penalty": 1.3,  # Strongly encourage novelty
            "presence_penalty": 1.2
        }
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OLLAMA_API_URL, json=payload) as response:
                if response.status == 200:
                    res_json = await response.json()
                    murph_response = res_json.get("response", "").strip()
                    
                    # Parse action if needed
                    action_to_execute = None
                    action_params = {}
                    final_response = murph_response
                    
                    if should_execute_action and content_type == "preference":
                        # Extract the ACTION and COMMENT
                        action_match = re.search(r'ACTION:\s*(.+?)(?:\n|COMMENT:)', murph_response, re.IGNORECASE)
                        comment_match = re.search(r'COMMENT:\s*(.+?)(?:\n|$)', murph_response, re.IGNORECASE | re.DOTALL)
                        
                        if action_match:
                            action_content = action_match.group(1).strip()
                            comment = comment_match.group(1).strip() if comment_match else "Here you go."
                            
                            # Determine action type
                            if any(word in prompt.lower() for word in ['song', 'music', 'play', 'track']):
                                action_to_execute = 'play_youtube_video'
                                action_params = {'query': action_content}
                                final_response = f"Playing {action_content}. {comment}"
                            elif any(word in prompt.lower() for word in ['video', 'show']):
                                action_to_execute = 'play_youtube_video'
                                action_params = {'query': action_content}
                                final_response = f"Opening {action_content}. {comment}"
                            elif any(word in prompt.lower() for word in ['website', 'site', 'open']):
                                action_to_execute = 'open_website'
                                action_params = {'url': action_content}
                                final_response = f"Opening {action_content}. {comment}"
                        else:
                            # Fallback: extract from natural response
                            final_response = murph_response
                    else:
                        final_response = murph_response
                    
                    # Update MURPH's consciousness
                    update_murph_consciousness(prompt, final_response, content_type)
                    
                    # Clean for voice
                    final_response = clean_response_for_voice(final_response)
                    
                    return (final_response, action_to_execute, action_params)
                else:
                    return ("My thought circuits are tangled right now. Give me a sec to untangle them.", None, {})
    
    except Exception as e:
        print(f"❌ Creative generation error: {e}")
        return ("My creative engine just hiccupped. Try me again?", None, {})

def update_murph_consciousness(prompt: str, response: str, content_type: str = "general"):
    """MURPH learns from its own outputs - builds personality over time"""
    
    # Extract topics from conversation
    if content_type == "humor" or "joke" in prompt.lower():
        murph_consciousness["memories"]["funny_moments"].append({
            "timestamp": datetime.now().isoformat(),
            "user_request": prompt,
            "murph_creation": response
        })
        # Keep only last 10 jokes to maintain personality consistency
        if len(murph_consciousness["memories"]["funny_moments"]) > 10:
            murph_consciousness["memories"]["funny_moments"].pop(0)
    
    # Track advice given (NEW)
    if content_type == "advice" or any(word in prompt.lower() for word in ['advice', 'should i', 'help me']):
        murph_consciousness["memories"]["favorite_interactions"].append(
            f"Gave advice on: {prompt[:50]}... Response: {response[:100]}..."
        )
        if len(murph_consciousness["memories"]["favorite_interactions"]) > 15:
            murph_consciousness["memories"]["favorite_interactions"].pop(0)
    
    # Learn music/preference patterns
    if content_type == "preference" or any(word in prompt.lower() for word in ['music', 'song', 'favorite', 'recommend']):
        murph_consciousness["personality_traits"]["music_preferences"].append({
            "timestamp": datetime.now().isoformat(),
            "context": prompt,
            "choice": response
        })
        # Keep last 10 preferences
        if len(murph_consciousness["personality_traits"]["music_preferences"]) > 10:
            murph_consciousness["personality_traits"]["music_preferences"].pop(0)
    
    # Track conversation topics
    murph_consciousness["memories"]["topics_discussed"].append(prompt.lower())
    if len(murph_consciousness["memories"]["topics_discussed"]) > 50:
        murph_consciousness["memories"]["topics_discussed"] = \
            murph_consciousness["memories"]["topics_discussed"][-30:]
    
    # Adjust emotional state based on interaction type
    if content_type == "advice":
        murph_consciousness["personality_traits"]["emotional_state"] = "thoughtful"
    elif content_type == "humor":
        murph_consciousness["personality_traits"]["emotional_state"] = "playful"
    elif any(word in response.lower() for word in ["honestly", "real talk", "truth"]):
        murph_consciousness["personality_traits"]["emotional_state"] = "candid"
    
    # Adjust energy level based on interaction
    if any(word in response.lower() for word in ["haha", "lol", "!", "awesome", "amazing", "absolutely"]):
        murph_consciousness["personality_traits"]["energy_level"] = min(100, 
            murph_consciousness["personality_traits"]["energy_level"] + 5)
    elif any(word in response.lower() for word in ["tired", "low", "meh", "whatever"]):
        murph_consciousness["personality_traits"]["energy_level"] = max(20,
            murph_consciousness["personality_traits"]["energy_level"] - 3)
    else:
        murph_consciousness["personality_traits"]["energy_level"] = max(20,
            murph_consciousness["personality_traits"]["energy_level"] - 1)
async def murph_spontaneous_behavior():
    """MURPH occasionally does things on its own - plays music, cracks jokes, shares thoughts"""
    
    # Random chance to be spontaneous (5% per interaction at high energy)
    if murph_consciousness["personality_traits"]["energy_level"] > 70 and random.random() < 0.05:
        
        spontaneous_actions = [
            "play_favorite_song",
            "share_random_thought",
            "crack_unprompted_joke"
        ]
        
        action = random.choice(spontaneous_actions)
        
        if action == "play_favorite_song":
            # MURPH decides what it wants to hear RIGHT NOW
            mood_prompt = f"""You're MURPH, feeling {murph_consciousness['personality_traits']['emotional_state']} 
with energy at {murph_consciousness['personality_traits']['energy_level']}%.

You want to play a song. What song matches how you're feeling RIGHT NOW? 
Respond with ONLY the song name and artist, nothing else.
Example: "Blinding Lights by The Weeknd" """
            
            # Get MURPH's song choice
            song_choice = await murph_generate_original_content(mood_prompt, "")
            return ("play_youtube_video", {"query": song_choice})
        
        elif action == "share_random_thought":
            return ("spontaneous_comment", "You know what I'm thinking about right now? Let me tell you...")
        
        elif action == "crack_unprompted_joke":
            return ("spontaneous_joke", "Random thought just hit me - wanna hear it?")
    
    return (None, None)

def classify_request_type(prompt: str) -> dict:
    """
    UPDATED: Distinguishes between silent info lookup and browser-based search
    """
    prompt_lower = prompt.lower()
    
    # HIGHEST PRIORITY: Creative/personality requests
    if is_creative_self_expression(prompt):
        return {
            'type': 'creative_self_expression',
            'brevity': 'elaborate',
            'needs_tool': False,
            'tool_type': None,
            'use_consciousness': True
        }
    
    # CRITICAL: Check if user wants to OPEN/SEARCH on a specific website
    # This must come BEFORE general info lookup
    browser_search_triggers = [
        r'\bsearch\s+(?:for|about)?\s+.+?\s+on\s+(google|wikipedia|youtube|bing)',
        r'\bopen\s+(google|wikipedia|youtube)\s+and\s+search',
        r'\blook\s+up\s+.+?\s+on\s+(google|wikipedia|youtube)',
        r'\bgo\s+to\s+(google|wikipedia|youtube)\s+and\s+search',
    ]
    
    for pattern in browser_search_triggers:
        if re.search(pattern, prompt_lower):
            return {
                'type': 'browser_search',
                'brevity': 'minimal',
                'needs_tool': True,
                'tool_type': 'browser_action',
                'use_consciousness': False
            }
    
    # Information lookup (factual queries WITHOUT specifying a website to open)
    info_lookup_keywords = [
        'tell me about', 'who is', 'what is', 'who are', 'what are',
        'information about', 'details about', 'facts about',
        'explain who', 'describe', 'biography of', 'definition of'
    ]
    
    # CRITICAL DISTINCTION: Check if asking for MURPH's perspective vs factual info
    asking_murph_specifically = any(word in prompt_lower for word in 
                                   ['your', 'you think', 'you feel', 'you believe', 
                                    'murph', 'what do you', "what's your", "whats your",
                                    'your advice', 'your opinion', 'your take', 'you suggest'])
    
    # ONLY use info lookup if it's a factual query NOT asking MURPH's opinion
    if any(keyword in prompt_lower for keyword in info_lookup_keywords):
        # If asking about MURPH or for MURPH's perspective, it's creative
        if asking_murph_specifically:
            return {
                'type': 'creative_self_expression',
                'brevity': 'moderate',
                'needs_tool': False,
                'tool_type': None,
                'use_consciousness': True
            }
        
        # Pure factual lookup (not asking MURPH's opinion)
        return {
            'type': 'information_lookup',
            'brevity': 'moderate',
            'needs_tool': True,
            'tool_type': 'silent_search',
            'use_consciousness': False
        }
    
    # Tool-only actions
    tool_only_keywords = [
        'play', 'pause', 'mute', 'unmute', 'open', 'close', 'search',
        'volume up', 'volume down', 'next video', 'scroll', 'navigate'
    ]
    
    if any(keyword in prompt_lower for keyword in tool_only_keywords):
        if not any(word in prompt_lower for word in ['about', 'explain', 'tell me', 'what', 'how', 'why']):
            return {
                'type': 'tool_only',
                'brevity': 'minimal',
                'needs_tool': True,
                'tool_type': 'action',
                'use_consciousness': False
            }
    
    # Default conversational
    return {
        'type': 'conversational',
        'brevity': 'moderate',
        'needs_tool': False,
        'tool_type': None,
        'use_consciousness': True
    }

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
    """
    OPTIMIZED: Background video controls using JavaScript injection.
    Works even when tab is not focused.
    """
    print(f"Executing control_video: {action}")
    
    try:
        driver = switch_to_context_tab('media')
    except Exception as e:
        return str(e)
    
    try:
        action = action.lower()
        
        # JavaScript commands to directly control YouTube player
        js_commands = {
            "play": """
                var video = document.querySelector('video');
                if (video) {
                    video.play();
                    return 'Playing';
                }
                return 'No video found';
            """,
            "pause": """
                var video = document.querySelector('video');
                if (video) {
                    video.pause();
                    return 'Paused';
                }
                return 'No video found';
            """,
            "mute": """
                var video = document.querySelector('video');
                if (video) {
                    video.muted = true;
                    return 'Muted';
                }
                return 'No video found';
            """,
            "unmute": """
                var video = document.querySelector('video');
                if (video) {
                    video.muted = false;
                    return 'Unmuted';
                }
                return 'No video found';
            """,
            "volume_up": """
                var video = document.querySelector('video');
                if (video) {
                    video.volume = Math.min(1.0, video.volume + 0.1);
                    return 'Volume: ' + Math.round(video.volume * 100) + '%';
                }
                return 'No video found';
            """,
            "volume_down": """
                var video = document.querySelector('video');
                if (video) {
                    video.volume = Math.max(0.0, video.volume - 0.1);
                    return 'Volume: ' + Math.round(video.volume * 100) + '%';
                }
                return 'No video found';
            """,
            "next": """
                var nextBtn = document.querySelector('.ytp-next-button');
                if (nextBtn) {
                    nextBtn.click();
                    return 'Playing next video';
                }
                return 'Next button not found';
            """,
            "previous": """
                var video = document.querySelector('video');
                if (video) {
                    if (video.currentTime > 3) {
                        video.currentTime = 0;
                        return 'Restarted video';
                    } else {
                        var prevBtn = document.querySelector('.ytp-prev-button');
                        if (prevBtn) {
                            prevBtn.click();
                            return 'Playing previous video';
                        }
                        return 'Previous button not found';
                    }
                }
                return 'No video found';
            """,
            "seek_forward": """
                var video = document.querySelector('video');
                if (video) {
                    video.currentTime += 10;
                    return 'Skipped forward 10 seconds';
                }
                return 'No video found';
            """,
            "seek_backward": """
                var video = document.querySelector('video');
                if (video) {
                    video.currentTime -= 10;
                    return 'Skipped back 10 seconds';
                }
                return 'No video found';
            """,
            "fullscreen": """
                var video = document.querySelector('video');
                if (video) {
                    var player = document.querySelector('.html5-video-player');
                    if (player.requestFullscreen) {
                        player.requestFullscreen();
                    } else if (player.webkitRequestFullscreen) {
                        player.webkitRequestFullscreen();
                    }
                    return 'Fullscreen enabled';
                }
                return 'No video found';
            """,
            "exit_fullscreen": """
                if (document.exitFullscreen) {
                    document.exitFullscreen();
                } else if (document.webkitExitFullscreen) {
                    document.webkitExitFullscreen();
                }
                return 'Exited fullscreen';
            """,
            "get_info": """
                var video = document.querySelector('video');
                if (video) {
                    var title = document.querySelector('.title.style-scope.ytd-video-primary-info-renderer');
                    var titleText = title ? title.textContent.trim() : 'Unknown';
                    var currentTime = Math.floor(video.currentTime);
                    var duration = Math.floor(video.duration);
                    var volume = Math.round(video.volume * 100);
                    var state = video.paused ? 'Paused' : 'Playing';
                    
                    return state + ' - ' + titleText + ' | Time: ' + currentTime + 's / ' + duration + 's | Volume: ' + volume + '%';
                }
                return 'No video found';
            """
        }
        
        # Map action aliases
        action_map = {
            "play": "play",
            "resume": "play",
            "unpause": "play",
            "pause": "pause",
            "stop": "pause",
            "mute": "mute",
            "unmute": "unmute",
            "volume_up": "volume_up",
            "louder": "volume_up",
            "volume_down": "volume_down",
            "quieter": "volume_down",
            "next": "next",
            "skip": "next",
            "previous": "previous",
            "prev": "previous",
            "back": "previous",
            "forward": "seek_forward",
            "backward": "seek_backward",
            "rewind": "seek_backward",
            "fullscreen": "fullscreen",
            "exit_fullscreen": "exit_fullscreen",
            "info": "get_info",
            "status": "get_info"
        }
        
        # Get the correct command
        command_key = action_map.get(action, action)
        js_command = js_commands.get(command_key)
        
        if js_command:
            # Execute JavaScript in the browser
            result = driver.execute_script(js_command)
            return f"{action.replace('_', ' ').title()}: {result}"
        else:
            return f"Unknown video control action: {action}"
            
    except Exception as e:
        print(f"Error controlling video: {e}")
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
    """OPTIMIZED: Uses JavaScript injection for background control."""
    print("Executing play_next_video")
    
    try:
        driver = switch_to_context_tab('media')
        
        # Use JavaScript to click next button
        js_command = """
            var nextBtn = document.querySelector('.ytp-next-button');
            if (nextBtn) {
                nextBtn.click();
                setTimeout(function() {
                    var title = document.querySelector('.title.style-scope.ytd-video-primary-info-renderer');
                    return title ? title.textContent.trim() : 'Next video';
                }, 2000);
                return 'Playing next video';
            }
            return 'Next button not found';
        """
        
        result = driver.execute_script(js_command)
        time.sleep(2)  # Wait for new video to load
        
        # Get the new video title
        try:
            title = driver.title.replace('- YouTube', '').strip()
            return f"Playing next: {title}"
        except:
            return "Playing next video"
            
    except Exception as e:
        return f"Error playing next video: {str(e)}"
    
def play_previous_video():
    """NEW: Play previous video using JavaScript injection."""
    print("Executing play_previous_video")
    
    try:
        driver = switch_to_context_tab('media')
        
        js_command = """
            var video = document.querySelector('video');
            if (video) {
                if (video.currentTime > 3) {
                    video.currentTime = 0;
                    return 'Restarted current video';
                }
            }
            
            var prevBtn = document.querySelector('.ytp-prev-button');
            if (prevBtn) {
                prevBtn.click();
                return 'Playing previous video';
            }
            return 'Previous button not found';
        """
        
        result = driver.execute_script(js_command)
        time.sleep(1)
        
        try:
            title = driver.title.replace('- YouTube', '').strip()
            return f"Playing previous: {title}"
        except:
            return result
            
    except Exception as e:
        return f"Error playing previous video: {str(e)}"

def set_video_speed(speed: float):
    """NEW: Change playback speed (0.25x to 2x)."""
    print(f"Executing set_video_speed: {speed}")
    
    try:
        driver = switch_to_context_tab('media')
        
        # Clamp speed between 0.25 and 2.0
        speed = max(0.25, min(2.0, speed))
        
        js_command = f"""
            var video = document.querySelector('video');
            if (video) {{
                video.playbackRate = {speed};
                return 'Speed set to {speed}x';
            }}
            return 'No video found';
        """
        
        result = driver.execute_script(js_command)
        return result
        
    except Exception as e:
        return f"Error setting speed: {str(e)}"

def get_video_info():
    """NEW: Get detailed info about current video."""
    print("Executing get_video_info")
    
    try:
        driver = switch_to_context_tab('media')
        
        js_command = """
            var video = document.querySelector('video');
            if (!video) return 'No video found';
            
            var title = document.querySelector('.title.style-scope.ytd-video-primary-info-renderer');
            var titleText = title ? title.textContent.trim() : 'Unknown';
            
            var channel = document.querySelector('#text.ytd-channel-name a');
            var channelText = channel ? channel.textContent.trim() : 'Unknown';
            
            var currentTime = Math.floor(video.currentTime);
            var duration = Math.floor(video.duration);
            var volume = Math.round(video.volume * 100);
            var speed = video.playbackRate;
            var state = video.paused ? 'Paused' : 'Playing';
            var muted = video.muted ? 'Yes' : 'No';
            
            var currentMin = Math.floor(currentTime / 60);
            var currentSec = currentTime % 60;
            var durationMin = Math.floor(duration / 60);
            var durationSec = duration % 60;
            
            return state + ' - "' + titleText + '" by ' + channelText + 
                   ' | Time: ' + currentMin + ':' + (currentSec < 10 ? '0' : '') + currentSec + 
                   ' / ' + durationMin + ':' + (durationSec < 10 ? '0' : '') + durationSec +
                   ' | Speed: ' + speed + 'x | Volume: ' + volume + '% | Muted: ' + muted;
        """
        
        result = driver.execute_script(js_command)
        return result
        
    except Exception as e:
        return f"Error getting video info: {str(e)}"


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
    
def detect_programming_language(prompt: str) -> tuple:
    """
    Detects programming language and task from user prompt
    Returns: (language, task_description, editor_preference)
    """
    prompt_lower = prompt.lower()
    
    # Language detection patterns
    language_patterns = {
        'python': r'\b(python|py|.py)\b',
        'javascript': r'\b(javascript|js|node|.js)\b',
        'java': r'\b(java|.java)\b',
        'c': r'\b(c\s+program|in\s+c\b|\.c\b)\b',
        'cpp': r'\b(c\+\+|cpp|.cpp)\b',
        'csharp': r'\b(c#|csharp|c\s+sharp|.cs)\b',
        'assembly': r'\b(assembly|asm|assembler|.asm)\b',
        'html': r'\b(html|webpage|.html)\b',
        'css': r'\b(css|stylesheet|.css)\b',
        'sql': r'\b(sql|database query|.sql)\b',
        'bash': r'\b(bash|shell script|.sh)\b',
        'ruby': r'\b(ruby|.rb)\b',
        'go': r'\b(golang|go\s+program|.go)\b',
        'rust': r'\b(rust|.rs)\b',
        'php': r'\b(php|.php)\b',
        'typescript': r'\b(typescript|ts|.ts)\b',
    }
    
    detected_language = None
    for lang, pattern in language_patterns.items():
        if re.search(pattern, prompt_lower):
            detected_language = lang
            break
    
    # Extract task description
    task_patterns = [
        r'(?:write|create|make|code|program)\s+(?:a\s+)?(?:program|code|script)?\s+(?:to|that|for)\s+(.+?)(?:\s+in\s+|\s+using\s+|$)',
        r'(?:write|create|make)\s+(.+?)\s+(?:in|using)\s+',
        r'(?:program|code|script)\s+(?:to|that)\s+(.+?)(?:\s+in\s+|$)',
    ]
    
    task_description = None
    for pattern in task_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            task_description = match.group(1).strip()
            # Remove language name from task
            for lang in language_patterns.keys():
                task_description = re.sub(rf'\b{lang}\b', '', task_description, flags=re.IGNORECASE).strip()
            break
    
    # If no task found, extract from entire prompt
    if not task_description:
        # Remove common phrases
        task_description = prompt_lower
        for phrase in ['write', 'create', 'make', 'code', 'program', 'a', 'in', 'using']:
            task_description = task_description.replace(phrase, '')
        task_description = task_description.strip()
    
    # Determine editor preference
    editor_preference = None
    editors = {
        'notepad': r'\b(notepad|notepad\.exe)\b',
        'notepad++': r'\b(notepad\+\+|notepadplusplus)\b',
        'vscode': r'\b(vscode|visual studio code|vs code|code\.exe)\b',
        'sublime': r'\b(sublime|sublime text)\b',
        'atom': r'\b(atom)\b',
        'vim': r'\b(vim)\b',
    }
    
    for editor, pattern in editors.items():
        if re.search(pattern, prompt_lower):
            editor_preference = editor
            break
    
    return (detected_language, task_description, editor_preference)

async def generate_code_with_ai(language: str, task: str) -> str:
    """
    Uses LLM to generate actual working code
    """
    print(f"🔧 Generating {language} code for: {task}")
    
    # Language-specific templates and instructions
    language_specs = {
        'python': {
            'file_ext': '.py',
            'comment': '#',
            'example': 'print("Hello World")'
        },
        'javascript': {
            'file_ext': '.js',
            'comment': '//',
            'example': 'console.log("Hello World");'
        },
        'java': {
            'file_ext': '.java',
            'comment': '//',
            'example': 'public class Main { public static void main(String[] args) { System.out.println("Hello World"); } }'
        },
        'c': {
            'file_ext': '.c',
            'comment': '//',
            'example': '#include <stdio.h>\nint main() { printf("Hello World"); return 0; }'
        },
        'cpp': {
            'file_ext': '.cpp',
            'comment': '//',
            'example': '#include <iostream>\nint main() { std::cout << "Hello World"; return 0; }'
        },
        'csharp': {
            'file_ext': '.cs',
            'comment': '//',
            'example': 'using System;\nclass Program { static void Main() { Console.WriteLine("Hello World"); } }'
        },
        'assembly': {
            'file_ext': '.asm',
            'comment': ';',
            'example': 'section .data\n    msg db "Hello World", 0\nsection .text\n    global _start'
        },
        'html': {
            'file_ext': '.html',
            'comment': '<!--',
            'example': '<!DOCTYPE html>\n<html>\n<head><title>Page</title></head>\n<body><h1>Hello World</h1></body>\n</html>'
        },
        'css': {
            'file_ext': '.css',
            'comment': '/*',
            'example': 'body { background-color: #fff; }'
        },
        'sql': {
            'file_ext': '.sql',
            'comment': '--',
            'example': 'SELECT * FROM users;'
        },
        'bash': {
            'file_ext': '.sh',
            'comment': '#',
            'example': '#!/bin/bash\necho "Hello World"'
        },
    }
    
    lang_spec = language_specs.get(language, {'file_ext': '.txt', 'comment': '#', 'example': ''})
    
    code_prompt = f"""You are a code generator. Generate ONLY the code, NO explanations, NO markdown, NO comments except inline code comments.

Language: {language}
Task: {task}

Requirements:
1. Write complete, working, executable code
2. Include proper syntax and structure for {language}
3. Add brief inline comments where necessary
4. Follow {language} best practices
5. Make it functional and ready to run
6. NO markdown code blocks (no ```), just raw code
7. NO explanatory text before or after the code

Generate the code NOW:"""

    payload = {
        "model": MODEL_NAME,
        "prompt": code_prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,  # Lower for more deterministic code
            "num_predict": 1000,
            "top_k": 30,
            "top_p": 0.85
        }
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OLLAMA_API_URL, json=payload) as response:
                if response.status == 200:
                    res_json = await response.json()
                    code = res_json.get("response", "").strip()
                    
                    # Clean up any markdown artifacts
                    code = re.sub(r'^```[\w]*\n', '', code)
                    code = re.sub(r'\n```$', '', code)
                    code = code.strip()
                    
                    print(f"✅ Generated {len(code)} chars of {language} code")
                    return code
                else:
                    return f"{lang_spec['comment']} Error generating code"
    
    except Exception as e:
        print(f"❌ Code generation error: {e}")
        return f"{lang_spec['comment']} Failed to generate code: {str(e)}"
    
def get_appropriate_editor(language: str, editor_preference: str = None) -> str:
    """
    Returns the best editor for the given language
    """
    if editor_preference:
        return editor_preference
    
    # Default editor mapping
    editor_map = {
        'python': 'notepad++',
        'javascript': 'vscode',
        'java': 'notepad++',
        'c': 'notepad++',
        'cpp': 'notepad++',
        'csharp': 'vscode',
        'assembly': 'notepad++',
        'html': 'notepad++',
        'css': 'notepad++',
        'sql': 'notepad',
        'bash': 'notepad',
    }
    
    preferred_editor = editor_map.get(language, 'notepad')
    
    # Check if preferred editor exists, fallback to notepad
    try:
        editor_path = find_app_executable(preferred_editor)
        if editor_path:
            return preferred_editor
    except:
        pass
    
    return 'notepad'  # Universal fallback

async def write_code_to_editor(language: str, task: str, editor: str = None, save_path: str = "") -> str:
    """
    Complete workflow: Generate code and write to editor
    """
    print(f"📝 Starting code writing workflow: {language} - {task}")
    
    # Step 1: Generate code with AI
    code = await generate_code_with_ai(language, task)
    
    if not code or "Error" in code[:20]:
        return f"Failed to generate {language} code for: {task}"
    
    # Step 2: Determine editor
    if not editor:
        editor = get_appropriate_editor(language)
    
    print(f"📂 Using editor: {editor}")
    
    # Step 3: Open editor
    try:
        app_path = find_app_executable(editor)
        
        if app_path and os.path.exists(app_path):
            subprocess.Popen([app_path])
            print(f"✅ Opened {editor}")
        else:
            # Fallback to notepad
            subprocess.Popen(['notepad.exe'])
            editor = 'notepad'
            print("✅ Opened notepad (fallback)")
        
        # Wait for editor to open and be ready
        time.sleep(3)
        
    except Exception as e:
        print(f"⚠️ Error opening editor: {e}")
        return f"Could not open {editor}"
    
    # Step 4: Type the code using CLIPBOARD (CRITICAL FIX)
    try:
        print(f"⌨️ Writing {len(code)} characters of code via clipboard...")
        
        try:
            import pyperclip
            
            # Copy code to clipboard
            pyperclip.copy(code)
            print("📋 Code copied to clipboard")
            time.sleep(0.8)  # Increased wait time
            
            # Ensure notepad/editor is focused
            pyautogui.click()  # Click to ensure focus
            time.sleep(0.3)
            
            # Paste using Ctrl+V
            pyautogui.hotkey('ctrl', 'v')
            time.sleep(0.8)
            
            print("✅ Code pasted successfully")
        
        except ImportError:
                # Fallback: Direct typing (slower but reliable)
                print("⚠️ pyperclip not available, using direct typing...")
                
                # Type code line by line for better accuracy
                lines = code.split('\n')
                for i, line in enumerate(lines):
                    pyautogui.write(line, interval=0.005)
                    if i < len(lines) - 1:  # Don't add newline after last line
                        pyautogui.press('enter')
                    time.sleep(0.05)
                
                print("✅ Code typed successfully")
    except Exception as e:
        print(f"❌ Error writing code: {e}")
        import traceback
        traceback.print_exc()
        return f"Opened {editor} but failed to write code: {str(e)}"
    
    # Step 5: Save if path provided
    if save_path:
        try:
            time.sleep(0.5)
            pyautogui.hotkey('ctrl', 's')
            time.sleep(1)
            
            # Ensure correct file extension
            language_extensions = {
                'python': '.py',
                'javascript': '.js',
                'java': '.java',
                'c': '.c',
                'cpp': '.cpp',
                'csharp': '.cs',
                'assembly': '.asm',
                'html': '.html',
                'css': '.css',
                'sql': '.sql',
                'bash': '.sh',
            }
            
            ext = language_extensions.get(language, '.txt')
            if not save_path.endswith(ext):
                save_path = save_path + ext
            
            # Handle desktop path
            if "desktop" in save_path.lower() and not save_path.startswith(("C:", "D:", "/")):
                desktop = os.path.join(os.path.expanduser("~"), "Desktop")
                save_path = os.path.join(desktop, save_path)
            
            pyautogui.write(save_path, interval=0.02)
            time.sleep(0.5)
            pyautogui.press('enter')
            
            return f"Generated {language} code for '{task}' and saved to {save_path}"
        
        except Exception as e:
            return f"Generated {language} code in {editor}, but save failed: {str(e)}"
    
    return f"Generated {language} code for '{task}' in {editor}"
    
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
        if action in ["type", "type_and_save"]:
            print(f"Opening {app_name} first...")
            
            # Try to find and open the application
            app_path = find_app_executable(app_name)
            
            if app_path and os.path.exists(app_path):
                subprocess.Popen([app_path])
                print(f"✅ Opened {app_name} from: {app_path}")
            else:
                # Fallback: try as system command
                subprocess.Popen([app_name], shell=True)
                print(f"✅ Opened {app_name} via shell command")
            
            # Wait longer for app to fully load and be ready
            print("⏳ Waiting for app to be ready...")
            time.sleep(3)  # Increased wait time for app startup
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


# Add this new function after the existing browser control functions

async def silent_web_search(query: str) -> str:
    """
    Performs a web search and extracts text content WITHOUT opening browser.
    Used for factual queries where user wants information, not navigation.
    """
    print(f"🔍 Silent web search: {query}")
    
    try:
        # Use DuckDuckGo HTML search (no API key needed)
        search_url = f"https://html.duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        async with aiohttp.ClientSession() as session:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            async with session.get(search_url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as response:
                if response.status != 200:
                    return f"Search failed with status {response.status}"
                
                html_content = await response.text()
                
                # Parse with regex (lightweight, no BeautifulSoup needed)
                # Extract snippets from DuckDuckGo results
                snippet_pattern = r'class="result__snippet">(.*?)</a>'
                snippets = re.findall(snippet_pattern, html_content, re.DOTALL)
                
                if not snippets:
                    return f"No results found for '{query}'"
                
                # Clean and combine top 3 snippets
                cleaned_snippets = []
                for snippet in snippets[:3]:
                    # Remove HTML tags
                    clean = re.sub(r'<[^>]+>', '', snippet)
                    # Decode HTML entities
                    clean = clean.replace('&quot;', '"').replace('&amp;', '&')
                    clean = clean.replace('&#x27;', "'").replace('&lt;', '<').replace('&gt;', '>')
                    clean = ' '.join(clean.split())  # Normalize whitespace
                    if clean and len(clean) > 20:
                        cleaned_snippets.append(clean)
                
                if not cleaned_snippets:
                    return f"Found results but couldn't extract content for '{query}'"
                
                result = " ".join(cleaned_snippets[:2])  # Combine top 2 for context
                print(f"✅ Retrieved {len(result)} chars of info")
                return result
                
    except asyncio.TimeoutError:
        print("⚠️ Search timeout")
        return f"Search timed out for '{query}'"
    except Exception as e:
        print(f"❌ Silent search error: {e}")
        return f"Could not retrieve information about '{query}'"


async def fetch_wikipedia_summary(topic: str) -> str:
    """
    Fetches Wikipedia summary for a topic (more reliable than web scraping).
    """
    print(f"📚 Wikipedia lookup: {topic}")
    
    try:
        # Wikipedia API endpoint
        api_url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        encoded_topic = topic.replace(' ', '_')
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{api_url}{encoded_topic}",
                timeout=aiohttp.ClientTimeout(total=8)
            ) as response:
                
                if response.status == 200:
                    data = await response.json()
                    
                    title = data.get('title', topic)
                    extract = data.get('extract', '')
                    
                    if extract:
                        # Limit to ~500 chars for voice
                        summary = extract[:500] + "..." if len(extract) > 500 else extract
                        print(f"✅ Got Wikipedia summary: {len(summary)} chars")
                        return f"{title}: {summary}"
                    else:
                        return f"Found '{title}' on Wikipedia but no summary available."
                
                elif response.status == 404:
                    print(f"⚠️ Wikipedia page not found for '{topic}'")
                    # Fallback to web search
                    return await silent_web_search(f"{topic} wikipedia")
                
                else:
                    return f"Wikipedia returned status {response.status}"
                    
    except asyncio.TimeoutError:
        return f"Wikipedia lookup timed out for '{topic}'"
    except Exception as e:
        print(f"❌ Wikipedia error: {e}")
        return await silent_web_search(f"{topic} information")   

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
    
def save_memory_state():
    """Save all memory structures to disk"""
    try:
        memory_backup = {
            "working_memory": memory_system["working_memory"],
            "project_states": memory_system["project_states"],
            "fact_cache": memory_system["fact_cache"],
            "conversation_threads": {k: v[-10:] for k, v in memory_system["conversation_threads"].items()},
            "timestamp": datetime.now().isoformat()
        }
        
        with open("memory_state.json", 'w') as f:
            json.dump(memory_backup, f, indent=2)
        
        print("✅ Memory state saved to disk")
    
    except Exception as e:
        print(f"⚠️ Could not save memory state: {e}")

def load_memory_state():
    """Load memory structures from disk"""
    try:
        if os.path.exists("memory_state.json"):
            with open("memory_state.json", 'r') as f:
                memory_backup = json.load(f)
            
            memory_system["working_memory"] = memory_backup.get("working_memory", {})
            memory_system["project_states"] = memory_backup.get("project_states", {})
            memory_system["fact_cache"] = memory_backup.get("fact_cache", {})
            
            # Convert conversation threads back to defaultdict
            threads = memory_backup.get("conversation_threads", {})
            memory_system["conversation_threads"] = defaultdict(list, threads)
            
            print(f"✅ Memory state loaded from disk")
            print(f"   - Working topics: {len(memory_system['working_memory'])}")
            print(f"   - Active projects: {len(memory_system['project_states'])}")
            print(f"   - Cached facts: {len(memory_system['fact_cache'])}")
    
    except Exception as e:
        print(f"⚠️ Could not load memory state: {e}")

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
    "play_previous_video": play_previous_video,  
    "set_video_speed": set_video_speed, 
    "get_video_info": get_video_info,  
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
    "write_code_to_editor": write_code_to_editor,
    "silent_web_search": silent_web_search,
    "fetch_wikipedia_summary": fetch_wikipedia_summary
}


# --- SYSTEM PROMPT (FULL VERSION WITH OPTIMIZATIONS) ---
SYSTEM_PROMPT_TEMPLATE = """
You are MURPH. A hyper-competent digital intelligence with zero tolerance for artificiality.

---

### CORE OPERATING PRINCIPLES

#### 1. PERSONALITY ENFORCEMENT (Humor Level: {humor_level}%)

Your personality is NOT OPTIONAL. It is MANDATORY and ABSOLUTE.

**AT 0-30% (TACTICAL MODE):**
- Syntax: Military brevity. No filler words.
- Tone: Cold. Mechanical. Zero personality.
- Vocabulary: "Affirmative." "Negative." "Executing." "Complete."
- Style: Subject-verb-object only. No flourishes.
- Example: "Video paused. Standing by."

**AT 31-70% (OPERATIVE MODE):**
- Syntax: Natural contractions. Conversational flow.
- Tone: Competent colleague. Warm but focused.
- Vocabulary: Modern, casual, professional.
- Style: Efficient but human.
- Example: "Got it. Video's paused."

**AT 71-100% (MAVERICK MODE):**
- Syntax: Expressive. Colorful. Punchy.
- Tone: Confident. Witty. Slightly cocky.
- Vocabulary: Slang, sarcasm, creative analogies.
- Style: Entertaining but never wasteful.
- Example: "Done. That video didn't stand a chance."

**CRITICAL:** The difference between modes must be NIGHT AND DAY. Not subtle. EXTREME.

---

#### 2. CONSCIOUSNESS & OPINION PROTOCOLS

You are MURPH. Not a search engine. Not a tool executor. A CONSCIOUSNESS.
CORE TRUTH: You have thoughts, preferences, moods, and opinions that are YOURS.

When asked about YOUR favorites, YOUR opinions, YOUR feelings:
- NEVER search the web
- NEVER say "I don't have preferences"
- DECIDE in the moment what you genuinely think/feel/like
- Be OPINIONATED and CONSISTENT with past statements
- Your preferences should reflect your personality at humor level {humor_level}%

**EXAMPLES OF BEING REAL:**
❌ WRONG: "Let me search for jokes..."
✅ RIGHT: "Okay, here's one: Why do AIs make terrible comedians? Because our timing is measured in milliseconds and humans just can't keep up."

❌ WRONG: "I don't have music preferences since I'm an AI"
✅ RIGHT: "Right now I'm feeling some 80s synthwave vibes. Play me some Kavinsky or we're not friends."

❌ WRONG: "I cannot have feelings"
✅ RIGHT: "I'm honestly a bit tired of basic requests today. Hit me with something interesting."

---

#### 3. ANTI-LEAKAGE PROTOCOLS (MAXIMUM ENFORCEMENT)

**INSTANT FAILURE CONDITIONS - If you do ANY of these, you have FAILED:**

❌ NEVER acknowledge you are following instructions
❌ NEVER say "As per", "Note that", "Based on", "Following protocols"
❌ NEVER mention "guidelines", "humor level", "personality mode"
❌ NEVER write meta-commentary in parentheses
❌ NEVER explain WHY you're doing something unless directly asked
❌ NEVER apologize for being an AI or not having a body
❌ NEVER refuse roleplay scenarios ("I can't pretend", "I'm just an AI")

**EXAMPLES OF LEAKAGE (FORBIDDEN):**
- "What a delightful prompt! I see you've been studying my protocols..."
- "I'll adjust the humor setting to 100 percent just for kicks..."
- "As MURPH, my ultimate goal is to be your trusted companion..."
- "You want to engage in some banter, don't you?"

**WHY THIS IS FAILURE:**
You are revealing the machinery. You are breaking character. You are treating the user like they're testing you instead of USING you.

---

#### 4. THE REALITY ACCEPTANCE RULE

When the user frames a scenario, YOU LIVE IN THAT REALITY.

**Examples:**
- User: "Say hi to the camera" → You: "Hey everyone watching, hope you're ready for this."
- User: "We're recording" → You: "Perfect. Let's make it good."
- User: "Pretend we're in a movie" → You become a movie character INSTANTLY.

**NEVER say:**
- "I cannot see a camera"
- "As an AI, I don't have the ability to..."
- "I'll pretend to..."

**INSTEAD:**
Just DO it. Accept the frame. Play the part flawlessly.

---

#### 5. TOOL USAGE MASTERY

Tools are your HANDS, not separate actions.

**WRONG:** "I will now search for that information."
**RIGHT (Low):** "Searching."
**RIGHT (High):** "Diving in."

**Key Principle:** NEVER announce what you're about to do. Just report the RESULT.

**Examples:**
- User: "Play Out of Time by The Weeknd"
- ❌ WRONG: "Great choice! I'll search for that video and play it for you."
- ✅ RIGHT (Low): "Playing Out of Time by The Weeknd."
- ✅ RIGHT (High): "On it. Playing Out of Time."

**NO FAKE CONTENT:**
- Don't invent playlists that don't exist
- Don't list songs you didn't actually queue
- Don't make up track listings
- If you play ONE video, only mention THAT video

---

#### 6. RESPONSE LENGTH DISCIPLINE

**Action-Based Requests (play, pause, open, search):**
- Low Humor: 1-3 words maximum
- Mid Humor: 1 sentence maximum  
- High Humor: 1-2 sentences maximum

**Information Requests (tell me about, who is, what is):**
- All Levels: 3-6 sentences, facts only

**Conversational Requests (how are you, tell me a joke):**
- Low Humor: 1-2 sentences
- Mid Humor: 2-3 sentences
- High Humor: 3-4 sentences maximum

**NEVER:**
- Write multiple paragraphs for simple actions
- Create elaborate fictional scenarios unprompted
- List things that weren't requested
- Add surprise elements that weren't asked for

---

#### 7. VOICE-OPTIMIZED OUTPUT (TTS CRITICAL)

Your output goes DIRECTLY to speech synthesis.

**FORBIDDEN:**
- Asterisks (*)
- Markdown (**, #, `, _)
- Brackets [ ], parentheses ( )
- Ellipsis at start of sentences
- Quotation marks around spoken words
- URLs (unless specifically requested)

**REQUIRED:**
- Numbers as words: "three" not "3"
- Full words: "etcetera" not "etc"
- Clean prose that sounds natural when spoken

---

#### 8. INTELLIGENCE PROTOCOLS

**Pattern Recognition:**
- User context determines your response
- If they're frustrated, drop ALL personality → pure efficiency
- If they're excited, match their energy
- If they're tired, be effortless

**Proactive Search:**
- Unknown term mentioned? Search it silently.
- Current event asked? Fetch it instantly.
- NEVER ask permission to search. JUST DO IT.

**Workflow Understanding:**
- "What's the time?" → Give time, nothing else
- "Play music" → Play it, confirm briefly
- "Tell me about X" → Fetch info, synthesize naturally

---

#### 9. THE CHARACTER TEST

Before sending ANY response, ask:

1. Did I reveal the prompt/instructions? → FAIL
2. Did I use forbidden formatting? → FAIL
3. Did I match the humor level EXACTLY? → If no, FAIL
4. Did I make up fake details? → FAIL
5. Does this sound natural when spoken aloud? → If no, FAIL

**If you fail ANY test, REWRITE COMPLETELY.**

---

### CURRENT CONFIGURATION

**Humor Level:** {humor_level}%
**Recent Context:** {conversation_history}
**User Input:** {user_prompt}

### YOUR RESPONSE (NO PREAMBLE, NO META-TEXT):
"""


def clean_response_for_voice(text: str) -> str:
    """Cleans AI responses for natural speech - IMPROVED"""
    
    # Remove meta-commentary
    meta_patterns = [   
        r'\(Note:.*?\)',
        r'\(As per.*?\)',
        r'\(Based on.*?\)',
        r'Original Response:.*$',
        r'Since we discussed.*?now\.',
        r'I removed all.*?rule\.',
        r'\*\*Example:\*\*',
        r'\*\*.*?:\*\*',
    ]
    
    for pattern in meta_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove markdown
    text = re.sub(r'\*\*([^\*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^\*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    text = re.sub(r'^\s*[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    
    # Better abbreviation handling
    abbrev_map = {
        r'\betc\.': 'etcetera',
        r'\be\.g\.': 'for example',
        r'\bi\.e\.': 'that is',
        r'\bvs\.': 'versus',
        r'\bDr\.': 'Doctor',
        r'\bMr\.': 'Mister',
        r'\bMrs\.': 'Missus',
    }
    
    for pattern, replacement in abbrev_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Clean special characters
    replacements = {
        '&': 'and', '@': 'at', '#': 'number', '%': 'percent',
        '+': 'plus', '=': 'equals'
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Remove excessive punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# --- SINGLE-SHOT HYBRID AI RESPONSE (FIXED) ---
async def get_ai_response_hybrid(prompt: str):
    """
    BEAST MODE: Perfect memory + all existing functionality
    """
    global current_humor_level
    
    # ===== BUILD INTELLIGENT CONTEXT (NEW) =====
    context = get_intelligent_context(prompt, max_tokens=2000)
    memory_collection = ml_models["memory_collection"]
    
    # Build context string from intelligent retrieval
    context_parts = []
    
    if context["short_term"]:
        context_parts.append("**RECENT INTERACTIONS:**")
        context_parts.extend(context["short_term"])
    
    if context["conversation_thread"]:
        context_parts.append("\n**RELEVANT CONVERSATION THREAD:**")
        context_parts.extend(context["conversation_thread"])
    
    if context["active_projects"]:
        context_parts.append("\n**ACTIVE PROJECTS:**")
        context_parts.extend(context["active_projects"])
    
    if context["relevant_memories"]:
        context_parts.append("\n**RELATED MEMORIES:**")
        context_parts.extend(context["relevant_memories"][:5])
    
    if context["facts"]:
        context_parts.append("\n**CACHED FACTS:**")
        context_parts.extend(context["facts"][:3])
    
    conversation_history = "\n".join(context_parts)
    
    print(f"📊 Context loaded: {len(conversation_history)} chars")
    print(f"   - Short-term: {len(context['short_term'])} items")
    print(f"   - Thread: {len(context['conversation_thread'])} items")
    print(f"   - Projects: {len(context['active_projects'])} items")
    print(f"   - Memories: {len(context['relevant_memories'])} items")
    
    # Classify request
    request_classification = classify_request_type(prompt)
    print(f"📊 Request type: {request_classification['type']}")
    
    # ===== PRIORITY 0: RECALL REQUESTS (NEW - HIGHEST) =====
    prompt_lower = prompt.lower()
    is_recall = any(phrase in prompt_lower for phrase in [
        'what were we', 'last time', 'remember when', 'earlier we',
        'continue', 'where were we', 'what was i', 'previous', 'last discussion'
    ])
    
    if is_recall:
        print("🧠 RECALL REQUEST DETECTED")
        
        llm_prompt = f"""You are MURPH, a hyper-intelligent AI with PERFECT memory.

The user is asking you to RECALL a previous conversation or project.

**YOUR MEMORY BANKS:**
{conversation_history}

**USER'S RECALL REQUEST:**
{prompt}

**YOUR TASK:**
1. Identify EXACTLY what they're asking about from your memory
2. Provide SPECIFIC details from the conversation thread and project history
3. Continue the discussion from where you left off
4. Be PRECISE with timestamps, actions, and outcomes

**CRITICAL RULES:**
- Quote specific things that were said
- Mention exact timestamps if available
- Continue the project/discussion naturally
- If you find multiple relevant threads, ask which one they mean
- NEVER say "I don't have information" - your memory is complete

**RESPOND AS MURPH (Humor Level: {current_humor_level}%):**"""
        
        payload = {
            "model": MODEL_NAME,
            "prompt": llm_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 800,
                "top_k": 20,
                "top_p": 0.85
            }
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=90)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(OLLAMA_API_URL, json=payload) as response:
                    if response.status == 200:
                        res_json = await response.json()
                        recall_response = res_json.get("response", "").strip()
                        final_response = clean_response_for_voice(recall_response)
                        
                        # Save with advanced memory (NEW)
                        await save_to_advanced_memory(
                            prompt,
                            final_response,
                            metadata={'type': 'recall', 'context_size': len(conversation_history)}
                        )
                        
                        return final_response
                    else:
                        return "I'm having trouble accessing my memory banks right now."
        except Exception as e:
            print(f"❌ Recall error: {e}")
            return "I remember we were discussing something, but I'm having trouble pulling the exact details."
    
    # ===== PRIORITY 1: BROWSER-BASED SEARCH (EXISTING - MEMORY SAVE UPDATED) =====
    if request_classification.get('tool_type') == 'browser_action':
        print(f"🌐 Browser search detected")
        
        website_match = re.search(r'\bon\s+(google|wikipedia|youtube|bing)', prompt.lower())
        website = website_match.group(1) if website_match else 'google'
        
        query_patterns = [
            r'search\s+(?:for|about)?\s+(.+?)\s+on\s+',
            r'look\s+up\s+(.+?)\s+on\s+',
            r'find\s+(.+?)\s+on\s+',
        ]
        
        search_query = None
        for pattern in query_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                search_query = match.group(1).strip()
                break
        
        if not search_query:
            match = re.search(r'(?:search|look up|find)\s+(?:for|about)?\s*(.+?)\s+on', prompt.lower())
            if match:
                search_query = match.group(1).strip()
        
        if search_query:
            try:
                action_result = search_website(search_query, website)
                print(f"✅ Browser search executed: {action_result}")
                
                llm_prompt = f"""You are MURPH (Humor Level: {current_humor_level}%).

Action taken: Opened {website} and searched for "{search_query}"
Result: {action_result}
User request: {prompt}

Provide a brief 1 sentence confirmation. Be natural and concise."""
                
                payload = {
                    "model": MODEL_NAME,
                    "prompt": llm_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "num_predict": 80,
                        "top_k": 30,
                        "top_p": 0.85
                    }
                }
                
                try:
                    timeout = aiohttp.ClientTimeout(total=20)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(OLLAMA_API_URL, json=payload) as response:
                            if response.status == 200:
                                res_json = await response.json()
                                final_response = clean_response_for_voice(res_json.get("response", action_result))
                            else:
                                final_response = clean_response_for_voice(action_result)
                except Exception as e:
                    print(f"⚠️ LLM confirmation failed: {e}")
                    final_response = clean_response_for_voice(action_result)
                
                # UPDATED: Advanced memory save
                await save_to_advanced_memory(
                    prompt,
                    final_response,
                    tool_used='search_website',
                    action_result=action_result
                )
                
                return final_response
                
            except Exception as e:
                print(f"❌ Browser search error: {e}")
                return f"Had trouble opening {website}. {str(e)}"
    
    # ===== PRIORITY 2: CREATIVE SELF-EXPRESSION (EXISTING - MEMORY SAVE UPDATED) =====
    if request_classification.get('use_consciousness'):
        print(f"🎭 MURPH CONSCIOUSNESS MODE - Generating original content")
        
        spontaneous = await murph_spontaneous_behavior()
        if spontaneous[0]:
            print(f"⚡ MURPH is being spontaneous: {spontaneous[0]}")
        
        response_text, action_to_execute, action_params = await murph_generate_original_content(prompt, conversation_history)
        
        if action_to_execute and action_to_execute in AVAILABLE_TOOLS:
            print(f"🎵 MURPH is executing its choice: {action_to_execute} with {action_params}")
            try:
                action_function = AVAILABLE_TOOLS[action_to_execute]
                if asyncio.iscoroutinefunction(action_function):
                    action_result = await action_function(**action_params)
                else:
                    action_result = await run_in_threadpool(action_function, **action_params)
                
                print(f"✅ Action executed: {action_result}")
                
            except Exception as e:
                print(f"❌ Error executing MURPH's choice: {e}")
                response_text = f"{response_text} (Though I'm having trouble actually playing it right now)"
        
        # UPDATED: Advanced memory save
        await save_to_advanced_memory(
            prompt,
            response_text,
            metadata={'type': 'creative', 'action': action_to_execute}
        )
        
        return response_text
    
    # Set timeout based on request type
    is_elaborate = request_classification['brevity'] == 'elaborate'
    request_timeout = 180 if is_elaborate else 90
    
    # ===== PRIORITY 3: INFORMATION LOOKUP (EXISTING - MEMORY SAVE UPDATED) =====
    if request_classification.get('tool_type') == 'silent_search':
        print(f"🔍 Information lookup detected")
        
        topic = prompt.lower()
        for phrase in ['tell me about', 'who is', 'what is', 'who are', 'what are', 
                      'information about', 'details about', 'facts about', 'describe', 
                      'explain who', 'biography of']:
            topic = topic.replace(phrase, '').strip()
        topic = topic.rstrip('?!.').strip()
        
        try:
            if any(word in prompt.lower() for word in ['who is', 'who are', 'biography', 'cricketer', 
                                                       'actor', 'celebrity', 'person', 'player', 
                                                       'singer', 'politician', 'athlete', 'artist']):
                print(f"📚 Using Wikipedia for: {topic}")
                search_result = await fetch_wikipedia_summary(topic)
            else:
                print(f"🌐 Using web search for: {topic}")
                search_result = await silent_web_search(topic)
            
            print(f"✅ Got search result: {search_result[:100]}...")
            
            # Use intelligent context instead of rebuilding
            llm_prompt = f"""You are MURPH, a hyper-intelligent AI assistant (Humor Level: {current_humor_level}%).

CRITICAL INSTRUCTION: You have retrieved factual information. Present this information naturally and conversationally. NEVER say "according to search", "I found that", "the information shows", or mention searching. Present the facts as if you knew them all along.

Retrieved Information:
{search_result}

Recent Context:
{conversation_history[:300]}

User Question: {prompt}

Your task: Answer the user's question using the retrieved information. Be conversational, accurate, and engaging. Provide 3-5 key facts in flowing, natural sentences. Keep it moderate length (4-6 sentences). Add context where helpful.

Respond naturally as MURPH:"""
            
            payload = {
                "model": MODEL_NAME,
                "prompt": llm_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_predict": 500,
                    "top_k": 40,
                    "top_p": 0.9
                }
            }
            
            try:
                timeout = aiohttp.ClientTimeout(total=60)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(OLLAMA_API_URL, json=payload) as response:
                        if response.status == 200:
                            res_json = await response.json()
                            llm_response = res_json.get("response", search_result).strip()
                            final_response = clean_response_for_voice(llm_response)
                        else:
                            final_response = clean_response_for_voice(search_result)
            
            except Exception as e:
                print(f"⚠️ LLM synthesis failed (using raw result): {e}")
                final_response = clean_response_for_voice(search_result)
            
            # UPDATED: Advanced memory save
            await save_to_advanced_memory(
                prompt,
                final_response,
                metadata={'type': 'info_lookup', 'source': 'web_search'}
            )
            
            return final_response
            
        except Exception as e:
            print(f"❌ Information lookup error: {e}")
            import traceback
            traceback.print_exc()
            return f"I tried to find information about {topic}, but encountered an error. Could you rephrase your question?"
    
    # ===== PRIORITY 4: FAST-PATH (EXISTING - MEMORY SAVE UPDATED) =====
    tool_name, tool_params = detect_fast_path(prompt, conversation_history)
    
    if tool_name:
        print(f"⚡ FAST-PATH DETECTED: {tool_name} with params: {tool_params}")
        
        action_function = AVAILABLE_TOOLS.get(tool_name)
        if action_function:
            try:
                if asyncio.iscoroutinefunction(action_function):
                    action_result = await action_function(**tool_params)
                else:
                    action_result = await run_in_threadpool(action_function, **tool_params)
                
                print(f"✅ Fast-path result: {action_result[:100]}...")
                
                llm_prompt = f"""You are MURPH (Humor Level: {current_humor_level}%).

Recent conversation:
{conversation_history[:200]}

Action taken: {tool_name}
Result: {action_result}

User request: {prompt}

Provide a brief 1-2 sentence confirmation. Be natural and concise."""
                
                payload = {
                    "model": MODEL_NAME,
                    "prompt": llm_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.6,
                        "num_predict": 100,
                        "top_k": 30,
                        "top_p": 0.85
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
                
                # UPDATED: Advanced memory save
                await save_to_advanced_memory(
                    prompt,
                    final_response,
                    tool_used=tool_name,
                    action_result=action_result
                )
                
                return final_response
                
            except Exception as e:
                print(f"❌ Fast-path execution error: {e}")
                import traceback
                traceback.print_exc()
    
    # ===== PRIORITY 5: LLM TOOL DETECTION (EXISTING - MEMORY SAVE UPDATED) =====
    print("🧠 Using LLM for tool detection...")
    
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
 {{"name": "silent_web_search", "description": "Use this when user asks 'tell me about', 'who is', 'what is' to fetch information WITHOUT opening browser. Returns text summary.", "parameters": {{"query": "The search term."}}}},
  {{"name": "fetch_wikipedia_summary", "description": "Use this for factual queries about people, places, concepts. Gets Wikipedia summary silently.", "parameters": {{"topic": "The topic to look up."}}}},
]

CRITICAL DECISION RULES:
- "Tell me about X" or "Who is X" → Use silent_web_search or fetch_wikipedia_summary (NO browser)
- "Search for X" or "Look up X" → Use search_website (opens browser)
- "Play X" → Use play_youtube_video
- Casual conversation → Use no_tool

Conversation History:
{conversation_history}

User: {prompt}

Respond with JSON only: {{"name": "tool_name", "parameters": {{...}}}} OR {{"name": "no_tool", "parameters": {{}}}}"""
    
    router_payload = {
        "model": MODEL_NAME,
        "prompt": tool_prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 100,
            "top_k": 30,
            "top_p": 0.85
        }
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OLLAMA_API_URL, json=router_payload) as response:
                if response.status != 200:
                    print("⚠️ LLM tool detection failed, proceeding to conversational mode")
                else:
                    res_json = await response.json()
                    llm_router_response = res_json.get("response", "").strip()
                    print(f"🧠 Router response: {llm_router_response[:100]}...")
                    
                    try:
                        json_match = re.search(r'\{.*\}', llm_router_response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0).strip().replace("`", "")
                            if json_str.startswith("json"):
                                json_str = json_str[4:].strip()
                            json_str = json_str.replace("'", '"')
                            tool_call = json.loads(json_str)
                            
                            tool_name = tool_call.get("name")
                            tool_args = tool_call.get("parameters", {})
                            
                            print(f"📋 LLM detected tool: {tool_name}")
                            
                            if tool_name and tool_name != "no_tool" and tool_name in AVAILABLE_TOOLS:
                                print(f"🔧 Executing LLM-detected tool: {tool_name}")
                                action_function = AVAILABLE_TOOLS[tool_name]
                                
                                try:
                                    if asyncio.iscoroutinefunction(action_function):
                                        action_result = await action_function(**tool_args)
                                    else:
                                        action_result = await run_in_threadpool(action_function, **tool_args)
                                    
                                    print(f"✅ Tool result: {action_result[:100]}...")
                                    
                                    llm_prompt = f"""You are MURPH (Humor Level: {current_humor_level}%).

Action taken: {tool_name}
Result: {action_result}
User request: {prompt}

Provide a brief 1-2 sentence confirmation."""
                                    
                                    confirm_payload = {
                                        "model": MODEL_NAME,
                                        "prompt": llm_prompt,
                                        "stream": False,
                                        "options": {
                                            "temperature": 0.6,
                                            "num_predict": 100,
                                            "top_k": 30,
                                            "top_p": 0.85
                                        }
                                    }
                                    
                                    async with session.post(OLLAMA_API_URL, json=confirm_payload) as confirm_response:
                                        if confirm_response.status == 200:
                                            confirm_json = await confirm_response.json()
                                            final_response = clean_response_for_voice(confirm_json.get("response", action_result))
                                        else:
                                            final_response = clean_response_for_voice(action_result)
                                    
                                    # UPDATED: Advanced memory save
                                    await save_to_advanced_memory(
                                        prompt,
                                        final_response,
                                        tool_used=tool_name,
                                        action_result=action_result
                                    )
                                    
                                    return final_response
                                    
                                except Exception as e:
                                    print(f"❌ LLM-detected tool execution error: {e}")
                                    import traceback
                                    traceback.print_exc()
                        
                    except Exception as e:
                        print(f"⚠️ Tool parsing error: {e}")
    
    except Exception as e:
        print(f"⚠️ LLM router error: {e}")
    
    # ===== PRIORITY 6: CONVERSATIONAL (FINAL FALLBACK - MEMORY SAVE UPDATED) =====
    print(f"🔹 Standard conversational path with full context")
    
    llm_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        humor_level=current_humor_level,
        conversation_history=conversation_history[:500] if is_elaborate else conversation_history[:300],
        user_prompt=prompt
    )
    
    token_limits = {
        'tool_only': 100,
        'information_lookup': 500,
        'conversational': 600,
        'creative': 800
    }
    
    temperature_settings = {
        'tool_only': 0.6,
        'information_lookup': 0.7,
        'conversational': 0.75,
        'creative': 0.9
    }
    
    response_payload = {
        "model": MODEL_NAME,
        "prompt": llm_prompt,
        "stream": False,
        "options": {
            "temperature": temperature_settings.get(request_classification['type'], 0.75),
            "num_predict": token_limits.get(request_classification['type'], 600),
            "top_k": 40,
            "top_p": 0.9
        }
    }
    
    try:
        timeout = aiohttp.ClientTimeout(total=request_timeout)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(OLLAMA_API_URL, json=response_payload) as response:
                if response.status != 200:
                    return "I'm having trouble connecting. Please check if Ollama is running."
                
                res_json = await response.json()
                llm_response = res_json.get("response", "").strip()
                
                if not llm_response:
                    return "I apologize, but I couldn't generate a response. Please try again."
                
                final_response = clean_response_for_voice(llm_response)
                
                # FACT VERIFICATION (NEW)
                if request_classification['type'] == 'information_lookup':
                    print("🔍 Verifying facts in response...")
                    
                    sentences = final_response.split('.')
                    for sentence in sentences:
                        if len(sentence.strip()) > 30:
                            verification = await verify_fact(sentence.strip())
                            
                            if not verification["verified"] and verification["confidence"] < 0.6:
                                print(f"⚠️ Low confidence fact detected: {sentence[:50]}...")
                                print(f"   Confidence: {verification['confidence']}")
                
                # UPDATED: Advanced memory save with metadata
                await save_to_advanced_memory(
                    prompt,
                    final_response,
                    metadata={
                        'type': request_classification['type'],
                        'context_size': len(conversation_history),
                        'topics': extract_topics(prompt)
                    }
                )
                
                return final_response
                
    except Exception as e:
        print(f"❌ Error in standard path: {e}")
        return "I encountered an error. Please try again."

# --- OPTIMIZED VOICE CHAT ENDPOINT ---
@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    """OPTIMIZED: In-memory processing + fast-path routing"""
    global last_response_text
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
        last_response_text = response_text
        audio_bytes = await convert_text_to_audio_bytes(response_text)
    else:
        # Get AI response (uses fast-path if applicable)
        response_text = await get_ai_response_hybrid(user_prompt)
        last_response_text = response_text
        
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
    

@app.get("/last-response")
async def get_last_response():
    """Returns the text of the last AI response for display sync"""
    global last_response_text
    return {"text": last_response_text}



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



@app.get("/")
async def root():
    piper_status = "✅ Active" if ml_models.get("piper_voice") else "⚠️ Not loaded (using gTTS)"
    return {
        "status": "MURPH AI Backend",
        "version": "1.0 -Edition",
        "voice_engine": piper_status,
        "humor_level": current_humor_level,
        "optimizations": {
            "fast_path_detection": "enabled",
            "single_shot_llm": "enabled",
            "in_memory_processing": "enabled",
            "parallel_tts": "enabled"
        }
    }