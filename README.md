# 🎤 Speech-to-Text Doctor–Patient — End-to-End Medical Transcription System

### 🔥 AI-Powered Transcription | Speaker Diarization | FastAPI | Real-Time Processing

[![Python](https://img.shields.io/badge/Python-3.9+-blue)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)]()
[![Status](https://img.shields.io/badge/Status-Active-success)]()
[![AI Models](https://img.shields.io/badge/AI-Whisper%20%7C%20Deepgram-orange)]()

---

## 🚀 Overview

**Speech-to-Text Doctor–Patient** is an AI-powered system that converts medical conversations into structured text with speaker identification.

It helps in:
- 🏥 Medical transcription  
- 📋 Clinical documentation  
- 🤖 Healthcare automation  
- 🎙️ Voice-based medical systems

  <img width="259" height="326" alt="Screenshot 2026-04-06 110046" src="https://github.com/user-attachments/assets/032fe76c-fe8f-4ed7-9c38-5c2e08ea026c" />

---

## 🔥 Features

- 🎙️ Speech-to-text transcription  
- 👥 Speaker diarization (Doctor / Patient)  
- ✂️ Audio extraction from video  
- 🔄 Chunk-based processing for long audio  
- ⚡ FastAPI backend

- 📄 Structured transcript output  

---

## ⚙️ Tech Stack

- Python  
- FastAPI  
- OpenAI Whisper / Deepgram  
- MoviePy  
- Pydub  

---

## 📦 Setup

1. Install Python 3.9 or higher  

2. Install required packages:

```bash
pip install -r requirements.txt

▶️ How to Use
Run the FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
Open in browser
http://localhost:8000/docs
