## Gemini API Class
## Introduction
"What one generation tolerates, the next generation will embrace." — John Wesley
This repository aims to simplify integration with Google's Gemini API across text, image, audio, video, and music generation use cases.

## Installation Prerequisites
## General Requirements
- Ensure Python is installed: Download Python
- Set up your workspace and open it in VSCode or another preferred code editor.
- Open the terminal and run:
 ` ` `pip install pillow
pip install -q -U google-genai
pip install pyaudio sounddevice ` ` `


## Gemini API Key Setup
- Visit the Google Gemini API documentation.
- Create a project at the Google Cloud Console.
- From the dashboard, select Create API Key, then assign it to your Project.
- Copy your API key for use in the class instantiation:
 ` ` `myAI = geminiAI(your_api_key) ` ` `



## Getting Started
To run the class, execute geminiAPI.py.
Note: A Gemini API key is required to enable AI-generated responses.
The class is modular, allowing selective integration of specific components based on your use case.


## Current Iteration
This initial release focuses on fast, replicable Gemini API integration. It supports:
• Text generation (gemini-2.0-flash)
• Image generation (gemini-2.0-flash-preview-image-generation)
• Imagen generation (imagen-3.0-generate-002)
• Video generation (veo-2.0-generate-001)
• Individual speech generation (gemini-2.5-flash-preview-tts)
• Multi-voice speech generation (gemini-2.5-flash-preview-tts)
• Music generation via Lyria (models/lyria-realtime-exp)


## Future Developments
Planned features for upcoming iterations:
• Context caching
• Thought-chain emulation
• Function calling
• Document understanding
• Image understanding
• Video understanding
• Audio understanding
• Code execution
• URL context awareness
• Integrated Google Search


## Citation and Terms of Use
- License: CC BY
This license permits distribution, remixing, and commercial use with attribution.
- Restrictions:
-   Users may not use automated systems (e.g., scrapers, AI models) to extract or analyze this repository or its data.
- Acceptance:
-   By downloading, installing, or using this software, you acknowledge and accept these terms.
