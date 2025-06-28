import asyncio
import time
import PIL
from google import genai
from PIL import Image
from google.genai import types
import os
from io import BytesIO
import wave
import numpy as np
import pyaudio
import sounddevice as sd

"""
Run the following commands in the terminal:
pip install pillow
pip install -q -U google-genai
pip install pyaudio

Documentation:
https://ai.google.dev/gemini-api/docs

Generate your own API keys:
https://aistudio.google.com/app/apikey
"""

api_key = ""

class geminiAI():
    def __init__(self, Key):
        self.createOutput()
        self.text = geminiText(Key)
        self.geminImage = geminiImage(Key)
        self.imagen3 = Imagegen(Key)
        self.veo3 = videoGeneration(Key)
        self.singleSpeech = singleSpeech(Key)
        self.multiSpeech = MultiSpeech(Key)
        self.lyria = music(Key)

    def createOutput(self):
        #Forcefully creates an output folder
        try:
            if not os.path.exists("Output"):
                os.makedirs("Output")
        except Exception as e:
            print(f"There was an error: {str(e)}")
    
    def updateKey(self, key):
        #Updates the key
        self.client = genai.Client(api_key=key)

    def updateModel(self, newModel):
       if newModel:
          self.model = newModel

class geminiText(geminiAI):
    #This class handles the Text generation segment of the Gemini documentation
    def __init__(self, key):
        #Declaring and initialising AI variables
        self.client = genai.Client(api_key=key)
        self.model = "gemini-2.0-flash"
        self.response = None
        self.contents = None
        self.systemInstruction = None
        self.maxOutputTokens = 500
        self.temperature = 0.1
        self.chat = None
        self.configs = None
        self.updateConfig()

    def AiResponse(self):
        #Gets a one time response from Gemini
        try:
            if self.contents:
                self.response = self.client.models.generate_content(
                    model=self.model, contents=self.contents, config=self.configs
                )
        except Exception as e:
            print(f"There was an error: {str(e)}")
        
    def AiResponseStream(self):
        #Gets a one time response from Gemini
        try:
            if self.contents:
                self.response = self.client.models.generate_content_stream(
                    model=self.model, contents=self.contents, config=self.configs
                )
        except Exception as e:
            print(f"There was an error: {str(e)}")
            
    def startChat(self):
        #Initiatalises a chat 
        try:
            self.chat = self.client.chats.create(model=self.model, config=self.configs)
        except Exception as e:
            print(f"There was an error: {str(e)}")
        
    def sendChatMessage(self, message):
        #Sends a chat message
        if self.chat:
            self.response = self.chat.send_message(message)
        
    def chatHistory(self):
        #returns the chat history
        if self.chat:
            return self.chat.get_history()
    
    def displayChatHistory(self):
        #Displays the Chat history
        if self.chat:
            for message in self.chat.get_history():
                print(f'role - {message.role}',end=": ")
                print(message.parts[0].text)
            
    def updateTemperature(self, newTemperature):
        #Updates the temperature
        #"temperature" is a parameter that controls the randomness and creativity of the model's output
        if newTemperature > 0.0 and newTemperature < 2.0:
            self.temperature = newTemperature
            self.updateConfig()
        
    def updateMaxOutputTokens(self, newMaxOutputTokens):
        #the maximum number of tokens that an AI model can produce as output in a single response
        if newMaxOutputTokens > 0:
            self.maxOutputTokens = newMaxOutputTokens
            self.updateConfig()
    
    def updateSystemInstruction(self, newSystemInstruction):
        #Updates the role (Character)
        self.systemInstruction = newSystemInstruction
        self.updateConfig()
        
    def updateConfig(self):
        #Updates the config when changed
        self.configs = types.GenerateContentConfig(
            systemInstruction=self.systemInstruction,
            maxOutputTokens=self.maxOutputTokens,
            temperature=self.temperature
        )
            
    def updateContents(self, newContents):
        #Updates the Contents of the message (Prompt)
        self.contents = [newContents]
        
    def openImage(self, Link=""):
        #Returns an Image Open Link
        #To get a response contents must be set to [self.openImage(Link), "Prompt"]
        # i.e. myAI.updateContents([myAI.openImage("Image.png"), "What is this image?"])
        if os.path.isfile(Link):
            return Image.open(Link)
        
    def getResponse(self):
        #returns a full response all at once
        if self.response:
            return self.response.text
        
    def displayResponse(self):
        #Displays a full response all at once
        if self.response:
            print(self.response.text)
        
    def displayChunkResponse(self):
        #displays the response as it arrives
        if self.response:
            for chunk in self.response:
                print(chunk.text, end="")

class geminiImage(geminiAI):
    #This clas handles the Image generation of the Gemini documentation
    # Due to reqgion issues and laws certain functions may not be compatible
    #https://discuss.ai.google.dev/t/gemini-2-0-image-gen-not-found/79966/18

    """
    Choose Gemini when:

    You need contextually relevant images that leverage world knowledge and reasoning.
    Seamlessly blending text and images is important.
    You want accurate visuals embedded within long text sequences.
    You want to edit images conversationally while maintaining context.

    
    """
    def __init__(self, key):
        self.client = genai.Client(api_key=key)
        self.model ="gemini-2.0-flash-preview-image-generation"
        self.response = None
        self.contents = None
        self.image = None

    def getResponse(self):
        #This uses the free gemini 2.0 for Image generation for text-image
        # contents should be stored as (text 1, text 2, text 3, image)
        try:
            if self.image:
                self.contents = [self.contents, self.image]
            if self.contents:
                self.response = self.client.models.generate_content(
                    model=self.model,
                    contents=self.contents,
                    config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                    )
                )
        except Exception as e:
            print(f"There was an error: {str(e)}")

    def updateContents(self, newPrompt):
        #adds new line to content
        if newPrompt:
            if not self.contents:
                self.contents = (newPrompt)
                return
            self.contents = (self.contents, newPrompt)

    def uploadImage(self, path):
        # This loads and image into the content
        # image link should be addded last
        try:
            if os.path.exists(path):
                self.image = PIL.Image.open(path)
        except Exception as e:
            print(f"There was an error: {str(e)}")

    def clearContents(self):
        #clears prompt
        self.contents = None

    def displayTextResponse(self):
        #displays the text portion of the response
        if self.response:
            for part in self.response.candidates[0].content.parts:
                if part.text is not None:
                    print(part.text)
        else:
            return None

    def displayImage(self):
        #displays the image portion of the response
        if self.response:
            for part in self.response.candidates[0].content.parts:
                if part.inline_data is not None:
                    image = Image.open(BytesIO((part.inline_data.data)))
                    image.show()
        else:
            return None
        
    def getTextResponse(self):
        #returns the text portion of the response
        if self.response:
            response = ""
            for part in self.response.candidates[0].content.parts:
                if part.text is not None:
                    response = f"{response} {part.text}"
            return response
        else:
            return None

    def saveImage(self):
        #saves image to output folder
        if self.response:
            for part in self.response.candidates[0].content.parts:
                if part.inline_data is not None:
                    _, _, files = next(os.walk("Output"))
                    file_count = len(files)

                    image = Image.open(BytesIO((part.inline_data.data)))
                    image.save(f'Output/new_image{file_count}.jpg')
        else:
            return None

class Imagegen(geminiAI):
  #This version costs money to use!!!
  """
  Choose Imagen 3 when:

    Image quality, photorealism, artistic detail, or specific styles (e.g., impressionism, anime) are top priorities.
    Performing specialized editing tasks like product background updates or image upscaling.
    Infusing branding, style, or generating logos and product designs.

  """
  def __init__(self, Key):
      self.client = genai.Client(api_key=Key)
      self.model='imagen-3.0-generate-002'
      self.response = None
      self.numberOfImages = 1
      self.aspectRatio = "1:1"
      self.personGeneration = "allow_adult"
      self.contents = None
      self.image = None
      self.config = None
      self.updateConfig()

  def generateImage(self):
    #THis uses imagen3 to generate an image
    try:
        self.response = self.client.models.generate_images(
            model=self.model,
            prompt=self.contents,
            config=self.config
        )
    except Exception as e:
            print(f"There was an error: {str(e)}")

  def updateContents(self, newContents):
    #Updates prompt
    if type(newContents) == str:
        self.contents = newContents

  def updateNumberOfImages(self, NewNumber):
    #Changes the quantity of images generated
    if 1 <= NewNumber <= 4:
      self.numberOfImages = NewNumber
      self.updateConfig()

  def disablePersonGeneration(self):
    #disables generation of people
    self.personGeneration = "dont_allow"
    self.updateConfig()

  def enablePersonGeneration(self):
    #Allows generation of people
    self.personGeneration = "allow_adult"
    self.updateConfig()

  def enableAllPersonGeneration(self):
    #Allows all people to be generated including kids
    #This feature is not allowed in EU, UK, CH, MENA locations.
    self.personGeneration = "allow_all"
    self.updateConfig()

  def updateAspectRatio(self, newAspectRatio):
    #Checks and updates valid aspect ratios for image
    if newAspectRatio in ["1:1", "3:4", "4:3", "9:16", "16:9"]:
      self.aspectRatio = newAspectRatio
      self.updateConfig()

  def updateConfig(self):
      #Updates the config when changed
      self.configs = types.GenerateImagesConfig(
          number_of_images=self.numberOfImages, # int 1-4
          aspect_ratio=self.aspectRatio, # "1:1", "3:4", "4:3", "9:16", "16:9"
          person_generation=self.personGeneration # DONT_ALLOW, ALLOW_ADULT, ALLOW_ALL
      )

  def displayImage(self):
    #If response is generated it displays the image
    if self.response:
      for generated_image in self.response.generated_images:
        image = Image.open(BytesIO(generated_image.image.image_bytes))
        image.show()
    else:
        return None

  def saveImage(self):
    #if response is generated it saves the image
    if self.response:
      _, _, files = next(os.walk("Output"))
      file_count = len(files)
      if self.response.generated_images.generated_images:
        for generated_image in self.response.generated_images:
            img = Image.open(BytesIO(generated_image.image.image_bytes))
            img.save(f'Output/new_image{file_count}.jpg')
    else:
        return None
    
class videoGeneration(geminiAI):
  # This costs money !!!
  def __init__(self, Key):
    self.client = genai.Client(api_key=Key)
    self.model = "veo-2.0-generate-001"
    self.textModel = "gemini-2.0-flash"
    self.contents = None
    self.image = None
    self.negativeContents = None
    self.aspectRatio = "16:9"
    self.personGeneration = "allow_adult"
    self.numberOfVideos = 1
    self.operation = None
    self.durationSeconds = 5
    #self.enhancePrompt = True ### raise ValueError('enhance_prompt parameter is not supported in Gemini API.'), (So why is it in the fucking docs!?!)
    self.waitTime = 20
    self.configs=types.GenerateVideosConfig(
      negativePrompt = self.negativeContents,
      person_generation=self.personGeneration,
      aspect_ratio=self.aspectRatio, 
      number_of_videos = self.numberOfVideos,
      duration_seconds=self.durationSeconds,
      #enhance_prompt=self.enhancePrompt
    )

  def GenerateVideo(self):
    #if base prompt get response
    if self.contents:
      if self.image:
        self.operation = self.client.models.generate_videos(
          model = self.model,
          prompt = self.contents,
          image = self.image,
          config = self.configs
        )
      else:
        self.operation = self.client.models.generate_videos(
          model = self.model,
          prompt = self.contents,
          config = self.configs
        )

      while not self.operation.done:
        self.checkFinished()

  def updateContents(self, newContents):
    #Updates prompt
    if type(newContents) == str:
        self.contents = newContents

  def updateNegativeContents(self, newNegativeContent):
    #Updates negative prompt
    if type(newNegativeContent) == str:
        self.contents = newNegativeContent
        self.updateConfig()

  def uploadImage(self, path):
    # This loads and image into the content
    # image link should be addded last
    try:
      if os.path.exists(path):
        self.response = self.client.models.generate_content(
            model = self.textModel, contents=[Image.open(path), "What is this image?"], config=self.configs
        )
        self.image = [Image.open(path), self.response.text]
    except Exception as e:
      print(f"There was an error: {str(e)}")

  def checkFinished(self):
    time.sleep(self.waitTime)
    self.operation = self.client.operations.get(self.operation)

  def updateAspectRatio(self, newAspectRatio):
    #Checks and updates valid aspect ratios for image
    if newAspectRatio in ["9:16", "16:9"]:
      self.aspectRatio = newAspectRatio
      self.updateConfig()

  def disablePersonGeneration(self):
    #disables generation of people
    self.personGeneration = "dont_allow"
    self.updateConfig()

  def enablePersonGeneration(self):
    #Allows generation of people
    self.personGeneration = "allow_adult"
    self.updateConfig()

  def enableAllPersonGeneration(self):
    #Allows all people to be generated including kids
    #This feature is not allowed in EU, UK, CH, MENA locations.
    self.personGeneration = "allow_all"
    self.image = None
    self.updateConfig()
  # google.genai.errors.ClientError: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'allow_all for personGeneration is currently not supported.', 'status': 'INVALID_ARGUMENT'}}
  # bruv why is nothing supported?

  def updateNumberOfVideos(self, NewNumber):
    #Changes the quantity of images generated
    if 1 <= NewNumber <= 2:
      self.numberOfVideos = NewNumber
      self.updateConfig()

  def updateDuration(self, newNumber):
    # updates the duration of the video generated
    if type(newNumber) == int:
      if 5 <= newNumber <= 8:
        self.durationSeconds = newNumber
        self.updateConfig()

  def updateWaitTime(self, newTime):
    #updates the interval between checking
    if type(newTime) == int:
      if 1 <= newTime:
        self.durationSeconds = newTime
        self.updateConfig()

  """def toggleEnhancePrompt(self, state):
    if type(state) == bool:
      self.enhancePrompt = state"""

  def updateConfig(self):
    #Updates the config when changed
    self.configs=types.GenerateVideosConfig(
      negativePrompt = self.negativeContents,
      person_generation=self.personGeneration,
      aspect_ratio=self.aspectRatio, 
      number_of_videos = self.numberOfVideos,
      duration_seconds=self.durationSeconds,
      #enhance_prompt=self.enhancePrompt
    )
    
  def saveVideo(self):
    #if response is generated save
    if self.operation.response.generated_videos:
      for n, generated_video in enumerate(self.operation.response.generated_videos):
        _, _, files = next(os.walk("Output"))
        file_count = len(files)
        self.client.files.download(file=generated_video.video)
        generated_video.video.save(f"Output/video{file_count}.mp4")  # save the video
    else:
      return None
    
class speech(geminiAI):
  def __init__(self, Key):
    self.client = genai.Client(api_key=Key)
    self.pcm = None
    self.channels = 1
    self.rate = 24000
    self.sampleWidth = 2
    self.voiceNames = ["Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba", "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi", "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat"]
    self.voiceVibe = ["Bright", "Upbeat", "Informative", "Firm", "Excitable", "Youthful", "Firm", "Breezy", "Easy-going", "Bright", "Breathy", "Clear", "Easy-going", "Smooth", "Smooth", "Clear", "Gravelly", "Informative", "Upbeat", "Soft", "Firm", "Even", "Mature", "Forward", "Friendly", "Casual", "Gentle", "Lively", "Knowledgeable", "Warm"]
    self.languageNames =  ["Arabic (Egyptian)", "German (Germany)", "English (US)", "Spanish (US)", "French (France)", "Hindi (India)", "Indonesian (Indonesia)", "Italian (Italy)", "Japanese (Japan)", "Korean (Korea)", "Portuguese (Brazil)", "Russian (Russia)", "Dutch (Netherlands)", "Polish (Poland)", "Thai (Thailand)", "Turkish (Turkey)", "Vietnamese (Vietnam)", "Romanian (Romania)", "Ukrainian (Ukraine)", "Bengali (Bangladesh)", "English (India)", "Marathi (India)", "Tamil (India)", "Telugu (India)"]
    self.languageCodes  = ["ar-EG", "de-DE", "en-US", "es-US", "fr-FR", "hi-IN", "id-ID", "it-IT", "ja-JP","ko-KR", "pt-BR", "ru-RU", "nl-NL", "pl-PL", "th-TH", "tr-TR","vi-VN", "ro-RO", "uk-UA", "bn-BD", "en-IN & hi-IN bundle", "mr-IN", "ta-IN", "te-IN"]

  def saveResponse(self):
    #saves audio file to Output
    if self.response.candidates[0]:
      self.pcm = self.response.candidates[0].content.parts[0].inline_data.data
      _, _, files = next(os.walk("Output"))
      file_count = len(files)

      file_name=f'Output/out{file_count}.wav'
      self.wave_file(file_name)

  def wave_file(self, filename):
    # Set up the wave file to save the output:
    if self.response:
      with wave.open(filename, "wb") as wf:
          wf.setnchannels(self.channels)
          wf.setsampwidth(self.sampleWidth)
          wf.setframerate(self.rate)
          wf.writeframes(self.pcm)

  def streamResponse(self):
        self.stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=self.channels,
        rate=self.rate,
        output=True)
    
  def playAudio(self):
        #getting it to work in the method declared by the docs proved unhelpful
        if self.response.candidates[0].content.parts[0].inline_data.data and self.stream:
            self.stream.write(self.response.candidates[0].content.parts[0].inline_data.data)

  def updateChannels(self, newChannels):
    #Mono or stereo sound
    if  1 <= newChannels <= 2:
      self.channels = newChannels
      self.updateConfigs()

  def updateRate(self, newRate):
    #specifies the sample rate of the audio, measured in Hertz (Hz)
    if newRate in [8000, 16000, 24000, 44100, 48000]:
      self.rate = newRate
      self.updateConfigs()

  def updateSampleWidth(self, newSampleWidth):
    #Number of bits used to store sample
    if newSampleWidth in [1, 2, 3]:
      self.sampleWidth = newSampleWidth
      self.updateConfigs()

  def getVoiceOptions(self):
    #returns supported voice names and vibe
    return self.voiceNames, self.voiceVibe

  def displayVoiceOptions(self):
    #displays supported voice names and vibe
    for index in range(len(self.voiceNames)):
      print(self.voiceNames[index], self.voiceVibe[index])

  def getLanguageOptions(self):
    #returns suppoerted languages and codes
    return self.languageNames, self.languageCodes
  
  def displaylanguageOptions(self):
    #displays supported languages
    for index in range(len(self.languageNames)):
      print(self.languageNames[index], self.languageCodes[index])

class singleSpeech(speech):
  # Experimental feature
  def __init__(self, Key):
    super().__init__(Key)
    self.model = "gemini-2.5-flash-preview-tts",
    self.response = None
    self.contents = None
    self.voice = 'Kore'
    self.language = "en-US"
    self.configs = None
    self.updateConfigs()

  def getResponse(self):
    #uses flash instead od pro
    if self.contents:
      self.response = self.client.models.generate_content(
        model=self.model,
        contents=self.contents,
        config=self.configs
      )

  def updateVoice(self, newVoice):
    #changes the voice selected
    if newVoice in self.voiceNames:
      self.voice = newVoice
      self.updateConfigs()
    else:
      return None
    
  def updateLanguage(self, newLanguage):
    #changes the language recognised
    if newLanguage in self.languageCodes:
      self.language = newLanguage
      self.updateConfigs()
    else:
      return None

  def updateContents(self, newContents):
    #updates the propmt of what the AI will say, some sentences may not work
    #"This is a demonstration of changing the voice and language settings." this statement caused errors as the AI did not recognise it as what i wanted to say
    if type(newContents) == str:
      self.contents = newContents

  def updateConfigs(self):
    #singular voice generation allows for language to be implemented
    self.configs = types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(
            voice_name=self.voice
        )),
      language_code = self.language
    ))

class MultiSpeech(speech):
  # Experimental feature
  def __init__(self, Key):
    super().__init__(Key)
    self.model = "gemini-2.5-flash-preview-tts",
    self.response = None
    self.contents = None
    self.speakerNames = [None, None]
    self.voiceNames = ['Zephyr', 'Kore']
    #self.languages = ["en-US", "en-US"]
    self.voiceCount = 2 # Max is 2
    self.configs = None
    self.updateConfigs()

  def getResponse(self):
    #uses the flash version instead of the pro version
    if self.contents:
      self.response = self.client.models.generate_content(
        model=self.model,
        contents=self.contents,
        config=self.configs
      )
  
  def updateConfigs(self):
    #Multi cast does not support languages or at minimum does not allow for the same method of integration
    self.configs = types.GenerateContentConfig(
      response_modalities=["AUDIO"],
      speech_config=types.SpeechConfig(
         multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
            speaker_voice_configs=[
               types.SpeakerVoiceConfig(
                  speaker=self.speakerNames[0],
                  voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                      voice_name=self.voiceNames[0]
                    ))),
               types.SpeakerVoiceConfig(
                  speaker=self.speakerNames[1],
                  voice_config=types.VoiceConfig(
                     prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.voiceNames[1]
                     ))),
            ]
         )))
    
  def updateSpeakerName(self, pos, newName):
    #This is the role / Name the AI takes on
    if type(newName) == str:
      self.speakerNames[pos] = newName
      self.updateConfigs()
  
  def updateVoiceNames(self, pos, newVoice):
    #This ensures all changes to the voice are legal voices
    if newVoice in self.voiceNames:
      self.voiceNames[pos] = newVoice
      self.updateConfigs()
    
  """def updateLanguage(self, pos, newLanguage):
    if newLanguage in self.languageCodes:
      self.languages[pos] = newLanguage
      self.updateConfigs()
    else:
      return None
  """
  
  def updateContents(self, newPrompt):
    #This a allows users to add sentence by sentence
    #adds new line to content
    # Should follow strucutre:
    """TTS the following conversation between Joe and Jane:
         Joe: How's it going today Jane?
         Jane: Not too bad, how about you?     
    """
    if type(newPrompt) == str:
      if not self.contents:
        self.contents = f"{newPrompt}"
        return
      self.contents = f"{self.contents}, {newPrompt}"

  def clearContents(self):
    #clears prompt
    self.contents = None

class music(geminiAI):
    #Experimental

    #God this sounds fucking terrible
    #please christ have mercy on my enternal soul for producing such a horrific sound
    def __init__(self, Key):
        self.client = genai.Client(api_key=Key, http_options={'api_version': 'v1alpha'})
        self.model = 'models/lyria-realtime-exp'
        self.rate = 24000
        self.prompts = []
        self.contents = None
        self.config = None
        self.channels = 1
        self.weight = 1.0
        self.guidance = 4.0
        self.bpm = 90
        self.density = None
        self.brightness = None
        self.scale = "SCALE_UNSPECIFIED"
        self.scaleValues = ["C_MAJOR_A_MINOR", "D_FLAT_MAJOR_B_FLAT_MINOR", "D_MAJOR_B_MINOR", "E_FLAT_MAJOR_C_MINOR", "E_MAJOR_D_FLAT_MINOR", "F_MAJOR_D_MINOR", "G_FLAT_MAJOR_E_FLAT_MINOR", "G_MAJOR_E_MINOR", "A_FLAT_MAJOR_F_MINOR", "A_MAJOR_G_FLAT_MINOR", "B_FLAT_MAJOR_G_MINOR", "B_MAJOR_A_FLAT_MINOR", "SCALE_UNSPECIFIED"]
        self.scaleKeys = ["C major / A minor", "D♭ major / B♭ minor", "D major / B minor", "E♭ major / C minor", "E major / C♯/D♭ minor", "F major / D minor", "G♭ major / E♭ minor", "G major / E minor", "A♭ major / F minor", "A major / F♯/G♭ minor", "B♭ major / G minor", "B major / G♯/A♭ minor", "Default / The model decides"]
        self.muteBass = False
        self.muteDrums = False
        self.onlyBassAndDrums = False
        self.temperature = 1.1
        self.topK = 40
        self.seed = None
        self.session = None
        self.audioData = None
        self.updateConfig()

        #These are a non-exhaustive list of prompts you can use to prompt Lyria RealTime
        self.instrumentsChoices = [
            "303 Acid Bass", "808 Hip Hop Beat", "Accordion", "Alto Saxophone", "Bagpipes",
            "Balalaika Ensemble", "Banjo", "Bass Clarinet", "Bongos", "Boomy Bass",
            "Bouzouki", "Buchla Synths", "Cello", "Charango", "Clavichord",
            "Conga Drums", "Didgeridoo", "Dirty Synths", "Djembe", "Drumline",
            "Dulcimer", "Fiddle", "Flamenco Guitar", "Funk Drums", "Glockenspiel",
            "Guitar", "Hang Drum", "Harmonica", "Harp", "Harpsichord",
            "Hurdy-gurdy", "Kalimba", "Koto", "Lyre", "Mandolin",
            "Maracas", "Marimba", "Mbira", "Mellotron", "Metallic Twang",
            "Moog Oscillations", "Ocarina", "Persian Tar", "Pipa", "Precision Bass",
            "Ragtime Piano", "Rhodes Piano", "Shamisen", "Shredding Guitar", "Sitar",
            "Slide Guitar", "Smooth Pianos", "Spacey Synths", "Steel Drum", "Synth Pads",
            "Tabla", "TR-909 Drum Machine", "Trumpet", "Tuba", "Vibraphone",
            "Viola Ensemble", "Warm Acoustic Guitar", "Woodwinds"
        ]
        self.musicChoice  = [
            "Acid Jazz", "Afrobeat", "Alternative Country", "Baroque", "Bengal Baul",
            "Bhangra", "Bluegrass", "Blues Rock", "Bossa Nova", "Breakbeat",
            "Celtic Folk", "Chillout", "Chiptune", "Classic Rock", "Contemporary R&B",
            "Cumbia", "Deep House", "Disco Funk", "Drum & Bass", "Dubstep",
            "EDM", "Electro Swing", "Funk Metal", "G-funk", "Garage Rock",
            "Glitch Hop", "Grime", "Hyperpop", "Indian Classical", "Indie Electronic",
            "Indie Folk", "Indie Pop", "Irish Folk", "Jam Band", "Jamaican Dub",
            "Jazz Fusion", "Latin Jazz", "Lo-Fi Hip Hop", "Marching Band", "Merengue",
            "New Jack Swing", "Minimal Techno", "Moombahton", "Neo-Soul", "Orchestral Score",
            "Piano Ballad", "Polka", "Post-Punk", "60s Psychedelic Rock", "Psytrance",
            "R&B", "Reggae", "Reggaeton", "Renaissance Music", "Salsa",
            "Shoegaze", "Ska", "Surf Rock", "Synthpop", "Techno",
            "Trance", "Trap Beat", "Trip Hop", "Vaporwave", "Witch house"
        ]
        self.mood = [
            "Acoustic Instruments", "Ambient", "Bright Tones", "Chill", "Crunchy Distortion",
            "Danceable", "Dreamy", "Echo", "Emotional", "Ethereal Ambience",
            "Experimental", "Fat Beats", "Funky", "Glitchy Effects", "Huge Drop",
            "Live Performance", "Lo-fi", "Ominous Drone", "Psychedelic", "Rich Orchestration",
            "Saturated Tones", "Subdued Melody", "Sustained Chords", "Swirling Phasers", "Tight Groove",
            "Unsettling", "Upbeat", "Virtuoso", "Weird Noises"
        ]

    def getResponse(self):
        #Declares the stream, starts it and then requests the Live music
        self.updateStream()
        asyncio.run(self.genMusic())

    def updateStream(self):
        #Declares stream and starts it
        self.stream = sd.OutputStream(samplerate=self.rate, channels=self.channels, dtype='float32')
        self.stream.start()

    def updateChannels(self, newChannels):
        #Mono or stereo sound
        if  1 <= newChannels <= 2:
            self.channels = newChannels
            self.updateStream()

    def updateRate(self, newRate):
        #specifies the sample rate of the audio, measured in Hertz (Hz)
        if newRate in [8000, 16000, 24000, 44100, 48000]:
            self.rate = newRate
            self.updateStream()

    async def genMusic(self):
        #if prompt exists
        if self.prompts != []:

            async with (
                #creates the session
                self.client.aio.live.music.connect(model=self.model) as self.session,
                asyncio.TaskGroup() as tg,
            ):
                # Set up task to receive server messages.
                audioTask = tg.create_task(self.recieveAudio())

                # Send initial prompts and config
                await self.session.set_weighted_prompts(
                prompts=self.prompts
                )
                await self.session.set_music_generation_config(
                config=self.config
                )
                #plays the session
                await self.session.play()

    async def recieveAudio(self):
       while True:
          #for each message in the session
          async for message in self.session.receive():
            if message.server_content and message.server_content.audio_chunks:
                self.audioData = message.server_content.audio_chunks[0].data
                # Process audio
                await asyncio.sleep(10**-12)

                audio_data_np = np.frombuffer(self.audioData, dtype=np.float32)  
                # Play  audio
                self.stream.write(audio_data_np)

    def updateContents(self, newContents):
        #Updates prompt
        if type(newContents) == str:
            self.contents = newContents

    def updateWeight(self, newWeight):
        #updates the weight
        if type(newWeight) == int:
            self.weight = newWeight

    def addToPrompt(self):
        #addsthe content and weight into a new prompt
        self.prompts.append(types.WeightedPrompt(text=self.contents, weight=self.weight),)

    def clearPrompts(self):
        #resets prompt
        self.prompts = []

    def updateGuidance(self, newGuidance):
        #Controls how strictly the model follows the prompts. Higher guidance improves adherence to the prompt, but makes transitions more abrupt.
        if type(newGuidance) == float:
            if 0 <= newGuidance <= 6:
                self.guidance = newGuidance
                self.updateConfig()

    def updateBpm(self, newBpm):
        #Sets the Beats Per Minute you want for the generated music. You need to stop/play or reset the context for the model it take into account the new bpm.
        if type(newBpm) == int:
            if 60 <= newBpm <= 200:
                self.bpm = newBpm
                self.updateConfig()

    def updateDensity(self, newDensity):
        #Controls the density of musical notes/sounds. Lower values produce sparser music; higher values produce "busier" music
        if type(newDensity) == float:
            if 0 <= newDensity <= 1:
                self.density = newDensity
                self.updateConfig()

    def updateBrightness(self, newBrightness):
        #Adjusts the tonal quality. Higher values produce "brighter" sounding audio, generally emphasizing higher frequencies.
        if type(newBrightness) == float:
            if 0 <= newBrightness <= 1:
                self.density = newBrightness
                self.updateConfig()

    def updateScales(self, newScale):
        # Sets the musical scale (Key and Mode) for the generation. Use the Scale enum values provided by the SDK. You need to stop/play or reset the context for the model it take into account the new scale.
        if newScale in self.scaleValues:
            self.scale = newScale
            self.updateConfig()

    def toggleBass(self):
        #Controls whether the model reduces the outputs' bass.
        self.muteBass = not self.muteBass
        self.updateConfig()

    def toggleDrums(self):
        #Controls whether the model outputs reduces the outputs' drums.
        self.muteDrums = not self.muteDrums
        self.updateConfig()

    def toggleOnlyBassAndDrums(self):
        #Steer the model to try to only output bass and drums.
        self.onlyBassAndDrums = not self.onlyBassAndDrums
        self.updateConfig()

    def updateTemperature(self, newTemperature):
        #a hyperparameter that controls the randomness or creativity of an AI model's output
        if type(newTemperature) == float:
            if 0 <= newTemperature <= 3:
                self.temperature = newTemperature
                self.updateConfig()

    def updateTopK(self, newTopK):
        #a sampling technique used to control the randomness and coherence of generated text
        if type(newTopK) == int:
            if 0 <= newTopK <= 1000:
                self.topK = newTopK
                self.updateConfig()

    def updateSeed(self, newSeed):
        #Controls the randomly generated seed
        if type(newSeed) == int:
            if 0 <= newSeed <=  2147483647:
                self.seed = newSeed
                self.updateConfig()

    def updateConfig(self):
        #For bpm, density, brightness and scale, if no value is provided, the model will decide what's best according to your initial prompts.
        self.config = types.LiveMusicGenerationConfig(
            guidance=self.guidance,
            bpm=self.bpm,
            density=self.density,
            brightness=self.brightness, 
            scale=self.scale, 
            mute_bass=self.muteBass, 
            mute_drums=self.muteDrums, 
            only_bass_and_drums=self.onlyBassAndDrums, 
            temperature=self.temperature, 
            top_k=self.topK, 
            seed=self.seed
        )

    def displayScaleOptions(self):
        #displays scales and their values
        for i in range(len(self.scaleKeys)):
            print(self.scaleKeys, self.scaleValues)

    def displayInstruments(self):
        #displays a list of non comprehensive instruments
        for instrument in self.instrumentsChoices:
            print(instrument)

    def displayMusicChoice(self):
        #displays a list of non comprehensive music choices
        for choice in self.musicChoice:
            print(choice)

    def displayMood(self):
        #displays a list of non comprehensive moods
        for mood in self.mood:
            print(mood)

    def getInstruments(self):
        #returns a list of non comprehensive instruments
        return self.instrumentsChoices

    def getMusicChoice(self):
        #returns a list of non comprehensive music choices
        return self.musicChoice

    def getMood(self):
        #returns a list of non comprehensive moods
        return self.mood

    def play(self):
        #Plays the session
        if self.session:
            self.session.play()

    def pause(self):
        #pauses the session
        if self.session:
            self.session.pause()

    def stop(self):
        #stops the session
        if self.session():
            self.session.stop()

    def reset(self):
        #restarts the session
        if self.session():
            self.session.reset_context()
        
#Object Declaration
myAI = geminiAI(api_key)
