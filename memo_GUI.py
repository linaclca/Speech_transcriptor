import os
import wave
from dataclasses import dataclass

import pyaudio
import time
import threading
import tkinter as tk

class VoiceRecorder:
  def __init__(self):
    self.root = tk.Tk()
    # self.root.title("Voice Recorder")
    # self.root.geometry("300x100")
    self.root.resizable(False, False)

    # self.record_button = tk.Button(self.root, text="Record", command=self.record)
    # self.record_button.pack(pady=10)

    self.button = tk.Button(self.root, text="Record", font=("Arial", 120, "bold"), command=self.click_handler)
    self.button.pack()
    self.label = tk.Label(text="00:00:00", font=("Arial", 20))
    self.label.pack()

    self.recording = False
    self.frames = []

    self.root.mainloop()

  def click_handler(self):
    if self.recording:
      self.recording = False
      self.button.config(fg="black")
    else:
      self.recording = True
      self.button.config(fg="red")
      threading.Thread(target=self.record).start()

  def record(self):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)
    frames = []
    start_time = time.time()

    while self.recording:
      data = stream.read(1024)
      frames.append(data)
      # self.label.config(text=time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

      passed = time.time() - start_time
      secs = passed % 60
      mins = passed // 60
      hours = mins // 60
      self.label.config(text=f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    exists = True
    i = 1
    while exists:
      if os.path.exists(f"input/recording{i}.wav"):
        i += 1
      else:
        exists = False

    sound_file = wave.open(f"input/recording{i}.wav", "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(frames))
    sound_file.close()

VoiceRecorder()