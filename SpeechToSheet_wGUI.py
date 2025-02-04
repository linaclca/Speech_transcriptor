import os
import sys
print(sys.executable)
import wave
import pyaudio
import time
import threading
import tkinter as tk
import librosa
import numpy as np
import pretty_midi
import parselmouth
# from urllib.parse import quote
import soundfile as sf
from scipy.signal import find_peaks

### Simple Recorder UI/Widget
#
# (1) record a recording.wav file and save it to folder input/
# (2) rhythm
# (3) melody
# (4) buffer
class VoiceRecorder:
  def __init__(self):
    self.root = tk.Tk()
    self.root.resizable(False, False)

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

      passed = time.time() - start_time
      secs = passed % 60
      mins = passed // 60
      hours = mins // 60
      self.label.config(text=f"{int(hours):02d}:{int(mins):02d}:{int(secs):02d}")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    if not os.path.exists('input'):
      os.makedirs('input')
    sound_file = wave.open('input/recording.wav', 'wb')
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b"".join(frames))
    sound_file.close()

    self.process_recording()

  ### (2) recording to drums
  def process_recording(self):
    AUDIO_PATH = 'input/recording.wav'
    SR = 22050
    HOP_LENGTH = 512
    FRAME_LENGTH = 2048
    OUTPUT_PATH_MELODY = 'output/output_voice_to_melody.mid'
    OUTPUT_PATH_DRUM = 'output/speech_to_drums.mid'

    y, sr = librosa.load(AUDIO_PATH, sr=SR)
    pitch, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]

    events = []
    for onset_frame in onset_frames:
      idx = min(onset_frame, len(rms) - 1)
      loudness = rms[idx]
      centroid = spectral_centroid[idx]
      bandwidth = spectral_bandwidth[idx]
      pitch_value = pitch[idx] if pitch[idx] is not None else 0

      if pitch_value < 200 and centroid < 500:
        drum_type = 'Kick'
      elif pitch_value < 500 and centroid < 2000:
        drum_type = 'Snare'
      elif centroid > 2000 and bandwidth > 1500:
        drum_type = 'Hi-Hat'
      else:
        drum_type = 'Cymbal'

      event_time = librosa.frames_to_time(onset_frame, sr=sr, hop_length=HOP_LENGTH)
      events.append((event_time, drum_type, loudness, centroid, pitch_value))

    self.create_midi(events, OUTPUT_PATH_DRUM, rms)

    snd = parselmouth.Sound(AUDIO_PATH)
    pitch_par = snd.to_pitch(pitch_floor=50, pitch_ceiling=800)
    pitch_values = pitch_par.selected_array['frequency']
    time_step = 0.01

    i = 0
    frequency = []
    previous_freq = None
    melody = []

    for freq in pitch_values:
      if freq > 0:
        frequency.append(freq)
        i += 1
      elif freq == 0 and previous_freq is not None and previous_freq > 0:
        note = np.median(frequency)
        melody_segment = np.ones(i) * note
        melody = np.concatenate((melody, melody_segment))
        frequency = []
        i = 0
        melody = np.concatenate((melody, [0]))
      else:
        melody = np.concatenate((melody, [0]))

      previous_freq = freq

    melody[melody == 0] = np.nan

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

    current_midi_note = None
    start_time = None

    for i, freq in enumerate(melody):
      midi_note = self.frequency_to_midi(freq)
      current_time = i * time_step

      if midi_note is not None:
        if current_midi_note is None:
          current_midi_note = midi_note
          start_time = current_time
        elif midi_note != current_midi_note:
          note = pretty_midi.Note(
            velocity=100,
            pitch=current_midi_note,
            start=start_time,
            end=current_time,
          )
          instrument.notes.append(note)
          current_midi_note = midi_note
          start_time = current_time
      else:
        if current_midi_note is not None:
          note = pretty_midi.Note(
            velocity=100,
            pitch=current_midi_note,
            start=start_time,
            end=current_time,
          )
          instrument.notes.append(note)
          current_midi_note = None

    if current_midi_note is not None:
      note = pretty_midi.Note(
        velocity=100,
        pitch=current_midi_note,
        start=start_time,
        end=len(melody) * time_step,
      )
      instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(OUTPUT_PATH_MELODY)

    self.save_input_buffer()
    print(f"Melody MIDI file saved as {OUTPUT_PATH_MELODY}")

  ### (3) recording to melody
  def create_midi(self, events, output_file, rms):
    pm = pretty_midi.PrettyMIDI()
    drum_program = 0
    drum_track = pretty_midi.Instrument(program=drum_program, is_drum=True)

    drum_map = {
      'Kick': 36,
      'Snare': 38,
      'Hi-Hat': 42,
      'Cymbal': 49
    }

    for event_time, drum_type, loudness, _, _ in events:
      midi_note = drum_map.get(drum_type, 36)
      note_on = event_time
      note_off = event_time + 0.1
      velocity = min(int(loudness * 127 / np.max(rms)), 127)
      note = pretty_midi.Note(
        velocity=velocity,
        pitch=midi_note,
        start=note_on,
        end=note_off
      )
      drum_track.notes.append(note)

    pm.instruments.append(drum_track)
    pm.write(output_file)
    print(f"MIDI file saved as {output_file}")

  def frequency_to_midi(self, frequency):
    if np.isnan(frequency):
      return None
    return int(np.round(69 + 12 * np.log2(frequency / 440.0)))

  ### (4) save initial input audio to output/
  def save_input_buffer(self):
    exists = True
    i = 1
    if not os.path.exists('output'):
      os.makedirs('output')
    while exists:
      if os.path.exists(f"output/recording{i}.wav"):
        i += 1
      else:
        exists = False

    output_path = f"output/recording{i}.wav"
    os.rename('input/recording.wav', output_path)
    print(f"Input buffer saved to {output_path}")

VoiceRecorder()