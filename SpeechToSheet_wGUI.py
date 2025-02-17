import os
import sys

print(sys.executable)
import wave
import pyaudio
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import librosa
import numpy as np
import pretty_midi
import parselmouth
import soundfile as sf
from scipy.signal import find_peaks
import fluidsynth
from mido import MidiFile
import subprocess
from music21 import converter, environment
from PIL import Image, ImageTk

# Optionally, set the MuseScore path if needed (adjust as required):
us = environment.UserSettings()
# For example, on many systems the executable is named "mscore" or "MuseScore3":
us['musescoreDirectPNGPath'] = '/Applications/MuseScore 4.app/Contents/MacOS/mscore'  # Change if necessary


# Helper function to convert MIDI to a PNG image of the notation.
def midi_to_png(midi_path, output_png):
  """
  Loads a MIDI file with music21, writes a MusicXML file, then calls MuseScore
  to convert the MusicXML to a PNG image.
  """
  try:
    score = converter.parse(midi_path)
  except Exception as e:
    print(f"Error parsing {midi_path}: {e}")
    return False

  # Write the score as MusicXML (temporary file)
  xml_path = midi_path + ".musicxml"
  try:
    score.write('musicxml', fp=xml_path)
  except Exception as e:
    print(f"Error writing MusicXML for {midi_path}: {e}")
    return False

  # Ensure the output directory exists
  output_dir = os.path.dirname(output_png)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Call MuseScore (or mscore) to convert MusicXML to PNG.
  # The command-line interface of MuseScore typically accepts:
  #   musescore_exe input.musicxml -o output.png
  musescore_exe = us['musescoreDirectPNGPath']  # e.g., 'mscore' or 'MuseScore3'
  try:
    subprocess.run([musescore_exe, xml_path, "-o", output_png], check=True)
  except subprocess.CalledProcessError as e:
    print(f"Error converting {xml_path} to PNG: {e}")
    return False

  # Optionally, you can remove the temporary MusicXML file:
  if os.path.exists(xml_path):
    os.remove(xml_path)
  return True


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

    self.btnRecord = tk.Button(self.root, text="Record", font=("Arial", 120, "bold"), command=self.click_handler)
    self.btnRecord.pack()
    self.label = tk.Label(text="00:00:00", font=("Arial", 20))
    self.label.pack()

    self.recording = False
    self.frames = []

    # Dropdown menu for instruments
    instruments = ['piano', 'guitar', 'strings', 'choir', 'synth']
    self.dropdown_Var = tk.StringVar()
    self.dropdown_Var.set(instruments[0])  # Set default value
    self.instrument = instruments[0]  # Initialize the instrument attribute
    self.instrument_Label = tk.Label(text="Select an instrument")
    self.instrument_Label.pack(anchor=tk.W, pady=10)
    self.dropdownInstrument = tk.OptionMenu(self.root, self.dropdown_Var, *instruments)
    self.dropdownInstrument.pack(anchor=tk.W, pady=10)

    self.btnPlay = tk.Button(self.root, text="Play", font=("Arial", 80, "bold"), command=self.click_handler_play)
    self.btnPlay.pack(anchor=tk.W, padx=10)

    # NEW: Button to display sheet music notation.
    self.btnShowSheet = tk.Button(self.root, text="Show Sheet Music", font=("Arial", 80, "bold"),
                                  command=self.show_sheet_music)
    self.btnShowSheet.pack(anchor=tk.W, padx=10)

    self.playing = False

    self.dropdown_Var.trace('w', self.loadInstrument)

    self.root.mainloop()

  def loadInstrument(self, *args):
    self.instrument = self.dropdown_Var.get()  # Get the selected instrument
    print(f"Selected instrument: {self.instrument}")  # Debug print

    # Instrument sf
    if self.instrument == "piano":
      SOUNDFONT_PATH_MELODY = "sf2/Yamaha-Grand-Lite-v2.0.sf2"  # Piano SoundFont
      bank = 0
      preset = 1
    elif self.instrument == "guitar":
      SOUNDFONT_PATH_MELODY = "sf2/module90.sf2"
      bank = 0
      preset = 22
    elif self.instrument == "strings":
      SOUNDFONT_PATH_MELODY = "sf2/module90.sf2"
      bank = 0
      preset = 38
    elif self.instrument == "choir":
      SOUNDFONT_PATH_MELODY = "sf2/KBH-Real-Choir-V2.5.sf2"
      bank = 0
      preset = 1
    elif self.instrument == "synth":
      SOUNDFONT_PATH_MELODY = "sf2/module90.sf2"
      bank = 0
      preset = 0

    # Initialize FluidSynth
    self.synth = fluidsynth.Synth()
    self.synth.start(driver="coreaudio")

    # Load SoundFonts
    self.sfid_drums = self.synth.sfload("sf2/PNS_Drum_Kit.sf2")  # Path to your drum SoundFont
    self.sfid_melody = self.synth.sfload(SOUNDFONT_PATH_MELODY)  # Path to your melody SoundFont
    self.synth.program_select(0, self.sfid_drums, 0, 0)  # Channel 0, Bank 0, Preset 0
    self.synth.program_select(1, self.sfid_melody, bank, preset)  # Channel 1, Bank, Preset 0

  def click_handler(self):
    if self.recording:
      self.recording = False
      self.btnRecord.config(fg="black")
    else:
      self.recording = True
      self.btnRecord.config(fg="red")
      threading.Thread(target=self.record).start()

  def click_handler_play(self):
    if self.playing:
      self.playing = False
      self.btnPlay.config(fg="black")
    else:
      self.playing = True
      self.btnPlay.config(fg="green")
      threading.Thread(target=self.play_midi_files).start()

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

  def play_midi_files(self):
    midi1_path = 'output/speech_to_drums.mid'
    midi2_path = 'output/output_voice_to_melody.mid'
    print(self.instrument)

    # Create threads for each MIDI file
    thread1 = threading.Thread(target=play_midi_wrapper, args=(self.synth, midi1_path, 0))
    thread2 = threading.Thread(target=play_midi_wrapper, args=(self.synth, midi2_path, 1))

    # Start threads
    thread1.start()
    thread2.start()

    # Wait for threads to finish
    thread1.join()
    thread2.join()

  # NEW: Method to display sheet music notation for both MIDI files.
  def show_sheet_music(self):
    # Define the MIDI file paths (as generated by your process)
    drum_midi = 'output/speech_to_drums.mid'
    melody_midi = 'output/output_voice_to_melody.mid'

    # Check that the files exist
    if not os.path.exists(drum_midi) or not os.path.exists(melody_midi):
      messagebox.showerror("Error", "MIDI files not found. Please record and process audio first.")
      return

    # Define output PNG file paths
    drum_png = 'output/sheet_drum.png'
    melody_png = 'output/sheet_melody.png'

    # Convert the MIDI files to PNG sheet music images.
    drum_success = midi_to_png(drum_midi, drum_png)
    melody_success = midi_to_png(melody_midi, melody_png)

    if not (drum_success and melody_success):
      messagebox.showerror("Conversion Error", "There was an error converting MIDI to sheet music.")
      return

    # Create a new Toplevel window to display the sheet music
    sheet_window = tk.Toplevel(self.root)
    sheet_window.title("Music Notation Sheets")

    # Load the images using PIL and convert to Tkinter-compatible PhotoImage objects
    try:
      drum_img = Image.open(drum_png)
      melody_img = Image.open(melody_png)
    except Exception as e:
      messagebox.showerror("Image Error", f"Error loading sheet images: {e}")
      return

    drum_photo = ImageTk.PhotoImage(drum_img)
    melody_photo = ImageTk.PhotoImage(melody_img)

    # Display the drum notation at the top
    drum_label = tk.Label(sheet_window, text="Drum Notation", font=("Arial", 24))
    drum_label.pack(pady=(10, 0))
    drum_img_label = tk.Label(sheet_window, image=drum_photo)
    drum_img_label.image = drum_photo  # keep a reference
    drum_img_label.pack(pady=(0, 20))

    # Display the melody notation below
    melody_label = tk.Label(sheet_window, text="Melody Notation", font=("Arial", 24))
    melody_label.pack(pady=(10, 0))
    melody_img_label = tk.Label(sheet_window, image=melody_photo)
    melody_img_label.image = melody_photo
    melody_img_label.pack(pady=(0, 20))


def play_midi(synth, midi_file, channel):
  start_time = time.time()
  for msg in midi_file:
    if not msg.is_meta:
      msg = msg.copy(channel=channel)
      current_time = time.time() - start_time
      if msg.type == 'note_on':
        synth.noteon(channel, msg.note, msg.velocity)
      elif msg.type == 'note_off':
        synth.noteoff(channel, msg.note)
      time.sleep(max(0, msg.time - (time.time() - start_time - current_time)))


def play_midi_wrapper(synth, midi_file_path, channel):
  midi_file = MidiFile(midi_file_path)
  play_midi(synth, midi_file, channel)


VoiceRecorder()