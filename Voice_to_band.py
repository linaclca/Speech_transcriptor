import librosa
import numpy as np
import pretty_midi
import parselmouth
import soundfile as sf
from scipy.signal import find_peaks

### Parameters ###
AUDIO_PATH = 'input/A0101B.wav'
SR = 22050  # Standard sampling rate
HOP_LENGTH = 512
FRAME_LENGTH = 2048
OUTPUT_PATH_MELODY = 'output/output_voice_to_melody.mid'
OUTPUT_PATH_DRUM = 'output/speech_to_drums.mid'

### Voice to drums ###

# Load audio
y, sr = librosa.load(AUDIO_PATH, sr=SR)
# Load the audio using Parselmouth

# 1. Extract Pitch and Dynamics
pitch, voiced_flag, voiced_probs = librosa.pyin(y, fmin=50, fmax=2000, sr=sr, hop_length=HOP_LENGTH)
rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

# 2. Onset Detection
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP_LENGTH, backtrack=True)
onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=HOP_LENGTH)

# 3. Spectral Features (Formants)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=HOP_LENGTH)[0]

# 4. Classify Events (Speech-to-Drum Mapping)
events = []
for onset_frame in onset_frames:
    # Get features for the current onset
    idx = min(onset_frame, len(rms) - 1)
    loudness = rms[idx]
    centroid = spectral_centroid[idx]
    bandwidth = spectral_bandwidth[idx]
    pitch_value = pitch[idx] if pitch[idx] is not None else 0

    # Classify based on features - not perfect yet
    if pitch_value < 200 and centroid < 500:
        drum_type = 'Kick'  # Low pitch and low centroid
    elif pitch_value < 500 and centroid < 2000:
        drum_type = 'Snare'  # Midi pitch and centroid
    elif centroid > 2000 and bandwidth > 1500:
        drum_type = 'Hi-Hat'  # High centroid and bandwidth
    else:
        drum_type = 'Cymbal'  # Broad or unclear spectral features

    # Store event info
    event_time = librosa.frames_to_time(onset_frame, sr=sr, hop_length=HOP_LENGTH)
    events.append((event_time, drum_type, loudness, centroid, pitch_value))


# 5. Create MIDI from Detected Events
def create_midi(events, output_file):
    # Initialize PrettyMIDI object
    pm = pretty_midi.PrettyMIDI()

    # Create a drum instrument
    drum_program = 0  # Percussion
    drum_track = pretty_midi.Instrument(program=drum_program, is_drum=True)

    # Map drum types to MIDI notes
    drum_map = {
        'Kick': 36,
        'Snare': 38,
        'Hi-Hat': 42,
        'Cymbal': 49
    }

    # Add detected events as MIDI notes
    for event_time, drum_type, loudness, _, _ in events:
        midi_note = drum_map.get(drum_type, 36)  # Default to kick if unknown
        note_on = event_time
        note_off = event_time + 0.1  # Short note duration
        velocity = min(int(loudness * 127 / np.max(rms)), 127)  # Scale loudness to MIDI velocity
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=midi_note,
            start=note_on,
            end=note_off
        )
        drum_track.notes.append(note)

    # Add the drum track to the MIDI object
    pm.instruments.append(drum_track)

    # Write the MIDI file
    pm.write(output_file)
    print(f"MIDI file saved as {output_file}")

# Export detected events to MIDI
create_midi(events, OUTPUT_PATH_DRUM) # drop into DAW

# Display results
print("Detected drum events:")
for event in events:
    print(
        f"Time: {event[0]:.3f}s, Type: {event[1]}, Loudness: {event[2]:.3f}, Centroid: {event[3]:.2f}, Pitch: {event[4]:.2f}")



### Voice to melody ###

# Load the audio using Parselmouth
snd = parselmouth.Sound(AUDIO_PATH)
# Extract the pitch using Parselmouth
pitch_par = snd.to_pitch(pitch_floor=50, pitch_ceiling=800)

# Extract pitch values with custom pitch range
pitch_values = pitch_par.selected_array['frequency']

# Frame duration based on the time step in pitch
time_step = 0.01 #parselmouth.TimeFrameSampled.get_time_step(snd)

i = 0 
frequency = []
previous_freq = None
melody = []

for freq in pitch_values:
    #print(previous_freq)
    if freq > 0:
        frequency.append(freq)
        i = i + 1     
    elif freq == 0 and previous_freq is not None and previous_freq > 0 :
        note = np.median(frequency) 
        melody_segment = np.ones(i) * note
        melody = np.concatenate((melody, melody_segment))
        frequency = []  # Reset frequency list for the next segment
        i = 0
        melody = np.concatenate((melody, [0])) 
    else: 
        melody = np.concatenate((melody, [0]))

    previous_freq = freq
    

melody[melody == 0] = np.nan  # Replace 0 with NaN for better handling

# Function to quantize a frequency to the nearest MIDI note
def frequency_to_midi(frequency):
    """
    Convert a frequency (Hz) to the nearest MIDI note number.
    """
    if np.isnan(frequency):
        return None  # No pitch
    return int(np.round(69 + 12 * np.log2(frequency / 440.0)))

# MIDI track setup
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

current_midi_note = None
start_time = None  # To store the start time of the current note

for i, freq in enumerate(melody):
    # Convert frequency to MIDI note
    midi_note = frequency_to_midi(freq)
    current_time = i * time_step  # Current frame time

    if midi_note is not None:
        if current_midi_note is None:
            # Start a new note if no active note
            current_midi_note = midi_note
            start_time = current_time
        elif midi_note != current_midi_note:
            # End the current note and start a new one
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
            # End the current note if a break occurs
            note = pretty_midi.Note(
                velocity=100,
                pitch=current_midi_note,
                start=start_time,
                end=current_time,
            )
            instrument.notes.append(note)
            current_midi_note = None  # Reset the current note

# Handle any lingering note at the end of the melody
if current_midi_note is not None:
    note = pretty_midi.Note(
        velocity=100,
        pitch=current_midi_note,
        start=start_time,
        end=len(melody) * time_step,
    )
    instrument.notes.append(note)

# Add the instrument to the MIDI file
midi.instruments.append(instrument)

# Write the MIDI file
midi.write(OUTPUT_PATH_MELODY)

#print(f"Synthesized pitch contour saved to {OUTPUT_WAV_PATH}")
print(f"MIDI pitch contour saved to {OUTPUT_PATH_MELODY}")

