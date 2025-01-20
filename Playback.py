import mido
from mido import MidiFile
from collections import namedtuple
import fluidsynth
import multiprocessing


# File paths
DRUM_MIDI_PATH = "speech_to_drums.mid"  # Path to your drum MIDI file
MELODY_MIDI_PATH = "output_voice_to_melody.mid"  # Path to your melody MIDI file
SOUNDFONT_PATH_DRUMS = "sf2/PNS_Drum_Kit.sf2"  # Path to your SoundFont
SOUNDFONT_PATH_MELODY = "sf2/Yamaha-Grand-Lite-v2.0.sf2"


def play_midi(midi_path, soundfont_path, bank, preset,channel):
    # Initialize FluidSynth
    fs = fluidsynth.Synth()
    fs.start()

    # Load SoundFont
    sfid = fs.sfload(soundfont_path)
    if sfid == -1:
        raise RuntimeError(f"Failed to load SoundFont: {soundfont_path}")

    fs.program_select(channel, sfid, bank, preset)

    # Parse and play the MIDI file
    midi_file = mido.MidiFile(midi_path)
    for message in midi_file.play():
        if message.type == "note_on" and message.velocity > 0:
            fs.noteon(message.channel, message.note, message.velocity)
        elif message.type in ("note_off", "note_on") and message.velocity == 0:
            fs.noteoff(message.channel, message.note)

    # Cleanup
    fs.delete()
    print(f"Playback finished for {midi_path}")

def play_drums_and_melody(drum_midi_path, melody_midi_path, drum_soundfont, melody_soundfont,bank):
    # Create processes for drums and melody
    drum_process = multiprocessing.Process(
        target=play_midi, 
        args=(drum_midi_path, drum_soundfont, 128, 0,9)  # Adjust bank and preset for drums
    )
    melody_process = multiprocessing.Process(
        target=play_midi, 
        args=(melody_midi_path, melody_soundfont, 0, bank,0) # Adjust bank and preset for melody
    )

    # Start processes
    drum_process.start()
    melody_process.start()

    # Wait for processes to finish
    drum_process.join()
    melody_process.join()

if __name__ == "__main__":
    print("Choose an instrument: piano, guitar, strings, choir, synth")
    instrument = input("Enter instrument: ").strip().lower()

    if instrument == "piano":
        SOUNDFONT_PATH_MELODY = "sf2/Yamaha-Grand-Lite-v2.0.sf2"    # Piano SoundFont
        bank = 1
    elif instrument == "guitar":
        SOUNDFONT_PATH_MELODY = "sf2/module90.sf2"
        bank = 71
    elif instrument == "strings":
        SOUNDFONT_PATH_MELODY = "sf2/module90.sf2"
        bank = 38
    elif instrument == "choir":
        SOUNDFONT_PATH_MELODY = "sf2/KBH-Real-Choir-V2.5.sf2"
        bank = 1
    elif instrument == "synth":
        SOUNDFONT_PATH_MELODY = "sf2/module90.sf2"
        bank = 0

    # Play drums and melody concurrently
    play_drums_and_melody(DRUM_MIDI_PATH, MELODY_MIDI_PATH, SOUNDFONT_PATH_DRUMS, SOUNDFONT_PATH_MELODY,bank)


