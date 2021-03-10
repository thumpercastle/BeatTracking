from onsetDetection import complexNoveltyFunction, medfilt, normalise
import librosa
import numpy as np
from beatTracking import tempo_to_frames, compute_beat_sequence


def beatTracker(inputFile, return_all=False):

    x, sr = librosa.load(inputFile, sr=None)

    # Pass signal through a bandstop filter (EXCLUDED AS DOESN'T WORK)
    # filteredSignal = bandstopFilter(x, lf_cutoff=500, hf_cutoff=11000)

    # Compute complex novelty function
    novelties, novelties2d, hop_length, hop_time = complexNoveltyFunction(x=x, sampling_rate=sr)
    novelty_mag = np.abs(novelties)

    # Pass onsets through a median filter (borrowed from QMUL code)
    onsets = medfilt(novelty_mag, 11)
    # Normalise magnitude of onsets
    onsets = normalise(onsets)

    # Compute estimated tempo and convert to frames
    estimated_tempo = librosa.beat.tempo(x, sr=sr)
    print(f"estimated tempo: {estimated_tempo}")
    # Convert estimated tempo into a reference beat period
    beat_ref = tempo_to_frames(estimated_tempo, hop_length, sr)

    # Compute optimal beat sequence using method from Muller
    optimalBeatSequence = compute_beat_sequence(novelty=onsets, beat_ref=beat_ref, weight=0.1, return_all=False)

    # Convert hops to seconds
    optimalBeatSequence = np.multiply(optimalBeatSequence, hop_time)

    if return_all:
        return optimalBeatSequence, x, sr, hop_length
    else:
        return optimalBeatSequence