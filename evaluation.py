import pandas as pd
import numpy as np
import mir_eval
import os


def evaluate(beat_sequence, filepath):

    # Initialise groundTruth array
    groundTruth = np.fromfile(filepath[:-4] + '.beats', sep="\n")

    # Trim first 5 seconds
    groundTruth = mir_eval.beat.trim_beats(groundTruth)
    beat_sequence = mir_eval.beat.trim_beats(beat_sequence)

    # Evaluate
    f_score = mir_eval.beat.f_measure(groundTruth, beat_sequence)
    print(f"f-score: {f_score}")
    #cemgil = mir_eval.beat.cemgil(groundTruth, beat_sequence)
    #print(f"cemgil: {cemgil}")
    #goto = mir_eval.beat.goto(groundTruth, beat_sequence)
    #print(f"goto: {goto}")
    p_score = mir_eval.beat.p_score(groundTruth, beat_sequence)
    print(f"p-score: {p_score}")

    return f_score, p_score