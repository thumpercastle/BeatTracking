import numpy as np


def compute_beat_sequence(novelty, beat_ref, weight=1, return_all=False):
    """
    Implementation from audiolabs-erlangen.de
    This function computes the accumulated score D of all possible beat sequences. It does this
    by comparing them to an estimated beat period, and if the deviation is large enough, the index is discarded and
    the onset is not considered a beat. With a small enough deviation, the penalty is small, the accumulated
    score is large, and the index of that onset is stored as a beat in the P array.
    :param novelty: novelty functions computed using an onset detection function
    :param beat_ref: estimated beat period computed using an automated procedure (akin to an 'educated guess')
    :param weight: the weighting lambda to be applied to penalties
    :param return_all: return all elements B, D and P, or just B.
    :return: B if False, B, D, P if True
    """
    N = len(novelty)
    # Compute penalties for the whole range of possible beat periods
    penalty = compute_penalty(N, beat_ref)
    penalty *= weight
    novelty = np.insert(novelty, 0, 0)
    # Declare accumulated score array
    D = np.zeros(N+1)
    # Declare predecessor array
    P = np.zeros(N+1, dtype=int)
    # Initialise first element to novelty, as no difference information.
    D[1] = novelty[1]
    # Initialise first predecessor to 0
    P[1] = 0

    # forward calculation
    # This loop calculates the accumulated score for each index of D, based on all previous values
    # of D. If the highest value of D + penalty is <= 0, the novelty value alone is assigned to D
    # and 0 is assigned to predecessor (i.e. this is not considered a beat). Otherwise, the novelty
    # plus the maximum of all previous values is assigned to B, and the index is stored in P (i.e.
    # this is considered a beat).
    for n in range(2, N+1):
        m = np.arange(1, n)
        penalty_idx = np.subtract(n, m)
        scores = D[m] + penalty[penalty_idx]
        maxium = np.max(scores)
        if maxium <= 0:
            D[n] = novelty[n]
            P[n] = 0
        else:
            D[n] = novelty[n] + maxium
            P[n] = np.argmax(scores) + 1

    # backtracking
    # Declare array of beat sequence
    B = np.zeros(N, dtype=int)
    k = 0
    # First element in array is the index of the highest-scoring beat from D, i.e. the last beat detected in the
    # forward calculation
    B[k] = np.argmax(D)
    while(P[B[k]] != 0):
        k = k+1
        # The preceding element in P is assigned to the next index of B.
        B[k] = P[B[k-1]]
    # Slice the array to only those elements with indices
    B = B[0:k+1]
    # Reverse the array
    B = B[::-1]
    # Subtract 1 from all elements (as they represent indices) due to Python indexing conventions.
    B = B-1
    if return_all:
        return B, D, P
    else:
        return B

def compute_penalty(N, beat_ref):
    """
    Implementation from audiolabs-erlangen.de
    :param N: The number of onsets being passed to the compute_beat_sequence function.
    :param beat_ref: The estimated beat period.
    :return: penalties for all possible beat periods, 1D Numpy Array
    """
    # t is an array of potential beat periods (ranging from 1 to the distance between first and last frames, N).
    t = np.arange(1, N) / beat_ref
    # Divide t by the reference beat period, log2, square and negative, to compute penalties for all possible
    # beat periods in t.
    penalty = -np.square(np.log2(t))
    # Zero-pad the time array, though this appears redundant.
    t = np.concatenate((np.array([0]), t))
    # Zero-pad penalty array due to Python indexing conventions
    penalty = np.concatenate((np.array([0]), penalty))
    return penalty

def tempo_to_frames(tempo, hopSize, sampling_rate):
    """
    Converts a tempo in BPM to a number of frames
    :param tempo: a tempo in beats per minute (BPM)
    :param hopSize: size of hops between frames (given in number of samples)
    :param sampling_rate: sampling rate of the original signal
    :return: tempo period given in frames
    """
    tempo_in_frames = ((60/tempo) * sampling_rate) / hopSize
    return tempo_in_frames

def tempo_to_samples(tempo, hopSize, sampling_rate):
    """
    Converts a tempo in BPM to a number of samples
    :param tempo: a tempo in beats per minute (BPM)
    :param hopSize: size of hops between frames (given in number of samples)
    :param sampling_rate: sampling rate of the original signal
    :return: tempo period given in samples
    """
    tempo_in_samples = ((60/tempo) * sampling_rate)
    return tempo_in_samples