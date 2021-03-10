import numpy as np
import librosa, librosa.display



def complexNoveltyFunction(x, sampling_rate, win_time=0.04, hop_time=0.01):
    """
    Compute a complex novelty function for a given audio sample.
    Uses the method described in Fundamentals of Music Processing by Mueller, 2015.
    :param x: audio signal provided as amplitude per sample
    :param sampling_rate: sampling rate of the signal x
    :param win_time: window time to use in computation of the STFT, given in seconds
    :param hop_time: hop time to use in computation of the STFT, given in seconds
    :return:
    """
    # Step 0: Initialise parameters, convert into useful indices
    win_length = int(2 ** np.ceil(np.log2(win_time * sampling_rate)))       # Window length in frames
    hop_length = round(hop_time * sampling_rate)       # Hope length in frames

    # Step 1: STFT of signal
    X = librosa.stft(x, n_fft=2048, hop_length=hop_length, win_length=win_length, window='hann',
                     center=True)
    Xphase = np.unwrap(np.angle(X))
    Xmag = np.abs(X)

    # Step 2: Compute phase differences between frames (formula 6.11 from Muller), and wrap them
    phaseDifference = np.subtract(Xphase[1:], Xphase[:-1])                  # Compute phase differences
    phaseDifference = (phaseDifference + np.pi) % (2 % np.pi) - np.pi       # Wrap phase

    # Step 3: Compute the steady-state estimate Xhat(n+1,k) for next frames (formula 6.17)
    steadyStateEst = X[:-1] * np.exp(np.pi * 2 * 1j * (Xphase[:-1] + phaseDifference))

    # Step 4: Compute the novelty between estimates and actual fourier coefficients (formula 6.18)
    novelties2d = np.abs(steadyStateEst, X[1:])     # 2D array of novelties in time and frequency domain

    # Step 5: Find only increasing magnitude (formula 6.19)
    XmagFlattened = np.sum(Xmag, axis=0)        # Flatten array into time domain
    decreasingMagnitudes = np.less_equal(XmagFlattened[1:], XmagFlattened[:-1])       # Find indices of decreasing magnitudes
    decreasingMagnitudes = decreasingMagnitudes * 1                       # Convert bool to int
    decreasingMagnitudes = np.flatnonzero(decreasingMagnitudes)             # Find indices of increasing magnitudes

    # Step 6: Compute complex-domain novelty function by summing over all frequency coefficients (6.21)
    complexNovelties = np.sum(novelties2d, axis=0)
    complexNovelties[decreasingMagnitudes] = 0          # Assign 0 to all time indices where magnitude decreasing

    return complexNovelties, novelties2d, hop_length, hop_time


def energy_based_novelty(signal, windowTime, hopTime, samplingRate):
    """
    Compute time-domain onset function using the method described in Mueller (2015).
    AKA energy-based novelty.
    This function is not used in the beat-tracking procedure
    :param signal: signal amplitudes over time
    :param windowTime: window time to use in the STFT, given in seconds
    :param hopTime: hop time to use in the STFT, given in seconds
    :param samplingRate: sampling rate of the signal
    :return: energy-based novelty
    """
    hopSize = round(hopTime * samplingRate)

    # Round window length up to the nearest power of 2
    windowLength = int(2 ** np.ceil(np.log2(windowTime * samplingRate)))
    # Zero-pad the start and end of the signal
    signalPadded = np.concatenate([np.zeros(windowLength//2),signal,np.zeros(windowLength//2)])

    # Initialise frame and window arrays
    # frame = np.zeros(windowLength)
    # windowCount = (signalPadded/windowLength) + 1
    window = np.hamming(windowLength)

    # Compute local energy function
    windowIndices_m = np.arange(0, windowLength, 1)

    # Initialise summation array
    timeDomainOnset = np.zeros(signal.size)

    # Compute local energy function
    for n in range(0, signal.size, hopSize):
        xmnIndices = windowIndices_m + n
        #print(f"n: {n}")
        #print(f"xmnIndices: {xmnIndices}")
        xmn = np.take(signalPadded, xmnIndices)
        timeDomainOnset[n] = (1/windowLength + 1) * np.sum(np.matmul((xmn**2), window))

    # # Compute energy-based novelty function by finding differential between subsequent samples and ignoring negative values
    # # Find difference between subsequent frames
    energyBasedNovelty = np.delete(timeDomainOnset, timeDomainOnset.size-1, axis=0) - np.delete(timeDomainOnset, 0, axis=0)
    # # Find indices of positive energy differentials
    energyDiffNonZeroBool = energyBasedNovelty >= 0
    energyDiffNonZero = energyBasedNovelty[np.nonzero(energyDiffNonZeroBool)]

    # TODO: Simulate human perception of change in loudness using Just Noticeable Difference JND
    # jnd = np.zeros(timeDomainOnset.size-1)
    # jnd = energyBasedNovelty * (1/np.delete(timeDomainOnset, timeDomainOnset.size-1, axis=0))

    # Scale time domain function down
    scaledTDodf = timeDomainOnset * (1/np.max(timeDomainOnset))

    # Output
    energyBasedNovelty = scaledTDodf

    return energyBasedNovelty


def log_compression(X, gamma):
    Xmag = np.log10(1+gamma*abs(X))
    return Xmag


def bandstopFilter(x, lf_cutoff=100, hf_cutoff=6000, n_fft=2048, sampling_rate=22050, win_time=0.04, hop_time=0.01):
    """
    Apply a band stop filter between the lf_cutoff and the hf_cutoff frequencies.
    Low energy occurs on beats and high energy occurs on onsets.
    This function is not used in the beat-tracking procedure.
    :param novelties_stft: the 2D numpy array of the STFT of the original signal
    :param lf_cutoff: the lower cutoff frequency of the stopband
    :param hf_cutoff: the upper cutoff frequency of the stopband
    :param n_fft: the size of the STFT
    :param sampling_rate: the audio sampling rate (librosa default = 22050)
    :return: filtSigMagnitude = band stop filtered onsets (1D numpy array)
    """
    # Initialise parameters, convert into useful indices
    win_length = int(2 ** np.ceil(np.log2(win_time * sampling_rate)))       # Window length in frames
    hop_length = round(hop_time * sampling_rate)       # Hope length in frames

    # STFT of signal
    X = librosa.stft(x, n_fft=2048, hop_length=hop_length, win_length=win_length, window='hann',
                     center=True)

    nyquist = sampling_rate / 2
    freqBinWidth = nyquist / (1 + n_fft/2)
    freqBinCentres = np.around(np.arange(freqBinWidth/2, nyquist, freqBinWidth), decimals=1)
    stopbandBinLowerBound = np.flatnonzero(np.greater(freqBinCentres, lf_cutoff))        # Find indices of bins over the lf_cutoff frequency
    stopbandBinUpperBound = np.flatnonzero(np.less(freqBinCentres, hf_cutoff))          # Find indices of bins under the hf_cutoff
    stopbandFirstIdx = stopbandBinLowerBound[0]
    stopbandLastIdx = stopbandBinUpperBound[-1]
    filteredSignal = X
    filteredSignal[stopbandFirstIdx:stopbandLastIdx] = 0
    filtSigMagnitude = np.abs(np.sum(filteredSignal, axis=0))

    return filtSigMagnitude


def medfilt(x, k):
   """Apply a length-k median filter to a 1D array x.
   Boundaries are extended by repeating endpoints.
   BORROWED FROM QMUL LAB SESSIONS
   """
   # assert k % 2 == 1, "Median filter length must be odd."
   if x.ndim > 1: # Input must be one-dimensional."
      y = []  #np.empty((0,100), float)
      xt = np.transpose(x)
      for i in xt:
         y.append(medfilt(i, k))
      return np.transpose(y)
   k2 = (k - 1) // 2
   y = np.zeros((len(x), k), dtype=x.dtype)
   y[:,k2] = x
   for i in range(k2):
      j = k2 - i
      y[j:,i] = x[:-j]
      y[:j,i] = x[0]
      y[:-j,-(i+1)] = x[j:]
      y[-j:,-(i+1)] = x[-1]
   return np.median(y, axis=1)


def upsample(data, target_signal, hop_length):
    """
    Insert zeroes between hops place the onsets back into their correct time-domain indices
    :param data:
    :param target_signal:
    :param hop_length:
    :return:
    """
    dataUpsampled = np.zeros(target_signal.size)
    n = 0
    for i in range(0, dataUpsampled.size, hop_length):
        dataUpsampled[i] = data[n]
        n += 1

    return dataUpsampled


def pick_peaks(onsets):

    threshold = np.std(onsets)
    peaks = onsets
    peaks[peaks < threshold] = 0

    return peaks


def normalise(onsets):

    normalisedOnsets = onsets/np.max(onsets)

    return normalisedOnsets
