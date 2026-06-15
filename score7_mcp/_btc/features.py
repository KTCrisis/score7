"""Feature extraction + chord-index tables for BTC inference.

Extracted verbatim (logic-wise) from BTC's utils/mir_eval_modules.py — only the three
objects inference needs: idx2chord (25-class maj/min), idx2voca_chord (170-class large
vocabulary), and audio_file_to_features (log-magnitude CQT, 144 bins). The original file
also pulls in mir_eval for scoring; those parts are intentionally dropped, so this module
has no mir_eval dependency.

Source: https://github.com/jayg996/BTC-ISMIR19 (MIT, © 2019 Jonggwon Park).
"""

import librosa
import numpy as np

# 25-class major/minor vocabulary (+ "N" = no chord)
idx2chord = ['C', 'C:min', 'C#', 'C#:min', 'D', 'D:min', 'D#', 'D#:min', 'E', 'E:min',
             'F', 'F:min', 'F#', 'F#:min', 'G', 'G:min', 'G#', 'G#:min', 'A', 'A:min',
             'A#', 'A#:min', 'B', 'B:min', 'N']

root_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
quality_list = ['min', 'maj', 'dim', 'aug', 'min6', 'maj6', 'min7', 'minmaj7', 'maj7',
                '7', 'dim7', 'hdim7', 'sus2', 'sus4']


def idx2voca_chord():
    """170-class large vocabulary: 12 roots × 14 qualities (+ X unknown, N no-chord)."""
    table = {169: 'N', 168: 'X'}
    for i in range(168):
        root = root_list[i // 14]
        quality = quality_list[i % 14]
        table[i] = root if i % 14 == 1 else root + ':' + quality
    return table


def audio_file_to_features(audio_file, config):
    """Log-magnitude constant-Q transform, computed in inst_len-second chunks (matches
    BTC training). Returns (feature[bins, frames], seconds_per_frame, song_length_s)."""
    wav, sr = librosa.load(audio_file, sr=config.mp3['song_hz'], mono=True)
    cur = 0
    feature = None
    chunk = int(config.mp3['song_hz'] * config.mp3['inst_len'])
    while len(wav) > cur + chunk:
        tmp = librosa.cqt(wav[cur:cur + chunk], sr=sr, n_bins=config.feature['n_bins'],
                          bins_per_octave=config.feature['bins_per_octave'],
                          hop_length=config.feature['hop_length'])
        feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
        cur += chunk
    tmp = librosa.cqt(wav[cur:], sr=sr, n_bins=config.feature['n_bins'],
                      bins_per_octave=config.feature['bins_per_octave'],
                      hop_length=config.feature['hop_length'])
    feature = tmp if feature is None else np.concatenate((feature, tmp), axis=1)
    feature = np.log(np.abs(feature) + 1e-6)
    feature_per_second = config.mp3['inst_len'] / config.model['timestep']
    song_length_second = len(wav) / config.mp3['song_hz']
    return feature, feature_per_second, song_length_second
