import glob
import shutil
import numpy as np
import os
import progressbar
import sys
import pydub.effects
from numpy import array
from progressbar import ProgressBar
from pydub import AudioSegment

from WaveRNN.wavernn.utils import hparams as hp
from WaveRNN.wavernn.utils import dsp


def main():
    argv0: str = sys.argv[0]
    if argv0:
        workdir: str = os.path.dirname(argv0)
        if workdir:
            os.chdir(workdir)

    hp.configure("hparams.py")
    dsp.hp = hp

    dataset_paths: list[str] = list()
    for dataset_path in glob.glob("dataset/*"):
        if os.path.isdir(os.path.join(dataset_path, "wavs")):
            dataset_paths.append(dataset_path)

    dataset_paths.sort()

    for dataset_path in dataset_paths:
        print(f"Processing {dataset_path}")
        shutil.rmtree(os.path.join(dataset_path, "gtas"), ignore_errors=True)
        os.makedirs(os.path.join(dataset_path, "gtas"), exist_ok=True)
        wav_files: list[str] = glob.glob(os.path.join(dataset_path, "wavs", "*.wav"))
        bar: ProgressBar = progressbar.ProgressBar(maxval=len(wav_files))
        bar.start()
        for wav in wav_files:
            filename: str = os.path.splitext(os.path.basename(wav))[0]
            py_audio: AudioSegment = AudioSegment.from_file(wav)
            py_audio = pydub.effects.normalize(py_audio)
            py_audio.set_channels(1).set_frame_rate(hp.sample_rate)
            mel_file = os.path.join(dataset_path, "gtas", f"{filename}.npy")
            py_audio_samples: array = np.array(py_audio.get_array_of_samples()).astype(np.float32)
            py_audio_samples = py_audio_samples / (1 << 8 * 2 - 1)
            np.save(mel_file, dsp.melspectrogram(py_audio_samples))
            bar.update(bar.currval + 1)
        bar.finish()


if __name__ == "__main__":
    main()
