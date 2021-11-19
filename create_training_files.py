import subprocess
from typing import List
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
    hp.configure("hparams.py")
    dsp.hp = hp

    argv0: str = sys.argv[0]
    if argv0:
        workdir: str = os.path.dirname(argv0)
        if workdir:
            os.chdir(workdir)

    exec_list: List[str] = []
    exec_list.extend(glob.glob("*/*/create_tts_files.py"))
    exec_list.sort()
    for exec_filename in exec_list:
        print(f"=== {exec_filename}")
        sub_dir = os.path.join(os.path.dirname(exec_filename), "wav")
        if os.path.isdir(sub_dir):
            continue
        script: str = f"""
            eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
            conda deactivate
            conda activate cherokee-wavernn
            python "{exec_filename}"            
            exit $?
        """

        cp: subprocess.CompletedProcess = subprocess.run(script, shell=True, executable="/bin/bash", check=True)
        if cp.returncode > 0:
            raise Exception("Subprocess exited with ERROR")

    shutil.rmtree("dataset/wavs", ignore_errors=True)
    os.mkdir("dataset/wavs")

    langs: list[str] = list()
    idx: int = 0
    all_txts: List[str] = glob.glob("*/*/all.txt")
    print("\n=== Creating ground truth wavs")
    bar: ProgressBar = ProgressBar(maxval=len(all_txts))
    bar.start()
    for all_txt in all_txts:
        all_dir: str = os.path.dirname(all_txt)
        with open(all_txt, "r") as r:
            for line in r:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split("|")
                if len(fields) != 8:
                    continue
                wav_file = fields[3]
                lang = fields[2]
                if lang not in langs:
                    langs.append(lang)
                audio: AudioSegment = AudioSegment.from_file(os.path.join(all_dir, wav_file))
                audio = audio.set_channels(1).set_frame_rate(hp.sample_rate)
                export_path = os.path.join("dataset", lang, "wavs")
                os.makedirs(export_path, exist_ok=True)
                audio.export(os.path.join(export_path, f"{idx:09d}.wav"), format="wav")
                idx += 1
        bar.update(bar.currval + 1)
    bar.finish()

    cmd_list: list[str] = list()

    print()
    print("=== Creating MEL files")
    prepare_spectrograms()

    print()
    print("=== WaveRNN preprocess")
    os.chdir("WaveRNN")
    cmd_list.clear()
    cmd_list.append("python")
    cmd_list.append("scripts/preprocess.py")
    cmd_list.append("--data_root")
    cmd_list.append("../dataset")
    cmd_list.append("--output")
    cmd_list.append("../dataset")
    cmd_list.append("--hp_file")
    cmd_list.append("../hparams.py")
    cmd_list.append("--inputs")
    for lang in langs:
        cmd_list.append(lang)
    subprocess.run(cmd_list, check=True)
    os.chdir("..")


def prepare_spectrograms():
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
