import glob
import os
import shutil
import subprocess
import sys

from progressbar import ProgressBar
from pydub import AudioSegment
from typing import List


def main():
    run_create_scripts: bool = True
    argv0: str = sys.argv[0]
    if argv0:
        workdir: str = os.path.dirname(argv0)
        if workdir:
            os.chdir(workdir)

    if run_create_scripts:
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
                audio = audio.set_channels(1).set_frame_rate(22050)
                export_path = os.path.join("dataset", lang, "wavs")
                os.makedirs(export_path, exist_ok=True)
                audio.export(os.path.join(export_path, f"{idx:09d}.wav"), format="wav")
                idx += 1
        bar.update(bar.currval + 1)
    bar.finish()

    cmd_list: list[str] = list()

    print()
    print("=== Creating MEL files")
    cmd_list.append("python")
    cmd_list.append("prepare_spectrograms.py")
    subprocess.run(cmd_list, check=True)

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




if __name__ == "__main__":
    main()
