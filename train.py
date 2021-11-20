import os
import subprocess
import sys


def main() -> None:
    argv0: str = sys.argv[0]
    if argv0:
        workdir: str = os.path.dirname(argv0)
        if workdir:
            os.chdir(workdir)

    cmd_list: list[str] = list()
    print()
    print("=== WaveRNN preprocess")
    os.chdir("WaveRNN")
    cmd_list.clear()
    cmd_list.append("python")
    cmd_list.append("scripts/train_wavernn.py")
    cmd_list.append("--hp_file")
    cmd_list.append("../hparams.py")
    subprocess.run(cmd_list, check=True)
    os.chdir("..")


if __name__ == "__main__":
    main()