# cherokee-wavernn

Special project to train wavernn on Cherokee and English domain specific audio.
Ref: (CherokeeLanguage/Cherokee-TTS)[../../CherokeeLanguage/Cherokee-TTS]

## Environment

Environment is set for CUDA 11 and python 3.9.7

The setup script will create or reset the environment ```cherokee-wavernn``` 
to match the (environment.yml)[environment.yml] configuration
and then install the (WaveRNN)[WaveRNN] git submodule as a local dependency. 

```bash
bash setup-python-code-env.sh
```

Project was developed under Ubuntu 21.10 with miniconda locally installed.
