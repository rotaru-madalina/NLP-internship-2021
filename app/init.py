import os
import gzip
import shutil
import subprocess

from logging import error


CTC_MODEL_NAME = 'stt_en_jasper10x5dr'
CTC_MODEL_PATH = f'{CTC_MODEL_NAME}.nemo'
LM_3GRAM_PATH = '3-gram.arpa'
ID2WORD_MODEL_PATH = "id2word.dict"
ID2WORD_MODEL_FILEID = "https://github.com/alexjercan/asr-toolkit/releases/download/v1.0/id2word.dict"
LDA_MODEL_PATH = "lda_model.model"
LDA_MODEL_FILEID = "https://github.com/alexjercan/asr-toolkit/releases/download/v1.0/lda_model.model"
TOPIC_NAMES_PATH = "topic_names.csv"
TOPIC_NAMES_FILEID = "https://github.com/alexjercan/asr-toolkit/releases/download/v1.0/topic_names.csv"


def run_command(args, info=print, error=error, **kwargs):
    """Run command, transfer stdout/stderr back into Streamlit and manage error"""
    info(f"Running '{' '.join(args)}'")
    result = subprocess.run(args, capture_output=True, text=True, **kwargs)
    try:
        result.check_returncode()
        info(result.stdout)
    except subprocess.CalledProcessError as e:
        error(result.stderr)
        raise e


def download_lm(lm_path):
    print(f"Downloading {lm_path}")
    if os.path.exists(f"{lm_path}*"):
        os.remove(f"{lm_path}*")
    wget.download(f"https://www.openslr.org/resources/11/{lm_path}.gz", out=f"{lm_path}.gz")
    with gzip.open(f"{lm_path}.gz", 'rb') as f_in:
        with open(f"{lm_path}", 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_from_github(file_id, file_name):
    print(f"Downloading {file_name}")
    if os.path.exists(file_name):
        os.remove(file_name)
    wget.download(f"{file_id}", out=f"{file_name}")


def download_ctcmodel(model_name, model_path):
    print(f"Downloading {model_name}")
    model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name=model_name, strict=False)
    model.save_to(model_path)


if __name__ == "__main__":
    info = print
    error = error

    run_command(["sudo", "ln", "/usr/bin/python3", "/usr/bin/python", info, error])

    run_command(["python", "-m", "pip", "install", "torch"], info, error)
    run_command(["python", "-m", "pip", "install", "torchaudio"], info, error)
    run_command(["python", "-m", "pip", "install", "gensim"], info, error)
    run_command(["python", "-m", "pip", "install", "streamlit"], info, error)
    run_command(["python", "-m", "pip", "install", "nltk"], info, error)
    run_command(["python", "-m", "pip", "install", "wget"], info, error)
    run_command(["python", "-m", "pip", "install", "numpy"], info, error)
    run_command(["python", "-m", "pip", "install", "numba"], info, error)
    run_command(["python", "-m", "pip", "install", "librosa"], info, error)
    run_command(["python", "-m", "pip", "install", "transformers"], info, error)
    run_command(["python", "-m", "pip", "install", "wordcloud"], info, error)

    run_command(["python", "-m", "pip", "install", "git+https://github.com/NVIDIA/NeMo.git@main#egg=nemo_toolkit[all]"], info, error)

    run_command(["sudo", "apt-get", "install", "-y", "swig"], info, error)
    dirpath = os.path.join('NeMo')
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)
    run_command(["git", "clone", "https://github.com/NVIDIA/NeMo", "-b", "main"], info, error)
    run_command(["sudo", "bash", "scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh"], info, error, cwd="NeMo")

    import nltk
    import wget
    import nemo
    import nemo.collections.asr as nemo_asr

    if not os.path.exists(CTC_MODEL_PATH):
        download_ctcmodel(CTC_MODEL_NAME, CTC_MODEL_PATH)

    if not os.path.exists(LM_3GRAM_PATH):
        download_lm(LM_3GRAM_PATH)

    nltk.download("all")

    if not os.path.exists(LDA_MODEL_PATH):
        download_from_github(LDA_MODEL_FILEID, LDA_MODEL_PATH)

    if not os.path.exists(ID2WORD_MODEL_PATH):
        download_from_github(ID2WORD_MODEL_FILEID, ID2WORD_MODEL_PATH)

    if not os.path.exists(TOPIC_NAMES_PATH):
        download_from_github(TOPIC_NAMES_FILEID, TOPIC_NAMES_PATH)

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-xsum")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-xsum")