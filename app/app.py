import os
import io
import re

import streamlit as st
import soundfile as sf
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch
import torch.nn as nn
import pandas as pd

from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models.hdpmodel import HdpModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import matplotlib.pyplot as plt
from collections import Counter
from wordcloud import WordCloud

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
class BeamSearchDecoderWithLM(nn.Module):
    """Neural Module that does CTC beam search with a N-gram language model.
    It takes a batch of log_probabilities. Note the bigger the batch, the
    better as processing is parallelized. Outputs a list of size batch_size.
    Each element in the list is a list of size beam_search, and each element
    in that list is a tuple of (final_log_prob, hyp_string).
    Args:
        vocab (list): List of characters that can be output by the ASR model. For English, this is the 28 character set
            {a-z '}. The CTC blank symbol is automatically added.
        beam_width (int): Size of beams to keep and expand upon. Larger beams result in more accurate but slower
            predictions
        alpha (float): The amount of importance to place on the N-gram language model. Larger alpha means more
            importance on the LM and less importance on the acoustic model.
        beta (float): A penalty term given to longer word sequences. Larger beta will result in shorter sequences.
        lm_path (str): Path to N-gram language model
        num_cpus (int): Number of CPUs to use
        cutoff_prob (float): Cutoff probability in vocabulary pruning, default 1.0, no pruning
        cutoff_top_n (int): Cutoff number in pruning, only top cutoff_top_n characters with highest probs in
            vocabulary will be used in beam search, default 40.
    """

    def __init__(
        self, vocab, beam_width, alpha, beta, lm_path, num_cpus, cutoff_prob=1.0, cutoff_top_n=40
    ):

        try:
            from ctc_decoders import Scorer, ctc_beam_search_decoder_batch
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "BeamSearchDecoderWithLM requires the installation of ctc_decoders "
                "from scripts/asr_language_modeling/ngram_lm/install_beamsearch_decoders.sh"
            )

        super(BeamSearchDecoderWithLM, self).__init__()

        if lm_path is not None:
            self.scorer = Scorer(alpha, beta, model_path=lm_path, vocabulary=vocab)
        else:
            self.scorer = None
        self.beam_search_func = ctc_beam_search_decoder_batch
        self.vocab = vocab
        self.beam_width = beam_width
        self.num_cpus = num_cpus
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n

    @torch.no_grad()
    def forward(self, log_probs, log_probs_length):
        probs = torch.exp(log_probs)
        probs_list = []
        for i, prob in enumerate(probs):
            probs_list.append(prob[: log_probs_length[i], :])

        res = self.beam_search_func(
            probs_list,
            self.vocab,
            beam_size=self.beam_width,
            num_processes=self.num_cpus,
            ext_scoring_func=self.scorer,
            cutoff_prob=self.cutoff_prob,
            cutoff_top_n=self.cutoff_top_n,
        )
        return res


class ASRPipeline():
    def __init__(self):
        CTC_MODEL_PATH='stt_en_jasper10x5dr.nemo'
        LM_3GRAM_PATH = '3-gram.arpa'
        ID2WORD_MODEL_PATH = "id2word.dict"
        LDA_MODEL_PATH = "lda_model.model"
        TOPIC_NAMES_PATH = "topic_names.csv"
        FACEBOOK_BART = "facebook/bart-large-xsum"

        self.asr_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=CTC_MODEL_PATH).to(DEVICE)
        self.asr_model.eval()
        self.vocabulary = list(map(lambda x: x.upper(), self.asr_model.decoder.vocabulary))
        self.beam_search_lm = BeamSearchDecoderWithLM(
            vocab=self.vocabulary,
            beam_width=16,
            alpha=1.5, beta=1.5,
            lm_path=LM_3GRAM_PATH,
            num_cpus=max(os.cpu_count(), 1))

        self.lda_model = HdpModel.load(LDA_MODEL_PATH)
        self.id2word = Dictionary.load(ID2WORD_MODEL_PATH)
        self.num_topics = len(self.lda_model.get_topics())
        self.topic_keywords = [([word for word, prop in self.lda_model.show_topic(t)]) for t in range(self.num_topics)]
        self.topic_names = pd.read_csv(TOPIC_NAMES_PATH, index_col=0)['Topic_Name'].to_list()

        self.sum_tokenizer = AutoTokenizer.from_pretrained(FACEBOOK_BART)
        self.sum_model = AutoModelForSeq2SeqLM.from_pretrained(FACEBOOK_BART)

        self._audio_cache = {}
        print("Setup done!")

    def get_topic_names(self):
        return self.topic_names

    def get_topic_keywords(self):
        return self.topic_keywords

    def get_cache(self, key):
        return self._audio_cache.get(key, None)

    def add_cache(self, key, val):
        self._audio_cache[key] = val

    def get_text(self, input_signal, input_signal_length):
        """
            args:
                input_signal (list): shape (BATCH, LENGTH) LENGTH is just the maxlen, each tensor is padded
                input_signal_length (list): shape (BATCH) containing the length of each input signal

            return:
                tuple: list of strings with text predictions and list of list of tuples [[(topic, weight)]]

        """
        with torch.no_grad():
            log_probs, encoded_len, _ = self.asr_model(input_signal=input_signal, input_signal_length=input_signal_length)
            transcriptions = self.beam_search_lm(log_probs=log_probs, log_probs_length=encoded_len)

        audiobook_text = list(map(lambda xs: xs[0][1], transcriptions))

        return audiobook_text

    def get_topics(self, audiobook_text, threshold=0.05):
        """
            args:
                audiobook_text (list): list of strings with text

            return:
                sorted list of topics [[(topic, weight)]] removing threshold > weights
        """
        x = list(map(self.id2word.doc2bow, map(word_tokenize, map(str.lower, audiobook_text))))
        y_hat = self.lda_model[x]

        data = [{int(k): v for k, v in y if v >= threshold} for y in y_hat]
        data = [sorted(d.items(), key=lambda e: e[1], reverse=True) for d in data]
        return data

    def get_summary(self, audiobook_text):
        """
            args:
                audiobook_text (str):strings with text

            return:
                str: summary of text
        """
        inputs = self.sum_tokenizer.encode("summarize: " + audiobook_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.sum_model.generate(inputs, max_length=550, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.sum_tokenizer.decode(outputs[0])

        summary = re.sub("</s>", "", summary)

        return summary


def print_topics(topic_keywords, topic_names):
    data = [[topic_names[k], ", ".join(topic_keywords[k])] for k in range(len(topic_names))]
    df = pd.DataFrame(data, columns=["Topic", f"keywords"])
    st.table(df)


def print_pie_audiobook(topics, topic_names, audiobook_title):
    pie, ax = plt.subplots(figsize=[10, 6])

    values = list(map(lambda x: x[1], topics))
    keys = list(map(lambda x: topic_names[x[0]], topics))

    plt.pie(x=values, autopct="%.1f%%", explode=[0.05]*len(keys), labels=keys, normalize=True)
    plt.title(f"Topics contained by the {audiobook_title} book", fontsize=14)
    st.pyplot(pie)


def print_wordcloud(sentences, title="wordcloud"):
    all_freq = sum([Counter(words) for words in sentences], Counter())

    wordcloud = WordCloud(width=800, height=500, max_font_size=110).generate_from_frequencies(all_freq)

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.title(title, weight='bold', fontsize=14)
    st.pyplot(fig)


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


@st.cache(allow_output_mutation=True)
def init_pipeline():
    return ASRPipeline()

import streamlit.components.v1 as components
if __name__ == "__main__":
    st.set_page_config(layout='wide')

    pipeline = init_pipeline()

    topic_names = pipeline.get_topic_names()
    topic_keywords = pipeline.get_topic_keywords()

    st.title("Book Treasure Demo")

    audio_file = st.file_uploader("Upload Audio", type=['wav', 'flac'])
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()
        audio_path = Path(audio_file.name)
        audiobook_title=audio_path.stem
        audio_type = audio_file.type

        st.write('You selected `%s`' % audio_path)
        st.audio(audio_bytes, format=audio_type)
        if st.button("Run"):
            print(f"Loading {audio_path}")

            st.title(audiobook_title)
            audiobook_text = pipeline.get_cache(audiobook_title) or []

            if not audiobook_text:
                audiobook_gen = sf.blocks(io.BytesIO(audio_bytes), blocksize=1_000_000, dtype='float32')

                for i, audiobook_signal in enumerate(audiobook_gen):
                    print(f'Chunk {i}')
                    input_signal = torch.from_numpy(audiobook_signal).unsqueeze(0)[0].to(DEVICE).unsqueeze(0)
                    input_signal_length = torch.tensor([input_signal.shape[-1]], device=DEVICE)

                    text = pipeline.get_text(input_signal, input_signal_length)[0]
                    audiobook_text.append(text)

                audiobook_text = " ".join(audiobook_text)
                pipeline.add_cache(audiobook_title, audiobook_text)

            st.write(audiobook_text)

            st.title(f"Predicted topic for the book \"{audiobook_title}\"")
            topics = pipeline.get_topics([audiobook_text], threshold=0.05)[0]

            book_topic_keywords = [topic_keywords[k] for k, _ in topics]
            book_topic_names = [topic_names[k] for k, _ in topics]
            print_topics(book_topic_keywords, book_topic_names)

            print_wordcloud(book_topic_keywords, title="Topic keywords")

            _stopwords = set(stopwords.words('english'))
            print_wordcloud([list(filter(lambda word: word not in _stopwords, word_tokenize(audiobook_text.lower())))], title=f"Most frequent words from the book \"{audiobook_title}\"")

            st.title(f"Summary for the book \"{audiobook_title}\"")
            audiobook_summary = pipeline.get_summary(audiobook_text)
            st.write(audiobook_summary)

            print("Done!")

    st.title(f"All topics")
    print_topics(topic_keywords, topic_names)
