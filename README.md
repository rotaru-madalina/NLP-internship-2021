# Improvement on audio books search functionality leveraging Automatic Speech Recognition

## Objective

In this project our main goal is to create a audio book tagging and summarization pipeline using automatic speech recognition.

## Getting started

##### Tested using:
* Google Colaboratory
* conda 4.10.3

##### Requirements:
In order to run this application you need to use python
* [Python 3.8](https://www.python.org/downloads/)

Additional libraries are required to run the notebooks
* PyTorch 1.9+
* torchaudio
* nltk
* SpaCy
* sklearn
* gensim

##### Usage
1. Download the repository
```shell
$ git clone https://not/sure/what/the/path/is/nlp.git
$ cd nlp
```
2. Start the jupyter notebook kernel using conda or other methods
```shell
$ jupyter notebook
```
3. Access `localhost:8888` to see the available notebooks
4. Use the [00_DataUnderstanding](./00_DataUnderstanding.ipynb) notebook to visualize the audio files and the metrics of the [LibriSpeech](https://www.openslr.org/12) dataset.
5. Use the [01_LanguageModels_with_PyTorch](01_LanguageModels_with_PyTorch.ipynb) notebook to test the different models that we used (Jasper and the different Language models).
6. Use the [02_AudioBooksToText](02_AudioBooksToText.ipynb) notebook to generate the input for the nlp phase using the LibriSpeech data and the ASR model that worked the best.
7. Use the [03_TopicModeling](03_TopicModeling.ipynb) notebook to run the final phase of the pipeline and train the LDA and HDP models we tested.
8. [04_CompletePipeline](04_CompletePipeline.ipynb) is an example of how the models should be used, and can also work as a template for other applications.

## Demo application

See [README](./app/README.md)