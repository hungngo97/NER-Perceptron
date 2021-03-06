# -*- coding: utf-8 -*-
import re
import random
from constants.model_constants import (UNK, STOP, START)

DOC_START_MARK = '-DOCSTART-'
SENTENCE_END_MARK = '\n\n'
WORD_END_MARK = '\n'

_WORD_DATA_SEP = ['\r ', ' ']
WORD_DATA_SEP_PATTERN = '|'.join(map(re.escape, _WORD_DATA_SEP))

"""
    @Returns: list of documents in CONLL data which each item is
    a document consist of sentences, which is a list of the sentence's word
    and its properties { 'text': Michael, 'pos': Noun, 'chunk': ABC, 'ner_label': PER}
"""
def read_conll_file(filename):
    # TODO: Should add a START token here
    document_list = list()
    with open(filename, "r") as conll_file:
        file = conll_file.read()
        for document in file.split(DOC_START_MARK)[1:]:
            document_parse = list()
            print(document)
            for sentence in document.split(SENTENCE_END_MARK)[1:-1]:
                sentence_parse = list()
                for word in sentence.split(WORD_END_MARK)[:-1]:
                    text, pos, chunk, ner_label = re.split(WORD_DATA_SEP_PATTERN, word)
                    word_dict = {
                        'text': text,
                        'pos': pos,
                        'chunk': chunk,
                        'ner_label': ner_label
                    }
                    sentence_parse.append(word_dict)
                document_parse.append(sentence_parse)
            document_list.append(document_parse)
    return document_list

def format_conll_tagged(model, document_list, label_set):
    conll_text = ''
    for document in document_list:
        # document_text = '-DOCSTART-'
        # first_sentence_mock = '\r -X- O O O\n\n'
        # document_text += first_sentence_mock

        document_text = ''
        for sentence in document:
            sentence_text = ""
            texts = [word['text'] for word in sentence]
            # labels = [word['ner_label'] for word in sentence]
            predict_tags = model.predict_viterbi(texts, label_set)
            # sentence_features = feature_generator.generate_sentence_feature(texts, labels)
            assert len(predict_tags) == len(sentence)
            for i in range(len(sentence)):
                word_attrs = sentence[i]
                word_text = '%s %s %s %s %s' % (word_attrs['text'], word_attrs['pos'], word_attrs['chunk'],
                                                  word_attrs['ner_label'], predict_tags[i])
                word_text += '\n'

                sentence_text += word_text
            # sentence_text += '\n'
            document_text += sentence_text
        conll_text += document_text
    return conll_text

def get_predict_tags_output(model, document_list, label_set):
    conll_text = ''
    for document in document_list:
        # document_text = '-DOCSTART-'
        # first_sentence_mock = '\r -X- O O O\n\n'
        # document_text += first_sentence_mock

        document_text = ''
        for sentence in document:
            sentence_text = ""
            texts = [word['text'] for word in sentence]
            # labels = [word['ner_label'] for word in sentence]
            predict_tags = model.predict_viterbi(texts, label_set)
            # sentence_features = feature_generator.generate_sentence_feature(texts, labels)
            assert len(predict_tags) == len(sentence)
            for i in range(len(sentence)):
                word_attrs = sentence[i]
                word_text = '%s %s %s %s %s' % (word_attrs['text'], word_attrs['pos'], word_attrs['chunk'],
                                                  word_attrs['ner_label'], predict_tags[i])
                word_text += '\n'

                sentence_text += word_text
            # sentence_text += '\n'
            document_text += sentence_text
        conll_text += document_text
    return conll_text

"""
    @Returns a dict frequency of all NER labels from list of documents
"""
def get_labels(documents):
    labels = {}
    for document in documents:
        for sentence in document:
            for word in sentence:
                ner_label = word['ner_label']
                if ner_label not in labels:
                    labels[ner_label] = 0
                labels[ner_label] += 1
    return labels
                


"""
    Read sentences from a file and split all string into a list of words in each
    string
    
    @Returns: List of list of words with each inner list is a sentence
"""
def read_sentences_from_file(file_path, unk=True, unk_threshold=3):
    with open(file_path, "r") as file:
        sentences = []
        for sentence in file:
            words = [START]
            words = words + re.split("\s+", sentence.rstrip())
            words = words + [STOP]
            sentences.append(words)
        if (unk):
            sentences = unk_sentences(sentences, unk_threshold=3, unk_prob=0.5)
        return sentences
    
def unk_sentences(sentences, unk_threshold=3, unk_prob=0.5):
    token_frequency = dict()
    """
    1) Count the frequency of all tokens in corpus.
    2) Choose a cutoff and some UNK probability (e.g. 5 and 50%)
    3) For all **individual tokens** that appear at or below cutoff, replace 50% of them with UNK.
    4) Estimate the probabilities for from its counts just like any other regular
    word in the training set.
    5) At dev/test time, replace words model hasn't seen before with UNK.
    """
    for sentence in sentences:
        for word in sentence:
            token_frequency[word] = token_frequency.get(word, 0) + 1
            
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if (token_frequency[word] < unk_threshold):
                # Replace the current token with UNK with UNK probability
                if (random.random() > unk_prob):
                    sentence[i] = UNK
    return sentences
                
        
        
