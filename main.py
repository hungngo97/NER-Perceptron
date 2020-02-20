from utils.io_utils import (read_conll_file, get_labels)
from utils.math_utils import *
from constants.model_constants import (UNK, START, STOP)
from constants.constants import (DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.Perceptron import (Perceptron)
from evaluator.evaluator import ( PerplexityEvaluator )
from models.FeatureGenerator import ( FeatureGenerator )

#Read data input
parsed_document_list = read_conll_file(TRAIN_DATA_FILE)
parsed_test_document_list = read_conll_file(TEST_DATA_FILE)
parsed_dev_document_list = read_conll_file(DEV_DATA_FILE)


#
feature_generator = FeatureGenerator()
def make_data(documents):
    feature_generator = FeatureGenerator()
    data = []
    for document in documents:
        for sentence in document:
            texts = [word['text'] for word in sentence]
            labels = [word['ner_label'] for word in sentence]
            sentence_features = feature_generator.generate_sentence_feature(texts, labels)
            data.append((sentence_features, labels))
    return data

train_data = make_data(parsed_document_list)
            