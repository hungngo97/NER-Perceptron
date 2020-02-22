from utils.io_utils import (read_conll_file, get_labels, format_conll_tagged)
from utils.math_utils import *
from constants.model_constants import (UNK, START, STOP)
from constants.constants import (DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.Perceptron import (Perceptron)
from evaluator.evaluator import ( Evaluator )
from models.FeatureGenerator import ( FeatureGenerator )


NUM_ITER = 3
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
            data.append((texts, sentence_features, labels))
    return data




labels_set = get_labels(parsed_document_list)
train_data = make_data(parsed_document_list)
model = Perceptron(NUM_ITER, labels_set)

weights = model.train(train_data)

# Evaluator
evaluator = Evaluator()
confusion_matrix = evaluator.evaluate_model(train_data, model, labels_set)
evaluator.print_confusion_matrix(confusion_matrix)
prec, recall, fscore = evaluator.metrics(confusion_matrix)
print('prec', prec, 'recall', recall, 'f1', fscore)

# Output file
conll_text = format_conll_tagged(model, parsed_document_list, labels_set)