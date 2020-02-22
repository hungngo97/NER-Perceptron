from Perceptron import *
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

def make_train_data(documents):
    result = list()
    for document in documents:
        for sentence in document:
            words = list()
            texts = list()
            tags = list()
            for word in sentence:
                texts.append(word['text'])
                tags.append(word['ner_label'])
            result.append((texts, tags))
    return result

def test_sentence(data, i,  weights, labels):
    print('Tokens', data[i][0])
    print('Predicted', predict_seq(data[i][0], weights, labels))
    print('Actual', data[i][1])

def get_predict_tags_output(weights, document_list, labels):
    conll_text = ''
    for document in document_list:
        document_text = ''
        for sentence in document:
            sentence_text = ""
            texts = [word['text'] for word in sentence]
            # labels = [word['ner_label'] for word in sentence]
            predict_tags = predict_seq(texts, weights, labels)
            # sentence_features = feature_generator.generate_sentence_feature(texts, labels)
            assert len(predict_tags) == len(sentence)
            for i in range(len(sentence)):
                word_attrs = sentence[i]
                DELIMITER = ' '
                word_text = DELIMITER.join([word_attrs['text'], word_attrs['pos'], word_attrs['chunk'],
                                                  word_attrs['ner_label'], predict_tags[i]])
                word_text += '\n'

                sentence_text += word_text
            # sentence_text += '\n'
            document_text += sentence_text
        conll_text += document_text
    return conll_text

def find_key_with_suffix(test_dict, suffix):
    res = {key:val for key, val in test_dict.items() if key.endswith(suffix)} 
    return res

def find_key_with_prefix(test_dict, prefix):
    res = {key:val for key, val in test_dict.items() if key.startswith(prefix)} 
    return res
             
labels_set = get_labels(parsed_document_list)
labels = set(labels_set.keys())
train_data = make_train_data(parsed_document_list)
dev_data = make_train_data(parsed_dev_document_list)
weights = train(labels, train_data, do_averaging=True, devdata=dev_data,numpasses=3)
test_sentence(train_data, 40, weights, labels)
conll_text_predict = get_predict_tags_output(weights, parsed_document_list, labels)

with open('output.txt', 'w') as file:
    file.write(conll_text_predict)
    
conll_text_predict = get_predict_tags_output(weights, parsed_dev_document_list, labels)

with open('dev_output.txt', 'w') as file:
    file.write(conll_text_predict)
    
conll_text_predict = get_predict_tags_output(weights, parsed_test_document_list, labels)

with open('test_output.txt', 'w') as file:
    file.write(conll_text_predict)
    


