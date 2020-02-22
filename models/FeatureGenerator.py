import functools
import operator
import collections
class FeatureGenerator:
    def __init__(self):
        self.name = 'FeatureGenerator'
        
    def generate_sentence_feature(self, words, labels):
        assert(len(words) == len(labels))
        sentence_features = {}
        for i, word in enumerate(words):
            if i == 0:
                features = self.generate_local_feature(words, i, word, labels[i], '<START>')
            else:
                features = self.generate_local_feature(words, i, word, labels[i], labels[i - 1])
            sentence_features = self._merge_dictionaries([sentence_features, features])
        return sentence_features
    """
        Merge dictionaries together with same keys
        Example:
        initial dictionary [{‘b’: 10, ‘a’: 5, ‘c’: 90}, {‘b’: 78, ‘a’: 45}, {‘a’: 90, ‘c’: 10}]
        resultant dictionary : {‘b’: 88, ‘a’: 140, ‘c’: 100}
    """
    def _merge_dictionaries(self, dictionaries):
        result = dict(functools.reduce(operator.add, 
         map(collections.Counter, dictionaries)))
        return result
    
    def generate_feature(self,features, word, suffix):
        DET = [ 'an', 'a', 'the']
        POS = ['his', 'her', 'its', 'their', 'my', 'your', 'yours', 'mine', 'other',
               'another']
        features = {}
        PUNCT = [',', '.', '\\"', '-', "'s"]
        PARENS = ['(', ')', '[',']']
        PREPOSITION = ['on', 'in', 'at', 'before', 'after', 'inside', 'of', 
                       'for', 'to']
        NUM = ['first', 'second', 'third', 'forth', 'one', 'two', 'three',
               'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'teen'
               ,'ty', 'hundred', 'thousand', 'million', 'billion']
        
        # Check if previous word contains a  num
        if word in DET:
            features['<DET>' + suffix] = 1
        if word in POS:
            features['<POS>'+ suffix] = 1
        if word in PUNCT:
            features['<PUNCT>'+ suffix] = 1
        if word in PARENS:
            features['<PAREN>'+ suffix] = 1
        if any(punct in word for punct in PUNCT): 
            features['<PUNCT>'+ suffix] = 1
        if any(paren in word for paren in PARENS):
            features['<PAREN>'+ suffix] = 1
        if any(prep in word for prep in PREPOSITION):
            features['<PREP>'+ suffix] = 1
        if any(num in word for num in NUM):
            features['<NUM>'+ suffix] = 1
        if word.isdigit() and len(word) == 4:
            features['<YEAR>'+ suffix] = 1
        if any(s.isdigit() for s in word):
            features['<WORD_DIGIT>'+ suffix] = 1
        if all(s.isalpha() for s in word):
            features['<ALL_ALPHA>'+ suffix] = 1
        if word.isupper():
            features['<ALL_CAP>'+ suffix] = 1
        if word[0].isupper():
            features['<START_CAP>'+ suffix] = 1
        if any(s.isupper() for s in word):
            features['<CAP>'+ suffix] = 1
        features['<WORD>' + '-' + word+ suffix] = 1

        return features
            
            
    def generate_local_feature(self, words, i, word, current_label, prev_label):

        features = {}
        # Check previous word
        if (i - 1 >= 0):
            prev_word = words[i - 1]
            features = self.generate_feature(features, prev_word, '-PRE')
        if (i + 1 < len(words)):
            next_word = words[i + 1]
            features = self.generate_feature(features, next_word, '-SUFF')
        
        features = self.generate_feature(features, word, '')                        
        features[str(current_label) + '/' + str(prev_label)] = 1
        
        # Replicate feature for current label
        features = self.replicate_features(features, current_label, 'CUR')
        # Replicate feature for previous label
        features = self.replicate_features(features, prev_label, 'PREV')
        return features
        
    def replicate_features(self, features, label, delimiter):
        new_features = {}
        for feature, value in features.items():
            new_features[feature] = value
            new_features[feature + '-' + delimiter + '-' + label] = value
        return new_features
    
            
        