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
                features = self.generate_local_feature(i, word, labels[i], '<START>')
            else:
                features = self.generate_local_feature(i, word, labels[i], labels[i - 1])
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
            
            
    def generate_local_feature(self, i, word, current_label, prev_label):
        features = {}
        if i == 0:
            features['<START_SENTENCE>'] = 1
        # TODO: Can dynamically change length here
        features['<WORD>' + '-' + word] = 1
        if word.isdigit() and len(word) == 4:
            features['<YEAR>'] = 1
        if any(s.isdigit() for s in word):
            features['<WORD_DIGIT>'] = 1
        if word.isupper():
            features['<ALL_CAP>'] = 1
        if word[0].isupper():
            features['<START_CAP>'] = 1
        
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
    
            
        