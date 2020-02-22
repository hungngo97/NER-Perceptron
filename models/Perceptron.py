# -*- coding: utf-8 -*-
from operator import itemgetter
from collections import defaultdict
from random import shuffle
import random
from .FeatureGenerator import FeatureGenerator
class Perceptron:
    def __init__(self, max_iter, labels):
        self.START_TOKEN = "<<START_TOKEN>>"
        self.name = 'Perceptron'
        self.max_iter = max_iter
        # Weights is a dictionary in form of
        # { feature -> { label -> float }}
        # Matrix of shape (L, F) * (F, 1)
        self.weights = defaultdict(lambda: defaultdict(float))
        
        self.steps = defaultdict(lambda: defaultdict(int))
        self.accum_weights = defaultdict(lambda: defaultdict(float))
        self.clock = 1
        self.featureGenerator = FeatureGenerator()
        self.RANDOM_LOWER_BOUND = -1
        self.RANDOM_UPPER_BOUND = 1


    def calculate_value_from_weights(self, label, features):
        value = 0
        for feature, value in features.items():
            wt = self.weights[feature].get(label, random.uniform(self.RANDOM_LOWER_BOUND, self.RANDOM_UPPER_BOUND))
            value += wt * value
        return value
            


        
    def predict_viterbi(self, words, labels):
        value = 0
        result_ner_tags = []
        # Base case
        previous_label = None
        for i in range(len(words)):
            """if i == 0:
                previous_label = self.START_TOKEN
                # TODO: This should also use Viterbi
                # result_ner_tags.append(previous_label)
                continue # Base case for START token
                """
            #if i == 1:
            if i == 0:
                for potential_label in labels:
                    curr_local_feature = self.featureGenerator.generate_local_feature(words, i, words[i], potential_label, self.START_TOKEN)
                    # TODO: This self.accum-weights op is wrong
                    # curr_value = self.weights * curr_local_feature
                    curr_value = self.calculate_value_from_weights(potential_label, curr_local_feature)
                    if curr_value > value:
                        value = curr_value
                        previous_label = potential_label
                result_ner_tags.append(previous_label)
            else:
                # Recursive Viterbi 
                max_value_current = float('-inf')
                max_label_current = None
                for potential_label in labels:
                    # pi(i, yi) = pi(i - 1, y_i-1) * phi(i, y_i, y_i - 1)
                    curr_local_feature = self.featureGenerator.generate_local_feature(words, i, words[i], potential_label, previous_label)
                    # curr_value = value + self.weights * curr_local_feature
                    curr_value = self.calculate_value_from_weights(potential_label, curr_local_feature)
                    if curr_value > max_value_current:
                        max_value_current = curr_value
                        max_label_current = potential_label
                result_ner_tags.append(max_label_current)
                value = max_value_current
                previous_label = max_label_current
        return result_ner_tags

    def train(self, train_data):
        for i in range(self.max_iter):
            print('---- Iteration ' + str(i))
            for texts, features, labels in train_data:
                print('------ Training 1 sentence -----')
                pred_labels = self.predict_viterbi(texts, labels) 
                self.clock += 1
                if pred_labels != labels:
                    self.update_weights(texts, labels, pred_labels, features)
        self.average_weights()
        return self.weights

    def update_weights(self, texts, labels, pred_labels, features):
        assert labels != pred_labels
        print('Update weights', labels, pred_labels)
        predicted_features = self.featureGenerator.generate_sentence_feature(texts, pred_labels)
        for label in labels:
            for feature, value in features.items():
                pred_value = predicted_features.get(feature, 0.0)
                value_diff = value - pred_value
                self.weights[feature][label] += value_diff
                self.accum_weights[feature][label] += self.weights[feature][label]

    def average_weights(self):
        """
            Average weights after iterations for regularization
        """
        
        for feature, labels_weights in self.weights.items():
            avg_weight = {}
            for label, value in labels_weights.items():
                acc_wt = self.accum_weights[feature][label]
                # acc_wt += (self.clock )
                avg_weight[label] = acc_wt / self.clock
            self.weights[feature] = avg_weight


    # def predict_train(self, features):
    #     # (L, 1) * (1, N)
    #     """
    #         Output wanted to have is 
    #         {N, L}
    #         so that each row we can take the argmax to get the label
            
    #     """
    #     scores = { label: 0 for label in self.labels }
    #     for feature, value in features:
    #         # Weights should be in (F, 1)
    #         if feature in self.weights:
    #             for label, weight in self.weights[feature].items():
    #                 scores[label] += value * weight
    #     return max(scores.iteritems(),key=itemgetter(1))
    
    # """
    #     train_data is a list of training sentences which each sentence is in form
    #     of ( features_dict, list_ner_labels )
    # """
    # def train(self, train_data):
    #     for i in range(self.max_iter):
    #         print('---- Iteration ' + str(i))
    #         for features, labels in train_data:
    #             print('------ Training 1 sentence -----')
    #             pred_labels = self.predict_train(self, features) 
    #             self.clock += 1
    #             if pred_labels != labels:
    #                 self.update_weights(labels, pred_labels, features)
    #     self.average_weights()
    #     return self.weights
    
    # def update_weights(self, labels, pred_labels, features):
    #     assert labels != pre_labels
    #     for feature, value in features.items():
    #         self.update(feature, labels, 1)
    #         self.update(feature, pred_label, -1)
            
    # def update(self, feature, label, value):
    #     wt = self.weights[feature].get(label, 0.0)
    #     self.accum_weights[feature][label] += wt * (self.clock - self.timestamps[feature][label])
    #     self.weights[feature][label] += value
    #     self.timestamps[feature][label] = self.clock
        
    # def average_weights(self):
    #     """
    #         Average weights after iterations for regularization
    #     """
        
    #     for feature, labels_weights in self.weights.items():
    #         avg_weight = {}
    #         for label, value in labels_weights.items():
    #             acc_wt = self.accum_weights[feature][label]
    #             # acc_wt += (self.clock )
    #             avg_weight[label] = acc_wt / self.clock
    #         self.weights[feature] = avg_weight
                
        
