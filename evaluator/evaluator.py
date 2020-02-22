# -*- coding: utf-8 -*-
"""

"""
import math
from operator import itemgetter

class Evaluator:
    def __init__(self):
        self.name = "Perplexity Evaluator"
        
    def evaluate_model(self, data, model, label_set):
        """
        calculate confusion matrix for data using model
        model must implement predict method
        """
        confusion_matrix = {}
        for l in label_set:
            counts = {x:0 for x in label_set}
            confusion_matrix[l] = counts          
        for texts, features, labels in data:
            pred_labels = model.predict_viterbi(texts, label_set)
            assert len(labels) == len(pred_labels)
            for i in range(len(labels)):
                label = labels[i]
                pred_label = pred_labels[i]
                confusion_matrix[pred_label][label]+=1
        return confusion_matrix
    
    def print_confusion_matrix(self, confusion_matrix):
        """pretty print CF matrix"""
        key_order = sorted(confusion_matrix.keys())
        print ("\t"+'\t'.join('known_' + x for x in key_order))
        for pred_k in key_order:
            known_k = confusion_matrix[pred_k]
            sorted_known = sorted(known_k.items(), key = itemgetter(0))
            print ('%s\t%s'%('predicted_'+pred_k ,'\t'.join(str(x[1]) for x in sorted_known)))
            
            
    def is_entity(self, label): #hacky!
        return label != "O"
            
    def metrics(self, confusion_matrix):
        """
        prec, recall and fscore according to 
        http://www.cnts.ua.ac.be/conll2002/pdf/15558tjo.pdf    
        """ 
        correct_entities = 0
        pred_entities = 0
        known_entities = 0
        for pred_label, data in confusion_matrix.items():
            for known_label, count in data.items():
                if known_label == pred_label and self.is_entity(known_label):
                    correct_entities += count
                if self.is_entity(known_label):
                    known_entities += count
                if self.is_entity(pred_label):
                    pred_entities += count
        prec = correct_entities/pred_entities
        recall = correct_entities/known_entities
        fscore = 2*prec*recall/(prec + recall)
        return prec, recall, fscore
       
            
    def get_total_unigram(self, sentences):
        unigram_count = 0
        for sentence in sentences:
            unigram_count += len(sentence) - 2 #Not counting START and END
            #TODO: Fix this for UNK
        return unigram_count
    
    
    def get_total_bigram(self, sentences):
        bigram_count = 0
        for sentence in sentences:
            bigram_count += len(sentence) - 1 #Ignore the 1st (None, START)
        return bigram_count
    
    def get_total_trigram(self, sentences):
        trigram_count = 0 
        for sentence in sentences:
            trigram_count += len(sentence) - 2#Ignore the (None, None, Start) & (None, START, W_i)
        return trigram_count
    
    def get_unigram_perplexity(self, model, sentences):
        unigram_count = self.get_total_unigram(sentences)
        sentence_prob_log_sum = 0
        for sentence in sentences:
            sentence_prob_log_sum += model.calculate_sentence_log_probability(sentence)
            """try:
                sentence_prob_log_sum += model.calculate_sentence_log_probability(sentence)
            except:
                # If met a sentence that we haven't seen before, assign infinity
                sentence_prob_log_sum += math.log(2e-20, 2)"""
        return math.pow(2, -1 * sentence_prob_log_sum / unigram_count)
    
    def get_bigram_perplexity(self, model, sentences):
        bigram_count = self.get_total_bigram(sentences)
        bigram_prob_log_sum = 0
        for sentence in sentences:
            bigram_prob_log_sum += model.calculate_bigram_sentence_log_probability(sentence)
            """
            try:
                bigram_prob_log_sum += model.calculate_bigram_sentence_log_probability(sentence)
            except:
                # If met a sentence that we haven't seen before, assign infinity
                bigram_prob_log_sum += float('-inf')
            """
        return math.pow(2, -1 * bigram_prob_log_sum / bigram_count)
    
    def get_trigram_perplexity(self, model, sentences):
        trigram_count = self.get_total_trigram(sentences)
        trigram_prob_log_sum = 0
        for sentence in sentences:
            trigram_prob_log_sum += model.calculate_trigram_sentence_log_probability(sentence)
            """
            try:
                trigram_prob_log_sum += model.calculate_trigram_sentence_log_probability(sentence)
            except:
                # If met a sentence that we haven't seen before, assign infinity
                trigram_prob_log_sum += float('-inf')
            """
        return math.pow(2, -1 * trigram_prob_log_sum / trigram_count)
    
