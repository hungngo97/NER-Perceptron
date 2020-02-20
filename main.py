from utils.io_utils import (read_conll_file)
from utils.math_utils import *
from constants.model_constants import (UNK, START, STOP)
from constants.constants import (DEV_DATA_FILE, TRAIN_DATA_FILE, TEST_DATA_FILE)
from models.Perceptron import (Perceptron)
from evaluator.evaluator import ( PerplexityEvaluator )

#Read data input
parsed_document_list = read_conll_file(TRAIN_DATA_FILE)
logger.info('Parse completed (%d documents).' % len(parsed_document_list))
# COPY THIS STEP
logger.info('Building word list from train data.')
training_words = parsed_documents_to_words(parsed_document_list)

# Maybe copy the Context Annotator but don't need to until the end
logger.info('Annotate training data.')
annotate_data(training_words)

# No need
logger.info('Convert gold label tag scheme from IOB to BILOU.')
BILOU.encode(training_words, tag_attr='gold_label')

logger.info('Set gold label as tag (for model tag features).')
for word in training_words:
    word.tag = word.gold_label

logger.info('Train model.')
trained_model_params, class_lexicon = train(
    training_words, L1_FEATURES, NUM_TRAIN_ITERATIONS)
save_model(trained_model_params, args.model_file,
           class_lexicon, args.lexicon_file)


sentences = read_sentences_from_file(TRAIN_DATA_FILE)
dev_sentences = read_sentences_from_file(DEV_DATA_FILE)
test_sentences = read_sentences_from_file(TEST_DATA_FILE)

# Training models
unigram_model = UnigramLanguageModel(sentences, K_smoothing=0)
unigram_model.calculate_unigram_probablities("the")
unigram_model.calculate_unigram_probablities("a")
unigram_model.calculate_unigram_probablities("dog")

bigram_model = BigramLanguageModel(sentences, k_smoothing=0)
bigram_model.calculate_bigram_probability("the", "car")
bigram_model.calculate_bigram_probability("the", "dog")
bigram_model.calculate_bigram_probability("the", "boy")


trigram_model = TrigramLanguageModel(sentences, k_smoothing=0)
trigram_model.calculate_trigram_probability("the", "walking" , "car")


linear_model = LinearInterpolationLanguageModel(sentences, 0.3, 0.5, 0.2, 0.5, 0.5, k_smoothing=0)
# Evaluation
print("***************** Without Smoothing **************")
print("========== Train set score evaluation =======")
perplexity_evaluator = PerplexityEvaluator()
print_perplexity_score(
        perplexity_evaluator,sentences,
        unigram_model, bigram_model, trigram_model
        )

print("========== Dev set score evaluation =======")
print_perplexity_score(
        perplexity_evaluator,dev_sentences,
        unigram_model, bigram_model, trigram_model
        )

print("========== Test set score evaluation =======")
print_perplexity_score(
        perplexity_evaluator,test_sentences,
        unigram_model, bigram_model, trigram_model
        )

print("******************* With Smoothing ****************")
for k in [0.001, 0.01, 0.1, 1, 2, 3, 5]:
    unigram_model_smooth = UnigramLanguageModel(sentences, K_smoothing=k)
    bigram_model_smooth = BigramLanguageModel(sentences, k_smoothing=k)
    trigram_model_smooth = TrigramLanguageModel(sentences, k_smoothing=k)
    print("============== K = " + str(k) + " ====================")
    print("========== Train set score evaluation =======")
    perplexity_evaluator = PerplexityEvaluator()
    print_perplexity_score(
            perplexity_evaluator,sentences,
            unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
            )
    
    print("========== Dev set score evaluation =======")
    print_perplexity_score(
            perplexity_evaluator,dev_sentences,
            unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
            )
    
    print("========== Test set score evaluation =======")
    print_perplexity_score(
            perplexity_evaluator,test_sentences,
            unigram_model_smooth, bigram_model_smooth, trigram_model_smooth
            )

print("******************* Linear Interpolation model ****************")
linear_model = LinearInterpolationLanguageModel(sentences, 0.3, 0.3, 0.4, 0.5, 0.5, k_smoothing=0)

print("========== Train set score evaluation =======")
perplexity_evaluator = PerplexityEvaluator()
print(perplexity_evaluator.get_trigram_perplexity(linear_model, sentences))
print("========== Dev set score evaluation =======")
print(perplexity_evaluator.get_trigram_perplexity(linear_model, dev_sentences))

print("========== Test set score evaluation =======")
print(perplexity_evaluator.get_trigram_perplexity(linear_model, test_sentences))
