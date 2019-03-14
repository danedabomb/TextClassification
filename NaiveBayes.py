import sys
import getopt
import os
import math
import operator
import re
from functools import reduce
import collections

class NaiveBayes:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.dev and self.test. 
        """
        def __init__(self):
            self.train = []
            self.dev = []
            self.test = []

    class Example:
        """Represents a document with a label. klass is 'aid' or 'not' by convention.
             words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []

    def __init__(self):
        """NaiveBayes initialization"""
        self.FILTER_STOP_WORDS = False
        self.USE_BIGRAMS = False
        self.BEST_MODEL = False
        self.stopList = set(self.readFile('data/english.stop'))
        #TODO: add other data structures needed in classify() and/or addExample() below
        #count (w_i, c_j)
        self.counts_class = collections.defaultdict(lambda: 0)
        #count the total number of words in each class
        self.words_per_class = collections.defaultdict(lambda: 0)
        #count total number of all classes
        self.total = 0
        #count the vocab of all words in all classes
        self.unique_vocab = collections.defaultdict(lambda: 0)
        # holds each class
        self.classes = collections.defaultdict(lambda: 0)      
        

    #############################################################################
    # TODO TODO TODO TODO TODO 
    # Implement the Multinomial Naive Bayes classifier with add-1 smoothing
    # If the FILTER_STOP_WORDS flag is true, you must remove stop words
    # If the USE_BIGRAMS flag is true, your methods must use bigram features instead of the usual 
    # bag-of-words (unigrams)
    # If either of the FILTER_STOP_WORDS or USE_BIGRAMS flags is on, the other is meant to be off. 
    # Hint: Use filterStopWords(words) defined below
    # Hint: Remember to add start and end tokens in the bigram implementation
    # Hint: When doing add-1 smoothing with bigrams, V = # unique bigrams in data. 

    def classify(self, words):
        """ TODO
            'words' is a list of words to classify. Return 'aid' or 'not' classification.
        """

        #calculate priors for 'aid' and 'not'
        prob_prior_aid = self.classes['aid'] / self.total
        prob_prior_not = self.classes['not'] / self.total

        #initialize probabilities
        prob_aid = 0
        prob_not = 0

        if self.USE_BIGRAMS == False: #no b flag, follow ch. 4 algorithm
            for word in words:
                if word not in self.unique_vocab:
                    continue
                else:
                    type = 'aid'
                    count_word_class = self.counts_class[(word, type)]
                    total_class = self.words_per_class[type]
                    prob_aid += math.log((count_word_class + 1) / (total_class + len(self.unique_vocab)))
        
                    type = 'not'
                    count_word_class = self.counts_class[(word, type)]
                    total_class = self.words_per_class[type]
                    prob_not += math.log((count_word_class + 1) / (total_class + len(self.unique_vocab)))

        if self.USE_BIGRAMS == True: #b flag, follow ch. 4 algorithm
            words.insert(0, "<s>") #insert '<s>' at beginning of words array
            words.append("</s>") #append '</s>' at end of words array
            prev_word = None
            for word in words:
                if prev_word == None:
                    prev_word = word
                else:
                    if (prev_word, word) in self.unique_vocab:
                        type = 'aid'
                        count_word_class = self.counts_class[(prev_word, word, type)]
                        total_class = self.words_per_class[type]
                        prob_aid += math.log((count_word_class + 1) / (total_class + len(self.unique_vocab)))
        
                        type = 'not'
                        count_word_class = self.counts_class[(prev_word, word, type)]
                        total_class = self.words_per_class[type]
                        prob_not += math.log((count_word_class + 1) / (total_class + len(self.unique_vocab)))
                        prev_word = word
                    else:
                        prev_word = word #remembered to move to next word before continuing (FINAL FIX! :))
                        continue

        #add log priors
        prob_aid += math.log(prob_prior_aid)
        prob_not += math.log(prob_prior_not) 

        #return proper classification
        if prob_aid > prob_not:
            return 'aid'
        else:
            return 'not'


    def addExample(self, klass, words):
        """
         * TODO
         * Train your model on an example document with label klass ('aid' or 'not') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier 
         * in the NaiveBayes class.
         * Returns nothing
        """
        
        self.classes[klass] += 1 #holds classes ('aid' and 'not')
        self.total += 1 #counts total D for use in prior probabilities for 'aid' and 'not'
        
        if self.FILTER_STOP_WORDS == True: #f flag
            words = self.filterStopWords(words) #filter out stop words

        if self.USE_BIGRAMS == True: #b flag
            words.insert(0, "<s>") #insert '<s>' at beginning of words array
            words.append("</s>") #append '</s>' at end of words array
            prev_word = None
            for word in words:
                if prev_word == None:
                    prev_word = word
                else:
                    self.unique_vocab[(prev_word, word)] += 0
                    self.counts_class[(prev_word, word, klass)] += 1
                    self.words_per_class[klass] += 1
                    prev_word = word

        if self.USE_BIGRAMS == False: #either no flags or f flag
            for word in words:
                self.unique_vocab[word] += 0
                self.counts_class[(word, klass)] += 1
                self.words_per_class[klass] += 1

        pass
        
    # END TODO (Modify code beyond here with caution)
    #############################################################################
    
    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here, 
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName,encoding="utf8")
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents)) 
        return result

    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

    def buildSplit(self,include_test=True):
    
        split = self.TrainSplit()
        datasets = ['train','dev']
        if include_test:
            datasets.append('test')
        for dataset in datasets:
            for klass in ['aid','not']:
                dataFile = os.path.join('data',dataset,klass + '.txt')
                with open(dataFile,'r', encoding="utf8") as f:
                    docs = [line.rstrip('\n') for line in f]
                    for doc in docs:
                        example = self.Example()
                        example.words = doc.split()
                        example.klass = klass
                        if dataset == 'train':
                            split.train.append(example)
                        elif dataset == 'dev':
                            split.dev.append(example)
                        else:
                            split.test.append(example)
        return split


    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered
    
def evaluate(FILTER_STOP_WORDS,USE_BIGRAMS):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.USE_BIGRAMS = USE_BIGRAMS
    split = classifier.buildSplit(include_test=False)
   
    for example in split.train:
        classifier.addExample(example.klass,example.words)

    train_accuracy = calculate_accuracy(split.train,classifier)
    dev_accuracy = calculate_accuracy(split.dev,classifier)

    print('Train Accuracy: {}'.format(train_accuracy))
    print('Dev Accuracy: {}'.format(dev_accuracy))


def calculate_accuracy(dataset,classifier):
    acc = 0.0
    if len(dataset) == 0:
        return 0.0
    else:
        for example in dataset:
            guess = classifier.classify(example.words)
            if example.klass == guess:
                acc += 1.0
        return acc / len(dataset)

        
def main():
    FILTER_STOP_WORDS = False
    USE_BIGRAMS = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fb')
    if ('-f','') in options:
      FILTER_STOP_WORDS = True
    elif ('-b','') in options:
      USE_BIGRAMS = True

    evaluate(FILTER_STOP_WORDS,USE_BIGRAMS)

if __name__ == "__main__":
        main()
