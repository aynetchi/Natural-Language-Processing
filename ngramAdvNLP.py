import os 
import random 
from nltk import word_tokenize, sent_tokenize
import math 
import csv
import numpy as np 
import pandas as pd

pd
class Process_Training_Data:
    
    def __init__(self, holmes_path, max_doc, ratio=0.5, seed=True):
        self.holmes_path = holmes_path 
        self.max_doc = max_doc
        self.file_names = os.listdir(self.holmes_path)
        
        self.ratio = ratio
        self.seed = seed 
        
        self.training, self.held_out = self.split(ratio = self.ratio, seed = self.seed)
        
        self.listed_data = []
        self.process_files()
        
        
    def split(self, ratio, seed):
        if seed:
            random.seed(53)
            #print("Random seed applied")
   
        #print(f"There are {len(self.file_names)} in the following file path: \n{self.holmes_path} folder")
        
        random.shuffle(self.file_names)
        index = int(len(self.file_names) * ratio)

        training = self.file_names[:index]
        held_out = self.file_names[index:]
        return training, held_out
    
    def process_files(self):
        n = 1
        for file in self.training:
            try:
                text = open(os.path.join(self.holmes_path, file))
    
                sentences = sent_tokenize(text.read().replace('\n', ' '))
                tokenized = [["__START"] + word_tokenize(sents.lower()) + ["__END"] for sents in sentences if len(sents)>3]
                self.listed_data += tokenized
                n += 1
                if n > self.max_doc:
                    break
            except UnicodeDecodeError:
                pass
                #print(f"UnicodeDecodeError processing {file}: ignoring rest of this file")
        #print("All files have been processed")


class LanguageModel:
    def __init__(self, splitter, discount=0.75, smoothing=True):
        self.splitter = splitter
        self.discount = discount
        self.smoothing = smoothing
        
        
        self.sentences = self.splitter.listed_data
        
        
        self.find_grams()


        
    def train(self, known):
        self.known = known
        self.unigram = self.make_unknowns(self.unigram_freq, 0)
        self.bigram = self.make_unknowns(self.bigram_freq, 1)
        self.trigram = self.make_unknowns(self.trigram_freq, 2)
        self.quadgram = self.make_unknowns(self.quadgram_freq, 3)
        
        self.bigram, self.kn_bigram = self.kneser_ney(1)
        self.trigram, self.kn_trigram = self.kneser_ney(2)
        self.quadgram, self.kn_quadgram = self.kneser_ney(3)

        self.convert_to_prob() 

    
        

    def find_grams(self):
        self.count_token = {} 
        self.unigram_freq, self.bigram_freq, self.trigram_freq, self.quadgram_freq = {}, {}, {}, {}
        self.kn_bigram, self.kn_trigram, self.kn_quadgram = {}, {}, {}
        for sent in self.sentences:
            length =  len(sent) - 1
            for i, token in enumerate(sent):
                # Adding token counts
                # self.count_token[token] = self.count_token.get(token, 0) + 1
                # Getting Unigram
                self.unigram_freq[token] = self.unigram_freq.get(token, 0) + 1
                # n-grams
                current, keys = self.ngram(1, 0, length, sent, token, i, self.bigram_freq)
                self.bigram_freq[keys] = current
                
                current, keys = self.ngram(1, 1, length, sent, token, i, self.trigram_freq)
                self.trigram_freq[keys] = current
                
                current, keys = self.ngram(1, 2, length, sent, token, i, self.quadgram_freq)
                self.quadgram_freq[keys] = current
                

                
    def return_gram(self, n):
        if n == 0:
            return self.unigram
        elif n == 1:
            return self.bigram
        elif n == 2:
            return self.trigram
        else:
            return self.quadgram
                  

    

    def ngram(self, n, n_1, length, sentence, token, index, gram):

        keys = self.update_previous(n_1, index, sentence)
        index += 1
        current = gram.get(keys, {})
        words = sentence[index: min(index + n, length)]
        for word in words:
            current[word] = current.get(word, 0) + 1

        return current, keys
    
    # function that returns the keys  
    def update_previous(self, n, index, sentence):
        
        multiplier = n - index
        if multiplier < 0:
            return tuple(sentence[index-n: index+1])
        else:
            if index == 0:
                end = n * ["__END"] + [sentence[index]]
                return tuple(end)
            else:
                end = ["__END"] * multiplier + sentence[:index+1]
                return tuple(end)
            
            
        
        
    
    def convert_to_prob(self):
        self.unigram = {k:v/sum(self.unigram.values()) for (k,v) in self.unigram.items()}
        self.bigram = {key:{k:v/max(sum(adict.values()), 0.0000001) for (k,v) in adict.items()} for (key,adict) in self.bigram.items()}
        self.trigram = {key:{k:v/max(sum(adict.values()), 0.0000001) for (k,v) in adict.items()} for (key,adict) in self.trigram.items()}
        self.quadgram = {key:{k:v/max(sum(adict.values()), 0.0000001) for (k,v) in adict.items()} for (key,adict) in self.quadgram.items()}
        
        if self.smoothing:
            self.kn_bigram = {k:v/max(sum(self.kn_bigram.values()), 0.0000001) for (k,v) in self.kn_bigram.items()}
            self.kn_trigram = {k:v/max(sum(self.kn_trigram.values()), 0.0000001) for (k,v) in self.kn_trigram.items()}
            self.kn_quadgram = {k:v/max(sum(self.kn_quadgram.values()), 0.0000001) for (k,v) in self.kn_quadgram.items()}
        
    def get_probability(self, n, token, context="", smoothing=True): 
        #print(tuple(context[-(n+1):]))
        unigram = self.unigram 
        if n > 0:
            grams = [self.bigram, self.trigram, self.quadgram]
            # print(grams)
            unidist = self.kn_bigram if self.smoothing else unigram
            ngram = [gram.get(tuple(context[-(i+1):]), gram.get("__UNK", {})) for i, gram in enumerate(grams)]
            #print(ngram[-1])
            P = [unidist.get(token, unidist.get("__UNK", 0))] + [gram.get(token, gram.get("__UNK", 0)) for gram in ngram]
            lambdas = [gram["__DISCOUNT"] for gram in ngram]
            #print(P[-1], [[f" + {lambdas[i]} * {P[i]}"] for i in range(n)])           
            return P[-1] + sum([P[i] * lambdas[i] for i in range(n)])
        else:
            return unigram.get(token, unigram.get("__UNK", 0))

    def compute_prob_sent(self, tokens, n):
        acc = 0
        for i, token in enumerate(tokens[1:]):
     
            acc += math.log(max(0.000001, self.get_probability(n, token, tokens[:i+1])))
        return acc,len(tokens[1:])
        
    def compute_probability(self, n):        
        total_p = 0
        total_N = 0
        for sent in self.sentences:
            p, N = self.compute_prob_sent(sent, n)
            total_p += p
            total_N += N   
        return total_p,total_N    
    
       
    def compute_perplexity(self, n):
        p, N = self.compute_probability(n)  
        return math.exp(-p/N)
                          

    def kneser_ney(self, n):
        gram_freq =  [self.bigram_freq, self.trigram_freq, self.quadgram_freq]
        gram = {k:{kk:value-self.discount for (kk,value) in adict.items()} for (k,adict) in gram_freq[n-1].items()}
        for k in gram.keys():
            lamb = len(gram[k])
            gram[k]["__DISCOUNT"] = lamb * self.discount
 
            
        kn = {}
        
        for (k,adict) in gram.items():
          for kk in adict.keys():
            kn[kk] = kn.get(kk,0) + 1
            
        return gram, kn     
           
    def make_unknowns(self, gram, n):
        self.number_unknowns = 0
        if n == 0:
            for (k,v) in list(gram.items()):
           
                if v < self.known:
                    del gram[k]
                    gram["__UNK"] = gram.get("__UNK",0) + v
                    self.number_unknowns += 1
            return gram
        else:
            for (k,adict) in list(gram.items()):
                #print("Loop 1", f"k: {k} --- adict: {adict}")
                for (kk,v) in list(adict.items()):
                    #print("Loop 2", f"k: {kk} --- adict: {v}")
                    isknown = self.unigram.get(kk,0)
                    #print(f"isknow: {isknown}")
                    if isknown == 0 and not kk == "__DISCOUNT":
                      adict["__UNK"] = adict.get("__UNK",0) + v
                      #print(adict["__UNK"])
                      del adict[kk]
                
                val = 0 in [self.unigram.get(key, 0) for key in list(k)]
                
                if val:
                    del gram[k]
                    current = gram.get("__UNK",{})
                    current.update(adict)
                    
                    gram["__UNK"] = current
                    
                else:
                    gram[k] = adict 
            return gram
                    
        
class question:
    
    def __init__(self, aline):
        self.fields = aline
        
    def get_field(self,field):
        return self.fields[question.colnames[field]]
  
    def add_answer(self,fields):
        self.answer=fields[1]

    def get_tokens(self):
        return ["__START"] + word_tokenize(self.fields[question.colnames["question"]]) + ["__END"]
    
    
    def get_left_context(self,window=1,target="_____"):
        found = -1
        sent_tokens = self.get_tokens()
        for i,token in enumerate(sent_tokens):
            if token == target:
                found = i
                break  
            
        if found >- 1:
            return sent_tokens[i-window:i]
        else:
            return []

    def get_right_context(self,window=1,target="_____"):
        found = -1
        sent_tokens = self.get_tokens()
        for i,token in enumerate(sent_tokens):
          if token == target:
            found = i
            break  
          
        if found >- 1:
          return sent_tokens[found + 1:found + window + 1]
            
        else:
          return []    
      

    
    
    def choose(self, lm, n):
        rc, lc = self.get_right_context(window = n), self.get_left_context(window = n)
        choices = ["a)","b)","c)","d)","e)"]
        #print()
        #print([self.get_field(q) for q in choices])
        if n == 1:
            # get_probility(n, roken context)
            #print(rc[0])
            probs = [lm.get_probability(n, rc[0], [self.get_field(q)]) * 
                     lm.get_probability(n, self.get_field(q), lc) for q in choices]
        elif n == 2:
            probs = [lm.get_probability(n, self.get_field(q), lc) *
                     lm.get_probability(n, rc[0], [lc[-1]] + [self.get_field(q)]) *
                     lm.get_probability(n, rc[1], [self.get_field(q)] + [rc[0]])
                     for q in choices]
            #print([self.get_field(q) for q in choices])
        elif n == 3:
            probs = [lm.get_probability(n, self.get_field(q), lc) *
                     lm.get_probability(n, rc[0], lc[-2:] + [self.get_field(q)]) *
                     lm.get_probability(n, rc[1], [lc[2]] + [self.get_field(q)] + [rc[0]]) *
                     lm.get_probability(n, rc[2], [self.get_field(q)] + rc[:2])
                     for q in choices]  
        else:
            probs = [lm.get_probability(n, self.get_field(q), lc) for q in choices]
        #print(probs)
        maxprob = max(probs)
        bestchoices = [ch for ch,prob in zip(choices,probs) if prob == maxprob]
    
        return np.random.choice(bestchoices), probs  
    
    def predict(self, lm, n):
        return self.choose(lm, n)
    
    def predict_and_score(self, lm, n):
        prediction, probs =  self.predict(lm, n)
        prediction = prediction[0]
        if prediction == self.answer:
          return 1, prediction, probs
        else:
          return 0, prediction, probs        
        

class SCC:
    def __init__(self, qs, ans):
        
        self.qs = qs 
        self.ans = ans
        self.read_files()
    
    def read_files(self):
        files = []
        
        for file in [self.qs, self.ans]:
            with open(file) as instream:
                line = list(csv.reader(instream))
                #print(line)
                files.append(line)
        
        qlines, alines = files

        #store the column names as a reverse index so they can be used to reference parts of the question
        question.colnames = {item:i for i,item in enumerate(qlines[0])}
        
        #create a question instance for each line of the file (other than heading line)
        self.questions=[question(qline) for qline in qlines[1:]]     
            
        #add answers to questions so predictions can be checked    
        for q,aline in zip(self.questions,alines[1:]):
            q.add_answer(aline)       
                
        
    def get_field(self, field):
        return [q.get_field(field) for q in self.questions] 
      
    def predict(self, n):
        return [q.predict(n) for q in self.questions]
      
    def predict_and_score(self, lm, n):

        predictions = []
        scores = []
        total_probs = []
        for q in self.questions:
          score, pred, probs = q.predict_and_score(lm, n)
          scores.append(score)
          predictions.append(pred)
          total_probs.append(probs)
    
        return sum(scores)/len(scores), predictions, total_probs

def get_data(paths, size, knowns=2):
    holmes_path, questions_file, answers_file = paths
    A = Process_Training_Data(holmes_path, 100, 0.5, seed=True)           
    LM = LanguageModel(A)
    LM.train(knowns)
    data = [SCC(questions_file, answers_file).predict_and_score(LM,n) for n in range(4)]
    predictions = [pred[1] for pred in data]
    probs = [pred[2] for pred in data]
    return predictions, probs

def evaluate(paths, file_sizes, knowns):
    holmes_path, questions_file, answers_file = paths
    accuracies = []
    perplexities = []
    for size in file_sizes:
        A = Process_Training_Data(holmes_path, size, 0.5, seed=True)
        LM = LanguageModel(A)
        for known in knowns:
            LM.train(known)
            acc = tuple([known, size]+[SCC(questions_file, answers_file).predict_and_score(LM,n)[0] for n in range(4)])
            perp = tuple([known, size]+[LM.compute_perplexity(n) for n in range(4)])
            accuracies.append(acc)
            perplexities.append(perp)
            print("Accuracy:", acc)
            print("Perplexity:", perp)
            print()
    
    label_acc = ['Knowns', 'Doc Size', 'Unigram Accuracy','Bigram Accuracy','Trigram Accuracy','Quadgram Accuracy']  
    label_perp = ['Knowns', 'Doc Size', 'Unigram Perplexity','Bigram Perplexity','Trigram Perplexity','Quadgram Perplexity']  
    return pd.DataFrame(accuracies, columns = label_acc), pd.DataFrame(perplexities, columns = label_perp)
    


holmes_path = "/Users/amir/Library/CloudStorage/OneDrive-UniversityofSussex/Advanced NLP/training"
questions_file = "/Users/amir/Library/CloudStorage/OneDrive-UniversityofSussex/Advanced NLP/testing/testing_data.csv"
answers_file = "/Users/amir/Library/CloudStorage/OneDrive-UniversityofSussex/Advanced NLP/testing/test_answer.csv"
paths = holmes_path, questions_file, answers_file

file_sizes = [10, 50, 100]
knowns = [2, 3, 4, 5]
predictions, probabilities = get_data(paths, 100)

#df_acc, df_perp = evaluate(paths, file_sizes, knowns)