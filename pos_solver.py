###################################
# CS B551 Spring 2021, Assignment #3
#
# Your names and user ids:
#
# (Based on skeleton code by D. Crandall)
#


import random
import math
import copy

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.

emission = {}
transition = {}
prior = {}
prior_prob = {}

class Solver:
   
   #emission dictionary 
    def emission_func(self,words,pos):
        global emission
        for word, partos in zip(words, pos):
            if word in emission:
                if partos in emission[word]:
                    emission[word][partos]= emission[word][partos] + 1
                else:
                    emission[word][partos] = 1    
            else:
                emission[word] = {partos : 1}         

    def word_dict(self,pos):
        global prior
        for partos in pos:
            if partos in prior:
                prior[partos] = prior[partos] + 1
            else:
                prior[partos] = 1
   

    def cal_prior_prob(self):
        global prior_prob
        global prior
        for items in prior.keys():
            prior_prob[items] = float(prior[items])/float(sum(prior.values()))
        return 
        
    #transition dictionary
    def transition_dict(self,words, pos_list):
        global transition
        pos_initial = pos_list[0]
        pos_list = pos_list[1:]
        for pos in pos_list:
            if pos in transition:
                if pos_initial in transition[pos]: 
                    transition[pos][pos_initial]= transition[pos][pos_initial] + 1
                else:
                    transition[pos][pos_initial] = 1    
            else: 
                transition[pos] = {pos_initial : 1} 
            
            pos_initial = pos
                      

    def calc_simple_postprob(self, sentence, label):
        words = list(sentence)
        pos_list = list(label)
        p = 0
        for word, pos in zip(words, pos_list):
            if word in emission:
                if pos in emission[word]:
                    p += math.log(emission[word][pos]) -  math.log(sum(prior.values())) + math.log(prior_prob[pos])
        return p


    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!

    def posterior(self, model, sentence, label):
        if model == "Simple":
            return self.calc_simple_postprob(sentence, label)

        elif model == "HMM":
            value  = 0
            for i in range(len(sentence)):
                if sentence[i] in emission.keys():
                    if label[i] in emission[sentence[i]].keys():
                        emission_prob = float(emission[sentence[i]][label[i]])/sum(prior.values())
                    else:
                        emission_prob = 1/1000000000000000000000000
                else:
                    emission_prob = 1/1000000000000000000000000
                
                if i == 0 :
                    value = value + math.log10(emission_prob) + math.log10(prior_prob[label[i]])
                else:
                    if label[i] in transition.keys():
                        if label[i-1] in transition[label[i]].keys():
                            transition_prob = float(transition[label[i]][label[i-1]])/sum(prior.values())
                        else:
                            transition_prob = 1/1000000000000000000000000
                    else:
                        transition_prob = 1/1000000000000000000000000

                    value = value + math.log10(emission_prob) + math.log10(transition_prob)
                
            return value

        elif model == "Complex":
            value  = 0
            for i in range(len(sentence)):
                if sentence[i] in emission.keys():
                    if label[i] in emission[sentence[i]].keys():
                        emission_prob = float(emission[sentence[i]][label[i]])/sum(prior.values())
                    else:
                        emission_prob = 1/1000000000000000000000000
                else:
                    emission_prob = 1/1000000000000000000000000
                
                if i == 0 :
                    value = value + math.log10(emission_prob) + math.log10(prior_prob[label[i]])
                elif i == 1:
                    if label[i] in transition.keys():
                        if label[i-1] in transition[label[i]].keys():
                            transition_prob = float(transition[label[i]][label[i-1]])/sum(prior.values())
                        else:
                            transition_prob = 1/1000000000000000000000000
                    else:
                        transition_prob = 1/1000000000000000000000000

                    value = value + math.log10(emission_prob) + math.log10(transition_prob)
                else :
                    if label[i] in transition.keys():
                        if label[i-1] in transition[label[i]].keys():
                            transition_prob1 = float(transition[label[i]][label[i-1]])/sum(prior.values())
                        else:
                            transition_prob1 = 1/1000000000000000000000000
                    else:
                        transition_prob1 = 1/1000000000000000000000000

                    if label[i] in transition.keys():
                        if label[i-2] in transition[label[i]].keys():
                            transition_prob2 = float(transition[label[i]][label[i-2]])/sum(prior.values())
                        else:
                            transition_prob2 = 1/1000000000000000000000000
                    else:
                        transition_prob2 = 1/1000000000000000000000000

                    value = value + math.log10(emission_prob) + math.log10(transition_prob1) + math.log10(transition_prob2)
                
            return value

        else:
            print("Unknown algo!")


    # Do the training!
    def train(self, data):
        words = []
        parts_of_speech = []
        sum_words = 0
        sum_pos  = 0
        for i in range(len(data)):
            words = list(data[i][0])
            parts_of_speech  =  list(data[i][1])
            sum_words += len(words)
            sum_pos += len(parts_of_speech)
            self.emission_func(words,parts_of_speech)
            self.word_dict(parts_of_speech) 
            self.transition_dict(words, parts_of_speech)
        
        self.cal_prior_prob()   
        # print(transition,'tranistionnnn')
        return
    
    #  Functions for each algorithm. Right now this just returns nouns -- fix this!

    def simplified(self, sentence):
        pos_list = ['noun', 'adv', 'verb', 'adp', 'prt', 'det', '.', 'pron', 'num', 'x', 'conj', 'adj']
        words = list(sentence)
        predict_final = []
        for word in words:
            predict_list = []
            if word in emission: 
                for pos in pos_list: 
                    if pos in emission[word]:
                        emission_prob = float(emission[word][pos])/sum(prior.values())
                        predict_list.append(emission_prob * prior_prob[pos])
                    else: 
                        predict_list.append(0.0000000000000000000000000000000000000001*prior_prob[pos])

                max_prob = max(predict_list)
                max_index = predict_list.index(max_prob)
                predict_final.append(pos_list[max_index])

            else: 
                predict_final.append(random.choice(pos_list))

        return predict_final

# Viterbi
#Reference taken from DJ Crandall's in-class activity
    def hmm_viterbi(self, sentence):
        N = len(sentence)
        words = list(sentence)
        return_lst = [ "noun" ] * len(sentence)
        states = ['noun', 'adv', 'verb', 'adp', 'prt', 'det', '.', 'pron', 'num', 'x', 'conj', 'adj']
        V_table = {"noun": [0] * N, "adv" : [0] * N, "verb": [0] * N, "adp": [0] * N, "prt": [0] * N, "det": [0] * N, ".": [0] * N,"pron": [0] * N, "num": [0] * N, "x": [0] * N,"conj": [0] * N, "adj": [0] * N}
        which_table = {"noun": [0] * N, "adv" : [0] * N, "verb": [0] * N, "adp": [0] * N, "prt": [0] * N, "det": [0] * N, ".": [0] * N,"pron": [0] * N, "num": [0] * N, "x": [0] * N,"conj": [0] * N, "adj": [0] * N}

        value = 1/1000000000000000000000000
         
        first_word = words[0]
        for s in states:
            if first_word in emission:
                if s in emission[first_word]:
                    emission_prob = float(emission[first_word][s])/sum(prior.values())
                    V_table[s][0] = emission_prob * prior_prob[s]
                else:
                    V_table[s][0] = prior_prob[s] * value
            else:
                V_table[s][0] = prior_prob[s] * value     

        for i in range(1, N):
            for s in states:
                (which_table[s][i], V_table[s][i]) =  max( [ (s0, V_table[s0][i-1] * transition[s][s0] / sum(prior.values())) if ( s0 in transition[s].keys()) else (s0, V_table[s0][i-1] * value) if (s in transition.keys())  else (s0, value * value) for s0 in states ], key=lambda l:l[1] ) 

                if words[i] in emission.keys():
                    if s in emission[words[i]].keys():
                        V_table[s][i] *= float(emission[words[i]][s])/sum(prior.values())
                    else:
                        V_table[s][i] *= value    
                else:
                    V_table[s][i] *= value
        
        maxValue = 0
        for s0 in states:
            if maxValue <  V_table[s0][N-1]:
                maxValue = V_table[s0][N-1]
                return_lst[N-1] = s0
        for i in range(N-2, -1, -1):
            return_lst[i] = which_table[return_lst[i+1]][i+1]
        return return_lst

#Gibbs Sampling 
# Discussed the approach with Mansi Jain (mj117@iu.edu)
    def complex_mcmc(self, sentence):
        N = len(sentence)
        samples = [[] for i in range (1000)]
        samples[0] = ["noun"] * N 
        samples = self.generate_sample(sentence, samples)
        predicted_results = [] * N
        pos_count={}
        for words in sentence:
            pos_count[words]={"det":0,"noun":0,"adj":0,"verb":0,"adp":0,".":0,"adv":0,"conj":0,"prt":0,"pron":0,"num":0,"x":0}
            
        for i in range(50,len(samples)):
            for j in range(N):
                word=sentence[j]
                pos_count[word][samples[i][j]] = pos_count[word][samples[i][j]] + 1
        
        for i in sentence:
            predicted_results.append(max(pos_count[i],key=pos_count[i].get))
        return predicted_results

    def generate_sample(self, sentence, samples):
        sentence_len = len(sentence)
        states = ['noun', 'adv', 'verb', 'adp', 'prt', 'det', '.', 'pron', 'num', 'x', 'conj', 'adj']
        for i in range(1,1000):
            samples[i] = copy.deepcopy(samples[i-1])
            for index in range(sentence_len):
                word = sentence[index]
                prob = [0] * len(states)
                if index > 1 :
                    tag0 = samples[i][index - 2]
                else :
                    tag0 = " "
                if index > 0 :
                    tag1 = samples[i][index - 1] 
                else :
                    tag1 = " "

                maxValue = - float("inf")

                for j in range(12): 
                    current_tag = states[j]
                    if word in emission.keys():
                        if current_tag in emission[word].keys():
                            emission_prob = float(emission[word][current_tag])/sum(prior.values())
                        else:
                            emission_prob = .000000001
                    else:
                        emission_prob = .000000001
                    
                    if current_tag in transition.keys():
                        if tag1 in transition[current_tag].keys():
                            tag1_current = float(transition[current_tag][tag1])/sum(prior.values())
                        else:
                            tag1_current = .000000001
                    else:
                        tag1_current = .000000001
                    
                    if current_tag in transition.keys():
                        if tag0 in transition[current_tag].keys():
                            tag0_current = float(transition[current_tag][tag0])/sum(prior.values())
                        else:
                            tag0_current = .000000001
                    else:
                        tag0_current = .000000001                

                    if index == 0:
                        prob[j] = emission_prob * prior_prob[current_tag]
                    elif index == 1:
                        prob[j] = emission_prob * tag1_current
                    else:
                        prob[j] = emission_prob * tag0_current * tag1_current
                    

                    if maxValue < prob[j]:
                        maxValue = prob[j]
                        samples[i][index] = current_tag

        return samples



    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        else:
            print("Unknown algo!")