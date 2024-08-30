# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:51:42 2024

@author: User 1
"""

import nltk
import numpy
import math

#doing preprocessing of the text
#n - is a number of words in each n-gram
def preprocessing(text, n):
   
    #tokenizing the text:
    tokens = nltk.word_tokenize(text)
    
    #make every word start from lower-case letter:
    tokens = [w.lower() for w in tokens]
    
    #eliminate all non-letter symbols:
    tokens = [w for w in tokens if w.isalpha()]
    
    #deviding text into sequences, of n words: 
    ng_tokens = []
    for i in range(0, len(tokens)-(n-1)):
        n_gram = ''
        for j in range(i, i+n):
            string = (str(n_gram), str(tokens[j]))
            n_gram = ' '.join(filter(None, string)) 
        ng_tokens.append(n_gram)
    
    f = open('tokens.txt', 'w')
    for i in range(0, len(ng_tokens)):
        f.write(str(ng_tokens[i]) + '\n')
    f.close()
    return ng_tokens


#getting vocabulary dictionary of n-gramms based on the tokens
def get_vocab_dict(tokens):
    vocab_list = []
    vocab_list = list(set(tokens))
    vocab_list.sort()
    vocab_dict = dict.fromkeys(vocab_list)
    i = 0
    for w in vocab_dict:
        vocab_dict[w] = i
        i += 1
    return vocab_dict

#getting matrix of frequencies
def get_term_matrix(text1, text2, text3, vocab_dict, output_file):
    matrix = []
    for i in range(0, len(vocab_dict)):
        matrix.append([0] * 3)
    
    #filling the matrix:
    for i in range(0, len(text1)):
        #obtaining the number of row in matrix, which is associated with current word in text 
        row = vocab_dict[text1[i]] 
        #the number of column is fixed for current text (text1)
        matrix[row][0] += 1
        
    for i in range(0, len(text2)):
        row = vocab_dict[text2[i]]
        matrix[row][1] += 1  
    
    for i in range(0, len(text3)):
        row = vocab_dict[text3[i]]
        matrix[row][2] += 1

    #output matrix:
    words = list(vocab_dict.keys())
    f = open(str(output_file), 'w')
    f.write('\t\t\t\t\t\t\t\t  t1  t2  t3\n')
    for i in range(0, len(vocab_dict)):
        f.write("{:<34}".format(words[i]))
        f.write("{:<4}".format(matrix[i][0]))
        f.write("{:<4}".format(matrix[i][1]))
        f.write("{:<4}".format(matrix[i][2]))
        f.write('\n')
    f.close()
    return matrix


#calculating IDF for every word in vocabulary:
def calc_IDFs(matrix, vocab_dict):
    IDFs = []
    for i in range(0, len(matrix)):
        count = 0
        for j in range(0, len(matrix[i])):
            if matrix[i][j] > 0:
                count += 1
                idf = math.log2(3 / count)
        IDFs.append(idf)
    return IDFs


def get_tf_idf_matrix(matrix, vocab_dict):
    tf_idf_matrix = []
    for i in range(0, len(vocab_dict)):
        tf_idf_matrix.append([0] * 3)
        
    #getting IDFs for every word in vocabulary:
    idfs = calc_IDFs(matrix, vocab_dict)
    
    #obtaining "updated" freqs: 
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[i])):
            tf_idf_matrix[i][j] = matrix[i][j] * idfs[i]
    return numpy.array(tf_idf_matrix)
 
   
#cosine similarity
def cos_sim(x,y):
    cos = numpy.dot(x,y) / (numpy.linalg.norm(x) * numpy.linalg.norm(y))
    return round(cos, 5)


#finds term matrix and calculates similarities for model where each..
#..grama consists of n words: 
def particular_n_gramas(text1, text2, text3, n):
    t1 = preprocessing(text1, n)
    t2 = preprocessing(text2, n)
    t3 = preprocessing(text3, n)
    full_text = t1 + t2 + t3
    
    #getting vocabulary:
    vocab_dict = get_vocab_dict(full_text)
    
    #getting term matrix:
    matrix = get_term_matrix(t1, t2, t3, vocab_dict, 'matrix' + str(n) + '.txt')

    #getting tf-idf matrix:
    tf_idf_matrix = get_tf_idf_matrix(matrix, vocab_dict)

    #calculating similarities between texts:
    print("For " + str(n) + '-gramas: \n')
    print("Cosine between text1 and text2 is:  " 
          + str(cos_sim(tf_idf_matrix[:,0], tf_idf_matrix[:,1])))
    print("Cosine between text1 and text3 is:  " 
          + str(cos_sim(tf_idf_matrix[:,0], tf_idf_matrix[:,2])))
    print("Cosine between text2 and text3 is:  " 
          + str(cos_sim(tf_idf_matrix[:,1], tf_idf_matrix[:,2])))
    print()
    print('===========================================\n')


#finds term matrix and calculates similarities for model where..
#..both 1-gramas, 2-gramas and 3-gramas are being considered together
def union_n_gramas(text1, text2, text3):
    t1_1_gramas = preprocessing(text1, 1)
    t1_2_gramas = preprocessing(text1, 2)
    t1_3_gramas = preprocessing(text1, 3) 
    #each text is presented as an union of..
    #..1-gramas, 2-gramas and 3-gramas of this text together:
    t1 = t1_1_gramas + t1_2_gramas + t1_3_gramas

    t2_1_gramas = preprocessing(text2, 1)
    t2_2_gramas = preprocessing(text2, 2)
    t2_3_gramas = preprocessing(text2, 3) 
    t2 = t2_1_gramas + t2_2_gramas + t2_3_gramas

    t3_1_gramas = preprocessing(text3, 1)
    t3_2_gramas = preprocessing(text3, 2)
    t3_3_gramas = preprocessing(text3, 3) 
    t3 = t3_1_gramas + t3_2_gramas + t3_3_gramas

    full_text = t1 + t2 + t3
    
    #getting vocabulary:
    vocab_dict = get_vocab_dict(full_text)
    
    #getting term matrix:
    matrix = get_term_matrix(t1, t2, t3, vocab_dict, 'matrix123.txt')

    #getting tf-idf matrix:
    tf_idf_matrix = get_tf_idf_matrix(matrix, vocab_dict)

    #calculating similarities between texts:
    print("For all gramas together: \n")
    print("Cosine between text1 and text2 is:  " 
          + str(cos_sim(tf_idf_matrix[:,0], tf_idf_matrix[:,1])))
    print("Cosine between text1 and text3 is:  " 
          + str(cos_sim(tf_idf_matrix[:,0], tf_idf_matrix[:,2])))
    print("Cosine between text2 and text3 is:  " 
          + str(cos_sim(tf_idf_matrix[:,1], tf_idf_matrix[:,2])))
    print()
    print('===========================================\n')



def main():
    f = open("text1.txt", 'r')
    text1 = f.read()

    f = open("text2.txt", 'r')
    text2 = f.read()

    f = open("text3.txt", 'r')
    text3 = f.read()
    
    print('\n===========================================\n')
    #getting solution (matrix and sim-s for each demension of gramas separately):
    for i in [1,2,3]:
        particular_n_gramas(text1, text2, text3, i)
    #getting solution for all demensions of gramas together:
    union_n_gramas(text1, text2, text3)
        
main()



