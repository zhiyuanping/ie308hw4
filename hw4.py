import numpy as np  
import matplotlib.pyplot as plt 
import glob 
import pandas as pd  
import re
import regex
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords 
import string
from collections import defaultdict

##load all emails
df = []
path1 = '/Users/daniel/Desktop/School_Work/IE/308/hw4/2013/*.txt'
files2013 = glob.glob(path1)
# iterate over the list getting each file 
for file in files2013:
   # open the file and then call .read() to get the text 
   with open(file, encoding="utf8", errors='ignore') as f:
      contents = f.read()
      df.append(contents)
path2 = '/Users/daniel/Desktop/School_Work/IE/308/hw4/2014/*.txt'
files2014 = glob.glob(path2)
# iterate over the list getting each file 
for file in files2014:
   # open the file and then call .read() to get the text 
   with open(file, encoding="utf8", errors='ignore') as f:
      contents = f.read()
      df.append(contents)

top_word = []
    
#build hashmap
def most_common(lst):
    return max(set(lst), key=lst.count)

hm = defaultdict(list)
stop_words = set(stopwords.words('english')) 
tl = 0
for i in range(len(df)):
    tokens = word_tokenize(df[i])
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if (word.isalpha()|word.isdigit())]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    tl = tl+len(words)
    most = most_common(words)
    max_count = words.count(most)
    top_word.append(max_count)
    for w in words:
        if (i not in hm[w]):
            hm[w].append(i)

avdl = tl/730
#build questions
bankr = hm["bankrupt"]
for b in bankr:
    print(df[b])

ceo = hm["ceo"]
for b in ceo:
    print(df[b])         
            
q = []
q.append("Which companies went bankrupt in month September of year 2008?")
q.append("Which companies went bankrupt in year 2009?")
q.append("Which companies went bankrupt in year 2010?")

q.append("What affects GDP?")
q.append("What percentage of drop or increase is associated with unemployment?")
q.append("What affects GDP?")
q.append("What percentage of drop or increase is associated with interest rate?")
q.append("What affects GDP?")
q.append("What percentage of drop or increase is associated with government spending?")

q.append("Who is the CEO of Actavis?")
q.append("Who is the CEO of Apple?")
q.append("Who is the CEO of BlackRock?")

qd = pd.DataFrame(q)
path='/Users/daniel/Desktop/School_Work/IE/308/hw4/'
filename = "questions.csv"
#qd.to_csv(path+filename)
    
#extract key words
noun = ["NN","NNP","NNS","NNPS"]
whi = ["NNP","NNPS","RB","CD"]
wha = ["NN","NNP","NNS","NNPS","VB","VBZ","VBD","VBN"]
who = ["NN","NNP","NNS","NNPS","VBD","VBZ"]
def extract_key(q):
    key = []
    n = re.findall(r'"(.*?)"',q)
    for i in range(len(n)):
        key.append(n[i-1].lower())
        
    text = word_tokenize(q)
    if "which" in text or "Which" in text:
        a = 1
    elif "what" in text or "What" in text:
        a = 2
    elif "who" in text or "Who" in text:
        a = 3
    # remove punctuation from each word    
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in text]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if (word.isalpha()|word.isdigit())]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    tagged = nltk.pos_tag(words)
    print(tagged)
    if a == 1:
        print("This is a which question")
        for j in range(len(tagged)):
            if ((tagged[j][1] == "JJ")&(j!=len(tagged)-1)):
                if (tagged[j+1][1] in noun):
                    key.append(tagged[j][0])
            elif (tagged[j][1] in whi):
                key.append(tagged[j][0])
    elif a == 2:
        print("This is a what question")
        for j in range(len(tagged)):
            if ((tagged[j][1] == "JJ")&(j!=len(tagged)-1)):
                if (tagged[j+1][1] in noun):
                    key.append(tagged[j][0])
            elif (tagged[j][1] in wha):
                key.append(tagged[j][0])
    elif a == 3:
        print("This is a who question")
        for j in range(len(tagged)):
            if ((tagged[j][1] == "JJ")&(j!=len(tagged)-1)):
                if (tagged[j+1][1] in noun):
                    key.append(tagged[j][0])
            elif (tagged[j][1] in who):
                key.append(tagged[j][0])
    else:
        key = extract_key2(q)
    # convert to lower case
    key = [w.lower() for w in key]
    key = [w for w in key if not w in stop_words]
    key = list(set(key))
    return(list(key))

def extract_key2(q):
    tokens = word_tokenize(q)
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word    
    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in words if (word.isalpha()|word.isdigit())]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return(words)
    
#select sentes
def count_common(list1,list2):
    return len(list(set(list1).intersection(list2)))

def select_sent(keywords,sent,hm_temp,top_word_temp,avdl,q):
    #okapi formula
    s = []
    for i in range(len(keywords)):
        keyword = keywords[i]
        l = hm_temp[keyword]
        for j in range(len(l)):
            if (l[j] not in s):
                s.append(l[j])
    print("There are {} candidate sentences".format(len(s)))
    score = []
    q_bigram = [b for l in q for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    for k in range(len(s)):
        senten = sent[s[k]]
        s_bigram = [b for l in senten for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
        sc = count_common(q_bigram,s_bigram)
        k3 = 0.5
        qtf = 1
        for kk in range(len(keywords)):
            kw = keywords[kk]
            d = s[k]
            v_idf = idf(kw,d,hm_temp)
            v_otf = otf(kw,d,sent,avdl)
            sc = sc+v_idf*v_otf*((k3+1)*qtf)/(k3+qtf)
            #tfidf
            #kw = keywords[kk]
            #v_tfidf = tfidf(kw,s[k],sent,hm_temp,top_word_temp)
            #sc = sc+v_tfidf
        score.append(sc)
    ans = []
    print(score)
    nn = len(s)
    if nn < 10:
        rr = nn
    else:
        rr = 10
    for i in range(rr):
        ind = score.index(max(score))
        ans.append(s[ind])
        score.remove(max(score))
        s.remove(s[ind])
    return ans
#idftf

def tfidf(w,n,sent,hm_temp,top_word_temp):
    w = w.lower()
    table = str.maketrans('', '', string.punctuation)
    w = w.translate(table)
    
    tokens = word_tokenize(sent[n])
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word    
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if (word.isalpha()|word.isdigit())]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    
    tf = words.count(w)
    max_count = top_word_temp[n]
    all_text = hm_temp[w]
    df = len(all_text)
    value = (0.5+0.5*tf/max_count)*np.log(len(sent)/df)
    return value

def otf(w,n,df,avdl):
    k1 = 0.5
    b = 0.5
    w = w.lower()
    table = str.maketrans('', '', string.punctuation)
    w = w.translate(table)
    
    tokens = word_tokenize(df[n])
    # convert to lower case
    tokens = [token.lower() for token in tokens]
    # remove punctuation from each word    
    stripped = [token.translate(table) for token in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    
    tf = words.count(w)
    dl = len(words)
    s = ((k1+1)*tf)/((k1*(1-b)+b*dl/avdl)+tf)
    return s

def idf(w,n,hm):
    w = w.lower()
    table = str.maketrans('', '', string.punctuation)
    w = w.translate(table)
    all_text = hm[w]
    df = len(all_text)
    #print(w)
    #print(freq)
    #print(max_count)
    #print(num)
    #s = (0.5+0.5*freq/max_count)*np.log(730/num)
    return np.log((730-df+0.5)/(df+0.5))
    
#select documents
def select_doc(keywords):
    #okapi formula
    docs = []
    docss = []
    for i in range(len(keywords)):
        keyword = keywords[i]
        l = hm[keyword]
        docss.append(l)
        for j in range(len(l)):
            if (l[j] not in docs):
                docs.append(l[j])
    for i in range(len(docss)):
        docs = list(set(docs).intersection(docss[i]))
    print("There are {} candidate docs".format(len(docs)))
    score = []
    for k in range(len(docs)):
        print(k)
        s = 0
        k3 = 0.5
        qtf = 1
        for kk in range(len(keywords)):
            kw = keywords[kk]
            d = docs[k]
            v_idf = idf(kw,d,hm)
            v_otf = otf(kw,d,df,avdl)
            s = s+v_idf*v_otf*((k3+1)*qtf)/(k3+qtf)
        score.append(s)
    doc = []
    nn = len(docs)
    if nn < 20:
        rr = nn
    else:
        rr = 20
    for i in range(rr):
        ind = score.index(max(score))
        doc.append(docs[ind])
        score.remove(max(score))
        docs.remove(docs[ind])
    return doc
        
#extract key words
qs = pd.read_csv("/Users/daniel/Desktop/School_Work/IE/308/hw4/questions.csv",header=None)
articles = []
for i in range(len(qs)):
    q = qs.loc[i][0]
    print(q)
    keywords = extract_key(q)
    print("The key words are:")
    print(keywords)
    doc = select_doc(keywords)
    print("The doc selected is: {}".format(doc))
    articles.append(doc)
    print("----------------------")
            
art = pd.DataFrame(articles)
path='/Users/daniel/Desktop/School_Work/IE/308/hw4/'
filename = "articles.csv"
art.to_csv(path+filename)
########

qs = pd.read_csv("/Users/daniel/Desktop/School_Work/IE/308/hw4/questions.csv",header=None)
articles = pd.read_csv("/Users/daniel/Desktop/School_Work/IE/308/hw4/articles.csv",header=None)
answers = []
for i in range(len(qs)):
    q = qs.loc[i][0]
    print(q)
    keywords = extract_key(q)
    print("The key words are:")
    print(keywords)
    doc = articles[i]
    print("The docs selected are: {}".format(doc))
    sent = []
    for i in range(len(doc)):
        ss = sent_tokenize(df[doc[i]])
        for s in ss:
            sent.append(s)
    #build hash map
    hm_temp = defaultdict(list)
    top_word_temp = []
    tl_temp = 0
    for i in range(len(sent)):
        tokens = word_tokenize(sent[i])
        # convert to lower case
        tokens = [w.lower() for w in tokens]
        # remove punctuation from each word    
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in tokens]
        # remove remaining tokens that are not alphabetic
        words = [word for word in stripped if (word.isalpha()|word.isdigit())]
        # filter out stop words
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        tl_temp = tl_temp + len(words)
        if (len(words) < 1):
            max_count = 1
        else:
            most = most_common(words)
            max_count = words.count(most)
        top_word_temp.append(max_count)
        for w in words:
            if (i not in hm[w]):
                hm_temp[w].append(i)
    avdl_temp = tl_temp/len(sent)
    sents = select_sent(keywords,sent,hm_temp,top_word_temp,avdl_temp,q)
    print("The sentences selected are: {}".format(sents))
    answer = []
    for i in range(len(sents)):
        answer.append(sent[sents[i]])
    print("The top sentence is {}".format(answer[0]))
    answers.append(answer)        
    print("----------------------")
    
answers_pd = pd.DataFrame(answers)
path='/Users/daniel/Desktop/School_Work/IE/308/hw4/'
filename = "answers.csv"
answers_pd.to_csv(path+filename)

