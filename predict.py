from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import csv
import random
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
import SQLAcess as sqlAccess

#Load Model in data
model = word2vec.Word2Vec.load("med250.model3.bin")

newsList = sqlAccess.GetData("select * from googlenews where stockname like '%2330%' ")
keywordList = []
with open('keywords.csv', newline='', encoding = 'utf-8') as f:		
	reader = csv.reader(f)			
	for row in reader:
		if row == []:
			continue					
		keyword = str(row[0])
				
		try:
			jieba.suggest_freq(keyword, True)
			keywordList.append(keyword)
		except:
			print(keyword)

stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()

instanceList = []
targetList = []
i=0
# jieba custom setting.
jieba.set_dictionary('jieba_dict/dict.txt')

while i<1000:
    featureDict = {}

    for index in range(len(keywordList)):
        featureDict[keywordList[index]] = 0

    content = newsList[i].get('newscontent')

    document_cut = jieba.cut(content)	
    result = ' '.join(document_cut)
    corpus = [result]     
    vector = TfidfVectorizer(stop_words=stpwrdlst)
    tfidf = vector.fit_transform(corpus)
	

    wordlist = vector.get_feature_names()
	
    weightlist = tfidf.toarray()  
	
	
	
    
    
    for k in range(len(wordlist)):
        tempCnt = float(0)
        if wordlist[k] in keywordList:
            try:
                te = model.wv[wordlist[k]]    
            except:
                continue    
            for z in range(len(model.wv[wordlist[k]])):
                try:
                    tempCnt += model.wv[wordlist[k]][z]
                except:
                    break
            print(tempCnt)
            featureDict[wordlist[k]] = float(tempCnt/500)
	            
        else:
	        aa=0
	            
	
    instanceList.append(list(featureDict.values()))
    targetList.append(random.randint(0,1))
    i=i+1

instanceList = np.array(instanceList)
targetList = np.array(targetList)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
y_rbf = svr_rbf.fit(instanceList, targetList).predict(instanceList)

for i in range(len(y_rbf)):
    y_rbf[i] = int(round(y_rbf[i]))

X = []
for i in range(len(targetList)): 
    tmp=[]
    tmp.append(i)
    X.append(tmp)
X = np.array(X)

count = 0
for i in range(len(targetList)):
    if targetList[i] == y_rbf[i]:
        	count = count +1

plt.scatter(X, targetList, c='k', label='data')
plt.plot(X, y_rbf, c='g', label='RBF model')
plt.xlabel('data')
plt.ylabel('target')

plt.title('Support Vector Regression, rate = ' + str( round(float(count)/float(len(targetList)), 2) ))
plt.legend()
plt.show()