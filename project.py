import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import csv
import statistics as st
import math
import numpy as np
import operator
np.set_printoptions(suppress=True)

url = "data/bankloans.csv"


df = pd.read_csv(url, names=['age','ed','employ','address','income','debtinc','creddebt','othdebt','default'])
#csv dosyamıza ait özellikler
feature = ['age','ed','employ','address','income','debtinc','creddebt','othdebt']

#feature aıt veriler
x = df.loc[:,feature]

#sonuc verilerimiz
y = df.loc[:,'default']


x = StandardScaler().fit_transform(x)
#-------------------PCA-----------------------------
print("--------------------")
print("---------PCA--------")
print("--------------------")
#kac özellik secılecegıne karar verılır
pca = PCA(n_components=3)

pct = pca.fit_transform(x)

principal_df = pd.DataFrame(pct,columns=['yeniOzellik1','yeniOzellik2','yeniOzellik3'])

finaldf= pd.concat([principal_df,df[['default']]],axis=1)
print(finaldf)





#-----------------Bayes-----------------
print("--------------------")
print("--------BAYES-------")
print("--------------------")


def loadCsv(filename):
    lines = csv.reader(open(filename, "r"));
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return  dataset

def calprob(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev * stdev)) * exponent

dataset = loadCsv(url)
size = 0.20
egitimList = []  # training set initialized as an empty list
testList = []  # testing set initialized as an empty list

# splitting the dataset into training and testing
for i in range(int(len(dataset) * size)):
    testList.append(dataset[i])
for i in range(int(len(dataset) * size + 1), len(dataset)):
    egitimList.append(dataset[i])
print('The lenth of the training set', len(egitimList))
print('The length of the testing set', len(testList))

classes = []
for i in dataset:
    if (i[-1] not in classes):
        classes.append(i[-1])
classdict = {}
classdict1 = {} #verilere ait ortalama ve standart sapma bilgileri tutulur
classprob = {} # tahminlerin tutulduğu liste
# initialization
for i in classes:
    classdict[i] = []
    classdict1[i] = []
    classprob[i] = 1

#sonuçumuz 2 secenekli olduğu için 2 boyutlu dizi oluşturuldu
for i in classes:
    for row in egitimList:
        if row[-1] == i:
            classdict[i].append(row[:-1])


for sınıfDegeri, oznitelik in classdict.items():
    for col in zip(*oznitelik):
    #veriler sonuc degerlerıne gore ayrılır. 2 sonuc bulunmakta verı setınde bu durumda verılerımız 0 ve 1 dıye 2 gruba ayrılır.
        classdict1[sınıfDegeri].append((st.mean(col), st.stdev(col)))
        #st.mean() fonksiyonu ortalama hesaplaması yapar.İlgili verilerin ortalamasını hesaplar.
        #st.stdev() fonksiyonu standart sapma hesaplaması yapar.
        #hesaplamları tuttugumuz dızıne atar

count = 0 # dogru tahmın sayısı
for row in testList: #test verileri için tahmın yapılır
    for i in classes:
        classprob[i] = 1
    for sınıfDegeri, oznitelik in classdict1.items():
    #her sınıfa ait veriler hesaplanır
        for i in range(len(row[:-1])):
            mean, std = oznitelik[i]
            x = row[i]
            classprob[sınıfDegeri] *= calprob(x, mean, std)  # bayes teoremi uygulanır
    print(row, "test verisi için sonuçlar ", classprob)
    # başarı oranı
    mini = 0
    cl = 0
    for c, d in classprob.items():
        if d > mini:
            mini = d
            cl = c

    if row[-1] == cl:
        count += 1

acc = count / len(testList)
print("Başarı oranı", acc)



#--------KNN--------
print("--------------------")
print("---------KNN--------")
print("--------------------")

traingSet = pd.read_csv("data/knn-bankloans.csv")
print(traingSet.head())


def euclideanDistance(testInstance, trainingSet, length):
    distance = 0
    for x in range(length):
        distance += np.square(testInstance[x] - trainingSet[x])
    return np.sqrt(distance)

def knn(trainingSet, testInstance, k):
    distances = {}
    length = testInstance.shape[1]
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
    sortdist = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(sortdist[x][0])
    Count = {}
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        if response in Count:
            Count[response] += 1
        else:
            Count[response] = 1
    sortcount = sorted(Count.items(), key=operator.itemgetter(1), reverse=True)
    return (sortcount[0][0], neighbors)

# making test data set
testSet = [[41,3,0,21,26,1.7,0.099008,0.342992,0],
           [23,1,3,0,19,8.4,0.2793,1.3167,1],
           [43,1,13,23,76,6.1,2.151104,2.484896,1],
           [30,2,4,4,21,18.3,0.491904,3.351096,0],
           [38,2,2,16,22,22.7,1.208548,3.785452,1],
           [38,1,0,8,23,8.1,0.897966,0.965034,1],
           [38,2,4,13,56,1.6,0.441728,0.454272,0],
           [31,1,2,2,22,16.4,1.739056,1.868944,1],
           [35,2,3,11,40,17.2,1.80256,5.07744,1],
           [42,2,11,5,73,6.8,2.869192,2.094808,0],]

for i in range(len(testSet)):
    print(i+1,". test verisi => ",testSet[i])
while(True):
    print("\n1 ile 10 arasında değer gir")
    testVerisi=input("Test Edilecek Veriyi Seçiniz => ")
    if int(testVerisi)>0 and int(testVerisi)<11:
        test = pd.DataFrame(testSet[int(testVerisi)-1])
        print("Seçilen test Verisi: ", test)

        k = input("\nK değeri giriniz: ")
        if int(k) != 0:
            sonuc, komsu = knn(traingSet, test, int(k))
            for i in range(len(komsu)):
                print("----------------------------")
                print("---  En yakın ", i + 1, ". Komşu   ---")
                print(traingSet.iloc[komsu[i]])
            print(test, " test verisine göre")
            print("K değeri: ", k, "'e göre Sınıfı: ", sonuc, " ,\nen yakın komşuları ", komsu)
    else:
        print("geçersiz değer girildi yeniden dene")



