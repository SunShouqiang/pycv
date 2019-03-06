import cv2
import numpy as np

#创建两个SIFT实例，一个提取关键点，一个是提取特征
detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm = 1, trees = 5)
matcher = cv2.FlannBasedMatcher(flann_params, {})       #Flann匹配

bow_kmeans_trainer = cv2.BOWKMeansTrainer(40)          #BOW训练器       指定簇数为40
extract_bow = cv2.BOWImgDescriptorExtractor(extract, matcher)

def extract_sift(fn):
  im = cv2.imread(fn,0)
  return extract.compute(im, detect.detect(im))[1]

#m = cv2.imread('CarData/TrainImages/pos-' + str(1 +1)+'.pgm')
#cv2.imshow('hel',m)

for i in range(8):            # 每个类都从数据集中读取八张图像（八个正样例，和八个负样例）
  bow_kmeans_trainer.add(extract_sift('CarData/TrainImages/pos-' + str(i +1) +'.pgm'))
  bow_kmeans_trainer.add(extract_sift('CarData/TrainImages/neg-' + str(i +1) +'.pgm'))
  # print('a')
  
voc = bow_kmeans_trainer.cluster()          #创建视觉单词词汇调用cluster函数，执行k-means分类并返回词汇
extract_bow.setVocabulary( voc )

def bow_features(fn):
  im = cv2.imread(fn,0)
  # print('d')
  return extract_bow.compute(im, detect.detect(im))

traindata, trainlabels = [],[]
for i in range(100):            # 检测范围越大越准确
  traindata.extend(bow_features('CarData/TrainImages/pos-' + str(i +1)+'.pgm')); trainlabels.append(1)        #（1代表正匹配，-1代表负匹配）
  # print('c')
  traindata.extend(bow_features('CarData/TrainImages/neg-' + str(i +1)+'.pgm')); trainlabels.append(-1)
  # print('b')

svm = cv2.ml.SVM_create()         #创建SVM实例
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

def predict(fn):
  f = bow_features(fn)
  p = svm.predict(f)
  print(fn, "\t", p[1][0][0])
  return p

# again placeholder paths
car ="car.jpg"      #测试项
car_img = cv2.imread(car)
car_predict = predict(car)

font = cv2.FONT_HERSHEY_SIMPLEX

if (car_predict[1][0][0] == 1.0):
  cv2.putText(car_img,'Car Detected',(10,30), font, 1,(0,255,0),2,cv2.LINE_AA)

if (car_predict[1][0][0] == -1.0):
  cv2.putText(car_img,'Car Not Detected',(10,30), font, 1,(0,0, 255),2,cv2.LINE_AA)

cv2.imshow('BOW + SVM Result', car_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
