Bee_classification with CNN
===

## 개요  
농림축산식품부에 따르면, 2021년 겨울 기준. 국내에서 월동중인 사육 꿀벌 약 39만 봉군(78억마리)가 사라졌다고 한다.  
 전세계 식량의 90%를 차지하는 100대 농작물 중 70%가 꿀벌을 포함한 곤충의 수분활동에 의해 생산된다.  
  꿀벌의 부재는 인류의 생존에 큰 영향을 미친다. 본 프로젝트에서는 꿀벌의 종류 분류 및 건강상태를 분류한다.  

## 환경  
Google Colab  

## 과정  

1. 꿀벌 데이터 확인  

판다스 라이브러리를 통해 상위 데이터 몇개를 확인한다.    
```python
data = pd.read_csv(filename)  
data.head()
```
|index|file|date|time|location|zip code|subspecies|health|pollen\_carrying|caste|
|---|---|---|---|---|---|---|---|---|---|
|0|041\_066\.png|8/28/18|16:07|Alvin, TX, USA|77511|-1|hive being robbed|false|worker|
|1|041\_072\.png|8/28/18|16:07|Alvin, TX, USA|77511|-1|hive being robbed|false|worker|
|2|041\_073\.png|8/28/18|16:07|Alvin, TX, USA|77511|-1|hive being robbed|false|worker|
|3|041\_067\.png|8/28/18|16:07|Alvin, TX, USA|77511|-1|hive being robbed|false|worker|
|4|041\_059\.png|8/28/18|16:07|Alvin, TX, USA|77511|-1|hive being robbed|false|worker|  

2. 꿀벌의 종, 건강 상태 판별을 위해 해당 열의 정보를 가져온다.  

```python
df = pd.DataFrame(data)
health = df['health']
subspecies = df['subspecies']
```

3. 구글 드라이브에 올린 꿀벌 이미지 들을 불러온다. (PIL 라이브러리 사용 (이미지 오픈))    

```python
X_pics = [Image.open(img_name).convert("RGB") for img_name in df["file"]]
```
> 이미지 중 RGB가 아닌 이미지가 몇개 들어있어 모든 이미지를 RGB타입으로 강제 변환한다.(추후 이미지 벡터를 만들 때 생기는 문제 방지)    

4. 이미지를 배열로 변환한다. (리사이징도 함께)  
```python
X = [ cv2.resize(np.array(i),(100,100)) for i in X_pics]
X = np.array(X)
``` 

5. 결과 데이터의 원핫 인코딩을 하기 위해 중복없는 결과 데이터를 만든다.      
```python
target_health = list(set(health))
target_species = list(set(subspecies))
```
> 결과: ['hive being robbed', 'Varroa, Small Hive Beetles', 'healthy', 'ant problems', 'missing queen', 'few varrao, hive beetles']  
['Carniolan honey bee', 'VSH Italian honey bee', 'Western honey bee', '1 Mixed local stock 2', 'Italian honey bee', 'Russian honey bee', '-1']  

6. 먼저 꿀벌 건강정보에 대한 원 핫 인코딩을 진행한다.  
```python
# 원 핫 인코딩
y_keys = {"healthy":np.array([1,0,0,0,0,0]),
         "few varrao, hive beetles":np.array([0,1,0,0,0,0]),
         "Varroa, Small Hive Beetles":np.array([0,0,1,0,0,0]),
         "ant problems":np.array([0,0,0,1,0,0]),
         "hive being robbed":np.array([0,0,0,0,1,0]),
         "missing queen":np.array([0,0,0,0,0,1])}

# 모든 꿀벌 건강상태의 원핫인코딩 설정        
y = [y_keys[i] for i in df.health]
y = np.array(y)
```

7. 이제 모델을 만들어야하므로 각종 딥러닝 모델 생성 라이브러리를 import한다.  
> Keras library 사용 (cnn을 위한 layer함수를 바로 사용할 수 있게 되어있어  사용)    

8. import한 library를 통해 cnn 모델을 만든다.  
cnn 모델은 convolution 계층 2계층 neural network 2계층을 이용했다.  
(모델 최적화 방법을 잘 몰라서 검색을 통해 모델을 구축했다.)  
convolution 계층: convolution -> activate -> pooling 순으로 진행
NN 계층: Dense -> dropout -> 최종 Dense(health classes 갯수만큼)  

```python
# 꿀벌 건강에 대한 학습
def train():

    model = Sequential()

    # convolution
    model.add(Convolution2D(10,3,3, input_shape=(100,100,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),padding="SAME"))
    
    # convolution
    model.add(Convolution2D(20,3,3, activation="relu"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2),padding="SAME"))

    model.add(Flatten())

    # NN
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(6, activation="softmax"))
    
    model.compile(optimizer="Adam", loss="categorical_crossentropy",metrics=["accuracy"])

    return model
```


9. 학습을 위한 train , test, validation 데이터 분리를 하였다.  

```python
# 원본데이터로 train, test를 분류하고  train데이터로 train, validation을 분류함

# train - 학습데이터(검증데이터 포함)  test- 테스트데이터
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1)

# train - 최종 학습데이터, val - 검증데이터
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
```

10. 학습 진행 (30에폭)  
``` python
model = train()
history1 = model.fit(X_train,y_train,verbose=True,validation_data=(X_val, y_val),epochs=30)
```

11. 학습 결과  

꿀벌 건강 상태 학습 결과  
loss: 0.1459 - accuracy: 0.9374 - val_loss: 0.1832 - val_accuracy: 0.9106

![image](https://user-images.githubusercontent.com/79445881/211449219-756d74c8-94dd-4a62-9fbf-264ef3a37ea0.png)

꿀벌 건강 상태 테스트 데이터 검증 결과  
loss: 0.2165 - accuracy: 0.9111  



꿀벌 종류 학습 결과  
loss: 0.1361 - accuracy: 0.9441 - val_loss: 0.1908 - val_accuracy: 0.9191  

![image](https://user-images.githubusercontent.com/79445881/211449288-0d2f0bfb-5212-4706-a469-0c756111d0a8.png)


꿀벌 종류 테스트 데이터 검증 결과  
loss: 0.1738 - accuracy: 0.9198  

