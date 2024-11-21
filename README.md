# SKN03-FINAL-2Team
SKN03-FINAL-2Team


# DeepFM 모델을 이용한 아이템-아이템 추천 시스템

이 프로젝트는 **DeepFM** 모델을 사용하여 아이템-아이템 추천 시스템을 구축하는 방법을 설명합니다. 주로 **아이템** 간의 상호작용을 모델링하고, **Deep Neural Networks (DNN)**와 **Factorization Machines (FM)**을 결합하여 추천 결과를 도출합니다.

## 1. 피쳐 데이터

각 **아이템(뮤지컬)**을 \( i \)로 표현하고, 각 **아이템의 특성**(예: 장르, 배우, 감독 등)은 피쳐로 벡터화하여 모델에 입력합니다.

- **장르 특성 벡터**: $\mathbf{x}_i^{\text{genre}}$
- **배우 특성 벡터**: $\mathbf{x}_i^{\text{actor}}$
- **뮤지컬id 특성 벡터**: $\mathbf{x}_i^{\text{musical_id}}$


각 특성에 대해 하나의 벡터가 주어지며, 이 벡터들을 결합하여 아이템 \( i \)의 최종 특성 벡터를 얻을 수 있습니다.

## 2. Factorization Machine (FM)에서의 상호작용 모델링

### Factorization Machine (FM) 모델 수식

FM 모델에서 예측값 \( \hat{y}_i \)는 다음과 같이 계산됩니다:

$\hat{y}_i = w_0 + \sum_{k=1}^{K} w_k x_{ik} + \sum_{k_1=1}^{K} \sum_{k_2=k_1+1}^{K} \langle \mathbf{v}_{k_1}, \mathbf{v}_{k_2} \rangle x_{ik_1} x_{ik_2}$

#### 여기서:
- $\( \hat{y}_i \)$: 아이템 \( i \)에 대한 예측값
- $\( w_0 \)$: 전체 데이터의 **편향 (bias)**, 상수 항
- $\( w_k \)$: 각 특성 \( k \)에 대한 **선형 가중치**
- $\( x_{ik} \)$: 아이템 \( i \)의 특성 \( k \) 값
- $\( \mathbf{v}_k \)$: **임베딩 벡터** (특성 \( k \)에 대한 벡터 표현)
- $\( \langle \mathbf{v}_{k_1}, \mathbf{v}_{k_2} \rangle \)$: 두 벡터 $\( \mathbf{v}_{k_1} \)$와 $\( \mathbf{v}_{k_2} \)$의 **내적** (2차 항에서 사용)

#### 1. 1차 항 (선형 항)

FM에서는 각 특성에 대해 **선형적인 영향을** 모델링합니다. 1차 항은 각 특성 값에 대해 선형 가중치를 적용한 합입니다:

$\[\sum_{k=1}^{K} w_k x_{ik}\]$

여기서 $\( w_k \)$는 특성 $\( k \)$에 대한 가중치, $\( x_{ik} \)$는 아이템 $\( i \)$의 특성 $\( k \)$ 값입니다.

#### 2. 2차 항 (상호작용 항)

FM은 **이차 상호작용**(second-order interactions)을 모델링하는데, 각 아이템의 특성 간 상호작용을 **내적(inner product)**을 통해 계산합니다.

아이템 \( i \)와 \( j \)에 대해 FM은 다음과 같이 **이차 상호작용**을 모델링합니다:

$\[\hat{y}_{ij}^{\text{FM}} = \mathbf{v}_i^\top \mathbf{v}_j\]$

- $\( \mathbf{v}_i \)$와 $\( \mathbf{v}_j \)$는 각 아이템 \( i \)와 \( j \)에 대한 **임베딩 벡터**입니다.
- 이 벡터들은 각 특성(장르, 배우 등)에 대해 학습된 벡터들로, 아이템 간의 **내적**을 계산하여 **상호작용**을 모델링합니다.

FM은 **이차 상호작용**만을 고려하고, 아이템 \( i \)와 \( j \)의 벡터 내적을 사용하여 상호작용을 모델링합니다.



## 3. Deep Neural Network (DNN)에서의 비선형 학습

DNN은 **비선형 관계**를 학습합니다. 여기서 $\( \mathbf{x}_i \)$는 아이템 \( i \)의 모든 특성 벡터가 결합된 형태로, 이를 **입력 벡터**로 사용하여 모델을 학습합니다.

아이템 \( i \)의 특성 벡터는 다음과 같이 결합됩니다:
$\[\mathbf{x}_i = \left[ \mathbf{x}_i^{\text{genre}}, \mathbf{x}_i^{\text{actor}}, \mathbf{x}_i^{\text{director}}, \dots \right]\]$

이 벡터 $\( \mathbf{x}_i \)$는 DNN의 **입력층**에 들어가고, DNN은 여러 층을 거쳐 비선형 변환을 통해 예측값을 출력합니다.

$\[\hat{y}_i^{\text{DNN}} = f\left( W_2 \cdot \text{ReLU}(W_1 \cdot \mathbf{x}_i + b_1) + b_2 \right)\]$

- $\( f \)$는 **활성화 함수**입니다.
- $\( W_1, W_2 \)$는 **가중치 행렬**이고, $\( b_1, b_2 \)$는 **편향**입니다.
- $\( \mathbf{x}_i \)$는 아이템 \( i \)의 모든 특성 벡터를 결합한 입력입니다.

## 4. 최종 예측값

DeepFM에서는 FM과 DNN의 출력을 결합하여 최종 예측값을 얻습니다. 이때 두 예측값을 더하거나 다른 방법으로 결합할 수 있습니다.

$\[\hat{y}_i = \hat{y}_i^{\text{FM}} + \hat{y}_i^{\text{DNN}}\]$

여기서:

- $\( \hat{y}_i^{\text{FM}} \)$는 **Factorization Machine**에서 계산된 예측값이고,
- $\( \hat{y}_i^{\text{DNN}} \)$는 **Deep Neural Network**에서 계산된 예측값입니다.

## 5. 결론

- **뮤지컬**은 **아이템**이고, 각 **아이템**은 여러 특성(예: 장르, 배우 등)으로 구성됩니다.
- **FM**은 **아이템 간의 상호작용**을 **벡터 내적**을 통해 모델링하고, **DNN**은 **아이템 특성**의 **비선형 관계**를 학습합니다.
- 최종 예측은 두 모델의 결과를 결합하여 계산됩니다.




## Git 사용법
### Clone(처음 복사할 때)
```bash
git clone https://github.com/Leejoowon123/SKN_03_FINAL.git
git add .
``` 
### git pull(받아올 때)
```bash
git pull origin main
```
### git status(상태 확인: 올리거나 작업하기 전 항상 현 브랜치 & 상태 확인)
```bash
git status
git branch
```

### 새 브랜치 생성 및 push 방법
```bash
git checkout -b 브랜치명

# 변경사항 추가 및 커밋
git add .
pip freeze > requirements.txt # 추가 설치 모듈/라이브러리 존재할 경우
git commit # i 클릭 후 커밋 메시지 작성 → 다 작성 후 esc 클릭 후 :wq 입력 후 엔터
git push origin 브랜치명 # origin 하면 자동으로 원격저장소에 같은 이름으로 push
```

# 프로젝트 사용법
- 가상환경 설치
```bash
py -3.12 -m venv .venv
```
- 가상환경 실행
```bash
.venv\Scripts\activate
```
- 필요 모듈 설치
```bash
pip install -r requirements.txt
```
- streamlit 실행
```bash
streamlit run main.py
```
- 터미널로 실행
```bash
py -m model.recommend_musical
```
  - 배우 입력(Actor A ~ Actor Z) & 장르 입력(1 ~ 5)
  - 반복(y | n)

## 문서 목록
- [모델 설명서](model/README.md)
- [데이터 명세서](model/READMEDATA.md)
- [데이터 파이프라인](model/DATAPIPELINE.md)


## 구현
### 추천 시스템 구현 화면
<p align="center">
  <img src="./READMEImages/추천_구현_화면_1.png" width="90%" />
  <img src="./READMEImages/추천_구현_화면_2.png" width="45%" height="300px" style="object-fit: contain; margin-bottom: 10px; margin-top: -10%;" />
  <img src="./READMEImages/추천_구현_화면_3.png" width="45%" height="300px" style="object-fit: contain; margin-bottom: 10px; margin-top: -10%;" />
</p>

### 추천 모델 성능 평가
<p align="center">
  <img src="./READMEImages/추천_모델_성능_평가.png" width="70%" />
</p>
<p align="center">
  <img src="./READMEImages/추천_모델_성능_평가_2.png" width="70%" />
</p>

