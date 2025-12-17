🥇 **최종 1위 (Private Leaderboard Winner)**  
<br>

# 📚 Book Rating Prediction
<br>

사용자의 과거 도서 평점 데이터를 기반으로  
미래 평점을 예측하는 추천 시스템 모델을 개발하는 프로젝트입니다.

정형 메타데이터, 텍스트 정보, 이미지 정보 등을 활용해  
RMSE 최소화를 목표로 다양한 모델을 실험하고 앙상블을 수행했습니다.

---

## 🎯 Objective

- 사용자–도서 평점 예측 (Regression)
- 평가 지표: **RMSE (Root Mean Square Error)**
- 극단적으로 빗나가는 예측을 줄이고 안정적인 성능 확보

---

## 📊 Dataset

| Category | Description |
|---|---|
| Users | 68,092 |
| Books | 149,570 |
| Ratings | Train 306,795 / Test 76,699 |
| Images | 149,523 (Book Cover Images) |

- User / Book 메타데이터 + Interaction 데이터 기반 문제

---

## 🧠 Approach

### 1. Data Preprocessing
- User / Book 메타데이터 결측치 처리
- Location, Category 등 비정형 텍스트 정규화
- High-cardinality 텍스트 피처를 의미 기반 클러스터링으로 변환

### 2. Feature Engineering
- Author / Publisher 기반 count 피처
- Book age 관련 파생 변수
- Sparse 데이터 특성을 고려한 안정화 피처 설계

### 3. Modeling
- **Collaborative Filtering**
  - MF (Matrix Factorization)
  - NCF
- **Context-based Models**
  - FM, DeepFM, Image_DeepFM
- **Tree-based Models**
  - CatBoost, XGBoost

---

## 📈 Evaluation Strategy

- **5-Fold Cross Validation**
- Fold별 예측값 평균으로 분산 감소
- 단일 모델 + 앙상블 성능 비교

### Final Choice
- 단일 모델 기준 **CatBoost**가 가장 안정적인 성능
- 이질적인 모델 조합을 활용한 **Soft Voting Ensemble** 적용

---

## 🏆 Results (Summary)

- Best Public Score: **≈ 2.11 RMSE**
- Best Private Score: **≈ 2.10 RMSE**
