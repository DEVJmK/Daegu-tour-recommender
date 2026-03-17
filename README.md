# 놀러가샵 🗺️
**소비특성 분석을 통한 대구시 관광지 추천 시스템**

---

## 프로젝트 개요

대구시 관광객의 체류 시간 감소 및 소비 활동 저조 문제를 해결하기 위해,
**리뷰 데이터 기반 클러스터링** + **신한카드 소비 데이터 기반 코사인 유사도 추천** 을 결합한 맞춤형 관광지 추천 시스템입니다.

---

## 파이프라인

```
데이터 수집
    ↓
리뷰 전처리 & 텍스트 벡터화 (GTR-T5-xl, GTR-T5-large 앙상블)
    ↓
클러스터링 (KMeans, 실루엣 점수 평가)
    ↓
소비 데이터 분석 (업종별 / 성별·연령별 벡터)
    ↓
코사인 유사도 기반 관광지 추천
```

---

## 디렉토리 구조

```
nolleoGaShop/
│
├── 1_preprocessing/
│   └── review_preprocessing.py          # 리뷰 전처리 & 임베딩 & 클러스터링
│
├── 2_data_merge/
│   └── concat_cluster_data.py           # 클러스터 파일에 장소 정보 병합
│
├── 3_recommendation/
│   └── cosine_similarity_recommend.py   # 소비 데이터 기반 추천 모델
│
├── data/                                # 데이터 파일 (Git 제외 - .gitignore 참고)
│   ├── tourish_data.csv
│   ├── cluster_0.xlsx ~ cluster_3.xlsx
│   └── 데이터 카드1234.xlsx
│
├── .gitignore
└── README.md
```

---

## 설치 방법

```bash
pip install -r requirements.txt
```

**주요 라이브러리:**
- `sentence-transformers` — 텍스트 임베딩 (GTR-T5 계열)
- `scikit-learn` — 클러스터링, 유사도 계산
- `pandas`, `numpy`, `openpyxl` — 데이터 처리
- `matplotlib` — 시각화

---

1. **전처리**: `1_preprocessing/review_preprocessing.py` 실행 → `cluster_0~3.xlsx` 생성
2. **병합**: `2_data_merge/concat_cluster_data.py` 실행 → `finish_cluster_0~3.xlsx` 생성
3. **추천**: `3_recommendation/cosine_similarity_recommend.py` 실행

---

## 데이터 주의사항

`data/` 폴더는 `.gitignore`에 의해 GitHub에 업로드되지 않습니다.
실제 데이터 파일은 별도로 준비해야 합니다.
