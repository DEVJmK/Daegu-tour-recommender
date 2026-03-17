"""
소비 데이터 기반 관광지 추천 모델
- 입력: data/데이터 카드1234.xlsx (신한카드 소비 데이터)
         data/finish_cluster_0~3.xlsx (클러스터 결과)
- 출력: 콘솔에 추천 관광지 및 가맹점 목록 출력

추천 방법:
  1. 사용자 클러스터 선택
  2. 업종별 소비 금액 → 코사인 유사도 계산
  3. 성별/연령대별 소비 가중치 적용
  4. 유사도 상위 5개 관광지 추천
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

plt.rcParams['font.family']        = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
pd.options.display.float_format    = '{:.4f}'.format


# ─────────────────────────────────────────────
# 데이터 로드 및 전처리
# ─────────────────────────────────────────────

CATEGORY_MAPPING = {
    '유흥':          '유흥',
    '외식업기타':     '외식',
    '중식/일식/양식': '외식',
    '한식':           '외식',
    '음/식료품':      '소매/쇼핑',
    '할인점/슈퍼마켓': '소매/쇼핑',
    '백화점/면세점':  '소매/쇼핑',
    '숙박':           '숙박',
    '스포츠/문화/레저': '스포츠 및 문화',
}


def load_card_data(path: str = "data/데이터 카드1234.xlsx") -> pd.DataFrame:
    """신한카드 소비 데이터 로드 및 대분류 범주화"""
    df = pd.read_excel(path, engine='openpyxl')
    df['소비관광지역명칭'] = (
        df['소비관광지역명칭']
        .str.replace(r'\(.*?\)', '', regex=True)
        .str.replace(r'\s+', '', regex=True)
    )
    df['대분류'] = df['대분류'].map(CATEGORY_MAPPING)
    return df


def build_industry_vector(card_df: pd.DataFrame) -> pd.DataFrame:
    """관광지별 업종별 평균 소비 금액 벡터"""
    grp = card_df.groupby(["소비관광지역명칭", "대분류"]).agg(
        금액=("카드이용금액_업종별", "sum"),
        건수=("카드이용건수_업종별", "sum"),
    ).reset_index()
    grp['평균금액'] = (grp['금액'] / grp['건수']).round(0)

    pivot = grp.pivot_table(
        index='소비관광지역명칭',
        columns='대분류',
        values='평균금액',
        aggfunc='sum',
        fill_value=0,
    ).reset_index()

    # 컬럼 순서 고정
    categories = ["소매/쇼핑", "숙박", "스포츠 및 문화", "외식", "유흥"]
    pivot.columns.name = None
    pivot = pivot.rename(columns={"소비관광지역명칭": "관광지"})
    for cat in categories:
        if cat not in pivot.columns:
            pivot[cat] = 0
    return pivot[["관광지"] + categories]


def build_demographic_vector(card_df: pd.DataFrame) -> pd.DataFrame:
    """관광지 × (성별, 연령대) 평균 소비 금액 피벗"""
    grp = card_df.groupby(["소비관광지역명칭", "성별", "연령대"]).agg(
        금액=("카드이용금액_성별연령별", "sum"),
        건수=("카드이용건수_성별연령별", "sum"),
    ).reset_index()
    grp['평균금액'] = (grp['금액'] / grp['건수']).round(0)

    pivot = grp.pivot_table(
        index='소비관광지역명칭',
        columns=['성별', '연령대'],
        values='평균금액',
        aggfunc='sum',
        fill_value=0,
    )
    return pivot


def load_cluster(cluster_id: int, prefix: str = "data/finish_cluster_") -> pd.DataFrame:
    """클러스터 xlsx 로드"""
    path = f"{prefix}{cluster_id}.xlsx"
    df = pd.read_excel(path, engine='openpyxl')
    print(f"클러스터 {cluster_id} 로드: {len(df)}행")
    return df


# ─────────────────────────────────────────────
# 추천 계산
# ─────────────────────────────────────────────

def recommend(
    cluster_id: int,
    user_gender: str,
    user_age_group: int,
    user_consumption: dict,
    top_n: int = 5,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """
    Parameters
    ----------
    cluster_id       : 선택할 클러스터 번호 (0~3)
    user_gender      : 'M' 또는 'F'
    user_age_group   : 연령대 정수 (예: 20, 30, 40)
    user_consumption : {'소매/쇼핑': 금액, '숙박': 금액, ...}
    top_n            : 추천 관광지 수
    alpha            : 인구통계 가중치 강도 (0: 무시, 1: 강하게 반영)

    Returns
    -------
    DataFrame with columns: 분류, 관광지, 가맹점명, 유사도
    """
    print(f"\n추천 시작 | 클러스터: {cluster_id} | 성별: {user_gender} | 연령: {user_age_group}대")

    # ── 데이터 로드 ──
    card_df    = load_card_data()
    ind_vec    = build_industry_vector(card_df)
    demo_vec   = build_demographic_vector(card_df)
    cluster_df = load_cluster(cluster_id)

    # ── 클러스터 내 관광지 필터링 ──
    tourist_spots = cluster_df["관광지"].unique()
    ind_filtered  = ind_vec[ind_vec["관광지"].isin(tourist_spots)]

    # ── 인구통계 가중치 ──
    try:
        demo_filtered = demo_vec.loc[
            demo_vec.index.isin(tourist_spots), (user_gender, user_age_group)
        ].reset_index()
        demo_filtered.columns = ["관광지", "가중치"]
        scaler = MinMaxScaler()
        demo_filtered["가중치"] = scaler.fit_transform(demo_filtered[["가중치"]])
    except KeyError:
        demo_filtered = pd.DataFrame({
            "관광지": tourist_spots,
            "가중치": np.zeros(len(tourist_spots))
        })

    # ── 사용자 소비 벡터 ──
    categories  = ["소매/쇼핑", "숙박", "스포츠 및 문화", "외식", "유흥"]
    user_vector = np.array([user_consumption.get(c, 0) for c in categories], dtype=float)

    # ── 유사도 계산 ──
    results = []
    for _, row in ind_filtered.iterrows():
        region     = row["관광지"]
        reg_vec    = row[categories].to_numpy(dtype=float)

        w_row = demo_filtered[demo_filtered["관광지"] == region]
        w_val = float(w_row["가중치"].values[0]) if not w_row.empty else 0.0
        if np.all(user_vector == 0) or np.all(reg_vec == 0):
            sim = 0.0
        else:
            sim = float(cosine_similarity(
                user_vector.reshape(1, -1),
                reg_vec.reshape(1, -1)
            )[0][0])
                  
        final_score = sim * (1 + alpha * w_val)
        results.append({"관광지": region, "유사도": final_score})

    similarity_df = pd.DataFrame(results).sort_values("유사도", ascending=False)

    # ── 상위 N개 관광지에서 가맹점 추출 ──
    top_spots     = similarity_df["관광지"].head(top_n).tolist()
    filtered      = cluster_df[cluster_df["관광지"].isin(top_spots)]
    unique_stores = filtered.drop_duplicates(subset=["관광지", "가맹점명"])

    final = pd.merge(unique_stores, similarity_df[["관광지", "유사도"]], on="관광지", how="inner")
    final = final[["분류", "관광지", "가맹점명", "유사도"]].sort_values("유사도", ascending=False)

    return final


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    # ── 사용자 입력 예시 ──
    choice_cluster = 0
    user_gender    = "F"   # 'M' 또는 'F'
    user_age_group = 20    # 20, 30, 40, ...

    user_consumption = {
        "소매/쇼핑":     50000,
        "숙박":          20000,
        "스포츠 및 문화": 30000,
        "외식":         100000,
        "유흥":          20000,
    }

    result = recommend(
        cluster_id       = choice_cluster,
        user_gender      = user_gender,
        user_age_group   = user_age_group,
        user_consumption = user_consumption,
        top_n            = 5,
    )

    print("\n" + "=" * 60)
    print("추천 결과")
    print("=" * 60)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
