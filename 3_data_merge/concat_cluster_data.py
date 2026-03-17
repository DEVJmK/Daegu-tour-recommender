"""
클러스터 데이터에 장소 상세 정보 병합
병합 기준: (분류, 관광지, 가맹점명) 일치 (대소문자 무시)
일치하는 모든 항목에 대해 상세 정보를 추가
"""

import pandas as pd


# 추가할 컬럼 목록
COLUMNS_TO_ADD = [
    '가게 이미지 URL', '별점', '블로그 리뷰 수',
    '주소', '웹사이트 URL', '위치값 주소', '위도', '경도'
]

CLUSTER_IDS   = [0, 1, 2, 3]
BASE_CSV_PATH = "data/data.csv"
INPUT_PREFIX  = "data/cluster_"
OUTPUT_PREFIX = "data/finish_cluster_"


def load_base_data(path: str) -> pd.DataFrame:
    """기준 CSV 파일 로드 및 키 컬럼 소문자 정규화"""
    df = pd.read_csv(path, encoding='cp949')
    for col in ['분류', '관광지', '가맹점명']:
        df[col] = df[col].str.lower()
    return df


def merge_cluster(cluster_df: pd.DataFrame, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    클러스터 데이터프레임에서 호텔 분류 행에만 base_df의 상세 정보를 병합
    """
    cluster_df = cluster_df.copy()
    for col in ['분류', '관광지', '가맹점명']:
        cluster_df[col] = cluster_df[col].str.lower()


    for idx, row in cluster_df.iterrows():
        match = base_df[
            (base_df['분류']    == row['분류'])    &
            (base_df['관광지']  == row['관광지'])  &
            (base_df['가맹점명'] == row['가맹점명'])
        ]
        if not match.empty:
            matched_row = match.iloc[0]
            for col in COLUMNS_TO_ADD:
                if col in matched_row:
                    cluster_df.at[idx, col] = matched_row[col]

    return cluster_df


def main():
    base_df = load_base_data(BASE_CSV_PATH)
    print(f"기준 데이터 로드: {len(base_df)}행")

    for cluster_id in CLUSTER_IDS:
        in_path  = f"{INPUT_PREFIX}{cluster_id}.xlsx"
        out_path = f"{OUTPUT_PREFIX}{cluster_id}.xlsx"

        cluster_df = pd.read_excel(in_path, engine='openpyxl')
        print(f"\n[Cluster {cluster_id}] 병합 중... ({len(cluster_df)}행)")

        merged_df = merge_cluster(cluster_df, base_df)
        merged_df.to_excel(out_path, index=False, engine='openpyxl')

if __name__ == "__main__":
    main()
