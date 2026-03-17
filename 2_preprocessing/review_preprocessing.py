"""
리뷰 전처리 / 텍스트 임베딩 / 클러스터링 파이프라인
- 입력: data/tourish_data.csv
- 출력: data/cluster_0~4.xlsx

주요 단계:
  1. 데이터 로드 및 전처리 (중복 제거, 불용어 제거)
  2. 관광지명 병합 (예: '강정고령보'+'디아크' → '강정고령보/디아크')
  3. 리뷰 블록 분리
  4. 주요 의견 추출
  5. GTR-T5-xl, GTR-T5-large 앙상블 임베딩 생성
  6. Autoencoder 차원 축소
  7. KMeans 클러스터링 (n_clusters=4)
  8. 클러스터별 xlsx 저장
"""

import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import pairwise_distances
from sentence_transformers import SentenceTransformer

# 한글 폰트 (Windows 기준)
plt.rcParams['font.family']    = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ─────────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────────

def load_data(file_path: str = "data/tourish_data.csv") -> pd.DataFrame:
    """CSV 파일 로드"""
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    print(f"로드 완료: {df.shape[0]}행 × {df.shape[1]}열")
    return df


# ─────────────────────────────────────────────
# 2. 관광지명 병합
# ─────────────────────────────────────────────

MERGE_TARGETS = {
    '강정고령보/디아크':                  ['강정고령보', '디아크'],
    '동화사/동화사집단시설지구/팔공산케이블카': ['동화사', '동화사집단시설지구', '팔공산케이블카'],
    '블로동/고분공원':                    ['블로동', '고분공원'],
    '사문진주막촌/화원유원지':             ['사문진주막촌', '화원유원지'],
    '서문시장/서문야시장':                 ['서문시장', '서문야시장'],
    '신세계백화점대구점/대구아쿠아리움':   ['신세계백화점대구점', '대구아쿠아리움'],
    '앞산전망대/앞산케이블카':             ['앞산전망대', '앞산케이블카'],
    '약령시/약령시한의약박물관':           ['약령시', '약령시한의약박물관'],
    '옥연지/송해공원':                    ['옥연지', '송해공원'],
}

def merge_tourist_spots(df: pd.DataFrame) -> pd.DataFrame:
    """여러 관광지명을 하나의 대표 이름으로 병합"""
    for merged_name, targets in MERGE_TARGETS.items():
        df.loc[df['관광지'].isin(targets), '관광지'] = merged_name
    return df


# ─────────────────────────────────────────────
# 3. 전처리
# ─────────────────────────────────────────────

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """블로그 리뷰 앞 10단어 기준 중복 제거"""
    def first_10_words(text):
        return ' '.join(str(text).split()[:10]) if pd.notna(text) else ''

    seen = set()
    unique_rows = []
    for _, row in df.iterrows():
        key = (row['가맹점명'], first_10_words(row.get('블로그 리뷰', '')))
        if key not in seen:
            seen.add(key)
            unique_rows.append(row)
    return pd.DataFrame(unique_rows)


def filter_empty_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """블로그 리뷰 또는 전처리 내용이 비어있는 행 제거"""
    mask = (
        df['블로그 리뷰'].notna() &
        (df['블로그 리뷰'].astype(str).str.strip() != '') &
        (df['블로그 리뷰'].astype(str).str.strip() != '[]') &
        df['전처리 내용'].notna() &
        (df['전처리 내용'].astype(str).str.strip() != '') &
        (df['전처리 내용'].astype(str).str.strip() != '[]')
    )
    return df[mask].reset_index(drop=True)


def clean_preprocess_content(df: pd.DataFrame) -> pd.DataFrame:
    """'전처리된 리뷰 내용' 헤더 및 '---' 제거"""
    def _clean(text):
        if pd.notna(text):
            return re.sub(r'^\*\*전처리된 리뷰 내용\*\*\s*---', '', text, flags=re.DOTALL).strip()
        return text
    df['전처리 내용'] = df['전처리 내용'].apply(_clean)
    return df


# ─────────────────────────────────────────────
# 4. 리뷰 블록 분리
# ─────────────────────────────────────────────

def split_review_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """'전처리 내용' 컬럼을 '[리뷰 블록 N]' 단위로 행 분리"""
    expanded = []
    for _, row in df.iterrows():
        content = row['전처리 내용']
        blocks  = re.split(r'\[', content)
        blocks  = [f"[{b.strip()}" for b in blocks if b.strip()]
        if not blocks:
            expanded.append(row)
        else:
            for block in blocks:
                new_row = row.copy()
                new_row['전처리 내용'] = block
                expanded.append(new_row)

    df_expanded = pd.DataFrame(expanded)
    # '**'로 시작하는 불필요한 블록 제거
    df_expanded = df_expanded[~df_expanded['전처리 내용'].str.startswith('[**', na=False)]
    return df_expanded.reset_index(drop=True)


def match_reviews_with_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """블록 번호에 맞는 블로그 리뷰 텍스트 매핑"""
    def _match(row):
        try:
            review_list = ast.literal_eval(row['블로그 리뷰'])
            match = re.search(r'\d+', row['전처리 내용'])
            block_num = int(match.group()) - 1 if match else -1
            if 0 <= block_num < len(review_list):
                row['블로그 리뷰'] = review_list[block_num]
            else:
                row['블로그 리뷰'] = None
        except (ValueError, SyntaxError):
            row['블로그 리뷰'] = None
        return row

    return df.apply(_match, axis=1)


# ─────────────────────────────────────────────
# 5. 주요 의견 추출 및 정제
# ─────────────────────────────────────────────

def extract_and_clean_opinions(df: pd.DataFrame) -> pd.DataFrame:
    """'주요 의견:' 이후 텍스트를 추출하고 특수문자 정제"""
    def _extract(text):
        if pd.notna(text):
            m = re.search(r'주요 의견:.*', text, re.DOTALL)
            return m.group(0).strip() if m else None
        return None

    def _clean(text):
        if pd.notna(text):
            text = re.sub(r'(주요 의견|요약|\n|[^\w\s])', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        return text

    df['주요 의견 및 요약'] = df['전처리 내용'].apply(_extract).apply(_clean)
    df = df[df['주요 의견 및 요약'].notna()]
    df = df[df['주요 의견 및 요약'].str.strip() != '']
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 6. 임베딩 (GTR-T5-xl + GTR-T5-large 앙상블)
# ─────────────────────────────────────────────

def generate_ensemble_embeddings(texts: list, device: str = 'cuda') -> np.ndarray:
    """
    GTR-T5-xl과 GTR-T5-large 두 모델의 임베딩을 평균 앙상블하여 반환
    GPU가 없을 경우 device='cpu' 로 변경
    """
    model_names = ['gtr-t5-xl', 'gtr-t5-large']
    all_embeddings = []

    for name in model_names:
        print(f"\n  모델 로드: {name}")
        try:
            model = SentenceTransformer(name, device=device)
            emb   = model.encode(texts, show_progress_bar=True)
            all_embeddings.append(emb)
            print(f"  임베딩 완료: {name} → shape {emb.shape}")
        except Exception as e:
            print(f"  모델 실패 ({name}): {e}")

    if not all_embeddings:
        raise RuntimeError("모든 임베딩 모델 실행 실패")

    ensemble = np.mean(all_embeddings, axis=0)
    print(f"\n앙상블 임베딩 shape: {ensemble.shape}")
    return ensemble


# ─────────────────────────────────────────────
# 7. (선택) Autoencoder 차원 축소
# ─────────────────────────────────────────────

def reduce_with_pca(embeddings: np.ndarray, n_components: int = 128) -> np.ndarray:
    """PCA로 차원 축소 (Autoencoder 대체용 — 간단 버전)"""
    reducer = PCA(n_components=min(n_components, embeddings.shape[1]))
    reduced = reducer.fit_transform(embeddings)
    print(f"PCA 차원 축소: {embeddings.shape[1]} → {reduced.shape[1]}")
    return reduced


# ─────────────────────────────────────────────
# 8. 클러스터링 및 평가
# ─────────────────────────────────────────────

def run_kmeans(embeddings: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> np.ndarray:
    """KMeans 클러스터링"""
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(embeddings)
    print(f"KMeans 클러스터 수: {n_clusters}")
    return labels


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    """실루엣 점수, 응집도, 분리도 계산"""
    unique_labels = np.unique(labels)
    num_clusters  = len(unique_labels) - (1 if -1 in labels else 0)
    print(f"클러스터 개수: {num_clusters}")

    valid = labels != -1
    sil_score = (
        silhouette_score(embeddings[valid], labels[valid])
        if num_clusters > 1 else "N/A"
    )

    cohesion = np.mean([
        np.mean(pairwise_distances(embeddings[labels == c]))
        for c in unique_labels if c != -1
    ])

    centroids  = np.array([embeddings[labels == c].mean(axis=0) for c in unique_labels if c != -1])
    separation = np.mean(pairwise_distances(centroids)) if len(centroids) > 1 else "N/A"

    return {
        "Silhouette Score":              sil_score,
        "Cohesion (Cluster Compactness)": cohesion,
        "Separation (Cluster Separation)": separation,
    }


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, title: str = "KMeans Clustering"):
    """PCA 2D 클러스터 시각화"""
    reduced = PCA(n_components=2).fit_transform(embeddings)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label="클러스터")
    plt.title(title)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
# 9. 결과 저장
# ─────────────────────────────────────────────

def save_cluster_files(df: pd.DataFrame, labels: np.ndarray, output_dir: str = "data"):
    """클러스터 레이블별로 xlsx 파일 저장"""
    df = df.copy()
    df['Cluster Label'] = labels

    for cluster_id in sorted(df['Cluster Label'].unique()):
        cluster_df = df[df['Cluster Label'] == cluster_id]
        path = f"{output_dir}/cluster_{cluster_id}.xlsx"
        cluster_df.to_excel(path, index=False, engine='openpyxl')
        print(f"  저장: {path}  ({len(cluster_df)}행)")


# ─────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("리뷰 전처리 및 클러스터링 파이프라인 시작")
    print("=" * 60)

    # 1. 로드
    df = load_data("data/tourish_data.csv")

    # 2. 관광지명 병합
    df = merge_tourist_spots(df)

    # 3. 전처리
    df = remove_duplicates(df)
    df = filter_empty_reviews(df)
    df = clean_preprocess_content(df)

    # 4. 리뷰 블록 분리
    df = split_review_blocks(df)
    df = match_reviews_with_blocks(df)

    # 5. 주요 의견 추출
    df = extract_and_clean_opinions(df)
    print(f"\n최종 행 수: {len(df)}")

    # 6. 임베딩
    texts      = df['주요 의견 및 요약'].tolist()
    embeddings = generate_ensemble_embeddings(texts, device='cuda')  # GPU 없으면 'cpu'

    # 7. 차원 축소 (PCA, Autoencoder 대체)
    reduced = reduce_with_pca(embeddings, n_components=128)

    # 8. 클러스터링
    labels  = run_kmeans(reduced, n_clusters=4)
    metrics = evaluate_clustering(reduced, labels)
    print("\n[클러스터링 평가]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    # 9. 시각화
    visualize_clusters(reduced, labels, title="GTR-T5 앙상블 KMeans 클러스터링")

    # 10. 저장
    save_cluster_files(df, labels, output_dir="data")
    print("\n✅ 완료: data/cluster_0~4.xlsx")


if __name__ == "__main__":
    main()
