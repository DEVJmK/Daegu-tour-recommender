import re
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sentence_transformers import SentenceTransformer

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data(file_path: str = "data/tourish_data.csv") -> pd.DataFrame:
    df = pd.read_csv(file_path, encoding="utf-8-sig")
    return df

MERGE_TARGETS = {
    '강정고령보/디아크': ['강정고령보', '디아크'],
    '동화사/동화사집단시설지구/팔공산케이블카': ['동화사', '동화사집단시설지구', '팔공산케이블카'],
    '블로동/고분공원': ['블로동', '고분공원'],
    '사문진주막촌/화원유원지': ['사문진주막촌', '화원유원지'],
    '서문시장/서문야시장': ['서문시장', '서문야시장'],
    '신세계백화점대구점/대구아쿠아리움': ['신세계백화점대구점', '대구아쿠아리움'],
    '앞산전망대/앞산케이블카': ['앞산전망대', '앞산케이블카'],
    '약령시/약령시한의약박물관': ['약령시', '약령시한의약박물관'],
    '옥연지/송해공원': ['옥연지', '송해공원'],
}


def merge_tourist_spots(df: pd.DataFrame) -> pd.DataFrame:
    for merged_name, targets in MERGE_TARGETS.items():
        df.loc[df['관광지'].isin(targets), '관광지'] = merged_name
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
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
    def _clean(text):
        if pd.notna(text):
            return re.sub(r'^\*\*전처리된 리뷰 내용\*\*\s*---', '', text, flags=re.DOTALL).strip()
        return text

    df['전처리 내용'] = df['전처리 내용'].apply(_clean)
    return df


def split_review_blocks(df: pd.DataFrame) -> pd.DataFrame:
    expanded = []
    for _, row in df.iterrows():
        content = row['전처리 내용']
        blocks = re.split(r'\[', content)
        blocks = [f"[{b.strip()}" for b in blocks if b.strip()]

        if not blocks:
            expanded.append(row)
        else:
            for block in blocks:
                new_row = row.copy()
                new_row['전처리 내용'] = block
                expanded.append(new_row)

    df_expanded = pd.DataFrame(expanded)
    df_expanded = df_expanded[~df_expanded['전처리 내용'].str.startswith('[**', na=False)]
    return df_expanded.reset_index(drop=True)


def match_reviews_with_blocks(df: pd.DataFrame) -> pd.DataFrame:
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


def extract_and_clean_opinions(df: pd.DataFrame) -> pd.DataFrame:
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


def generate_ensemble_embeddings(texts: list, device: str = 'cuda') -> np.ndarray:
    model_names = ['gtr-t5-xl', 'gtr-t5-large']
    all_embeddings = []

    for name in model_names:
        print(f"\n모델 로드: {name}")
        try:
            model = SentenceTransformer(name, device=device)
            emb = model.encode(texts, show_progress_bar=True)
            all_embeddings.append(emb)
            print(f"임베딩 완료: {name} → shape {emb.shape}")
        except Exception as e:
            print(f"모델 실패 ({name}): {e}")

    if not all_embeddings:
        raise RuntimeError("모든 임베딩 모델 실행 실패")

    ensemble = np.mean(all_embeddings, axis=0)
    print(f"\n앙상블 임베딩 shape: {ensemble.shape}")
    return ensemble


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        hidden_dim_1 = max(input_dim // 2, latent_dim * 2)
        hidden_dim_2 = max(input_dim // 4, latent_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

def reduce_with_autoencoder(
    embeddings: np.ndarray,
    n_components: int = 128,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = None,
) -> np.ndarray:
    input_dim = embeddings.shape[1]
    latent_dim = min(n_components, input_dim)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = torch.tensor(embeddings, dtype=torch.float32)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(dataset)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Autoencoder 학습 {epoch + 1}/{epochs} - loss: {avg_loss:.6f}")

    model.eval()
    with torch.no_grad():
        reduced = model.encoder(x.to(device)).cpu().numpy()

    print(f"Autoencoder 차원 축소: {input_dim} → {reduced.shape[1]}")
    return reduced


def run_kmeans(embeddings: np.ndarray, n_clusters: int = 4, random_state: int = 42) -> np.ndarray:
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(embeddings)
    print(f"KMeans 클러스터 수: {n_clusters}")
    return labels


def evaluate_clustering(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    print(f"클러스터 개수: {num_clusters}")

    valid = labels != -1
    sil_score = silhouette_score(embeddings[valid], labels[valid]) if num_clusters > 1 else "N/A"

    cohesion = np.mean([
        np.mean(pairwise_distances(embeddings[labels == c]))
        for c in unique_labels if c != -1
    ])

    centroids = np.array([embeddings[labels == c].mean(axis=0) for c in unique_labels if c != -1])
    separation = np.mean(pairwise_distances(centroids)) if len(centroids) > 1 else "N/A"

    return {
        "Silhouette Score": sil_score,
        "Cohesion (Cluster Compactness)": cohesion,
        "Separation (Cluster Separation)": separation,
    }


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, title: str = "KMeans Clustering"):
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


def save_cluster_files(df: pd.DataFrame, labels: np.ndarray, output_dir: str = "data"):
    df = df.copy()
    df['Cluster Label'] = labels

    for cluster_id in sorted(df['Cluster Label'].unique()):
        cluster_df = df[df['Cluster Label'] == cluster_id]
        path = f"{output_dir}/cluster_{cluster_id}.xlsx"
        cluster_df.to_excel(path, index=False, engine='openpyxl')
        print(f"저장: {path} ({len(cluster_df)}행)")


def main():
    print("=" * 60)
    print("리뷰 전처리 및 클러스터링 파이프라인 시작")
    print("=" * 60)

    df = load_data("data/tourish_data.csv")
    df = merge_tourist_spots(df)
    df = remove_duplicates(df)
    df = filter_empty_reviews(df)
    df = clean_preprocess_content(df)
    df = split_review_blocks(df)
    df = match_reviews_with_blocks(df)
    df = extract_and_clean_opinions(df)
    print(f"\n최종 행 수: {len(df)}")

    texts = df['주요 의견 및 요약'].tolist()
    embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings = generate_ensemble_embeddings(texts, device=embedding_device)

    reduced = reduce_with_autoencoder(
        embeddings,
        n_components=128,
        epochs=50,
        batch_size=64,
        lr=1e-3,
        device=embedding_device
    )

    labels = run_kmeans(reduced, n_clusters=4)
    metrics = evaluate_clustering(reduced, labels)

    print("\n[클러스터링 평가]")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    visualize_clusters(reduced, labels, title="GTR-T5 앙상블 KMeans 클러스터링")
    save_cluster_files(df, labels, output_dir="data")
    print("\n완료: data/cluster_0~4.xlsx")


if __name__ == "__main__":
    main()
