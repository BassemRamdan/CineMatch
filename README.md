# 🎬 CineMatch: Hybrid Movie Recommendation System

CineMatch is a modern, AI-powered movie recommendation system that combines the strengths of **Content-Based Filtering** and **Collaborative Filtering** to provide highly personalized movie suggestions. 

It is built completely natively in Python using `scikit-learn`, ensuring seamless installation on any operating system (including Windows) without the need for complex C++ build tools.

![CineMatch Banner](https://img.shields.io/badge/Status-Complete-success?style=for-the-badge) ![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge)

## 🌟 Features

*   **Content-Based Filtering**: Recommends movies similar to a user's chosen "Seed Movie" by analyzing movie genres using **TF-IDF Vectorization** and **Cosine Similarity**.
*   **Collaborative Filtering**: Learns latent user preferences and movie features from historical user ratings using **Singular Value Decomposition (TruncatedSVD)**.
*   **Hybrid Engine**: Intelligently combines both strategies with weighted scoring to mitigate the "Cold Start" problem and provide more robust recommendations.
*   **Cinematic UI Dashboard**: A stunning, animated Streamlit web interface with dynamic inputs, glowing neon CSS, and interactive strategy toggles.
*   **Built-in Evaluation**: Real-time test-set evaluation metrics directly in the dashboard (RMSE, MAE, Precision@10, Recall@10, F1-Score).

---

## 📂 Project Structure

```text
CineMatch/
│
├── data/
│   └── raw/                   # Place your movies.csv & ratings.csv here
│
├── src/
│   ├── preprocessing.py       # Data cleaning, cold-start filtering & train/test split
│   ├── content_engine.py      # TF-IDF & Cosine Similarity logic
│   ├── collaborative_engine.py# Matrix Factorization using scikit-learn (TruncatedSVD)
│   ├── hybrid_engine.py       # Weighted aggregation of normalized scores
│   └── evaluation.py          # RMSE, MAE, and ranking metrics functions
│
├── app/
│   └── main.py                # Main Streamlit Dashboard application
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🛠️ How to Install & Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BassemRamdan/CineMatch.git
   cd CineMatch
   ```

2. **Download the Dataset:**
   Download the [MovieLens dataset](https://grouplens.org/datasets/movielens/) and ensure `movies.csv` and `ratings.csv` are placed inside the `data/raw/` directory.

3. **Install Dependencies:**
   Install all the required python libraries.
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**
   Launch the Streamlit dashboard in your web browser.
   ```bash
   streamlit run app/main.py
   ```

---

## 🔬 How the Models Work

1. **TF-IDF (Term Frequency-Inverse Document Frequency):**
   Converts movie genres into numerical vectors, assigning higher weights to unique, descriptive genres and lower weights to overly common ones.

2. **Truncated SVD (Singular Value Decomposition):**
   Unlike older versions of this project that relied on libraries requiring C++ compilers (`scikit-surprise` or `implicit`), this project uses standard `scikit-learn`. It factors the massive User-Item rating matrix into smaller, dense matrices representing latent factors. The dot product of a user's vector and a movie's vector gives the predicted rating.

3. **MinMaxScaler:**
   Before combining the Content-based similarity score (range 0 to 1) and the Collaborative predicted rating (range 1 to 5), both scores are normalized to a strict `[0, 1]` scale so they can be accurately merged using a weighted average.
