
import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

# ---------- Helpers ----------
def _find_col(df: pd.DataFrame, candidates):
    # Try exact then case-insensitive matches
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None

@st.cache_data(show_spinner=False)
def load_pickles(data_dir: Path):
    with open(data_dir / "popular.pkl", "rb") as f:
        popular_df = pickle.load(f)
    with open(data_dir / "pt.pkl", "rb") as f:
        pt = pickle.load(f)              # pivot table / index of book titles
    with open(data_dir / "books.pkl", "rb") as f:
        books = pickle.load(f)           # full books dataframe
    with open(data_dir / "similarity_scores.pkl", "rb") as f:
        similarity_scores = pickle.load(f)  # 2D numpy array or list of lists
    return popular_df, pt, books, similarity_scores

def safe_image(url):
    if isinstance(url, (list, tuple)) and url:
        return url[0]
    return url

def recommend(selected_title, pt, books, similarity_scores, k=5):
    if selected_title not in pt.index:
        raise KeyError(f"'{selected_title}' not found in pivot index.")
    index = pt.index.get_loc(selected_title)
    sims = list(enumerate(similarity_scores[index]))
    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)[1:k+1]

    out = []
    for idx, _score in sims_sorted:
        title = pt.index[idx]
        row = books[books[_find_col(books, ['Book-Title', 'Title'])] == title]
        # De-duplicate by title
        row = row.drop_duplicates(subset=_find_col(books, ['Book-Title', 'Title']))
        title_col = _find_col(row, ['Book-Title', 'Title'])
        author_col = _find_col(row, ['Book-Author', 'Author'])
        img_col = _find_col(row, ['Image-URL-M', 'Image-URL', 'Image_URL', 'ImageURL', 'Image'])
        if row.empty or title_col is None:
            out.append({"title": title, "author": "Unknown", "img": None})
            continue
        out.append({
            "title": row[title_col].iloc[0],
            "author": row[author_col].iloc[0] if author_col else "Unknown",
            "img": row[img_col].iloc[0] if img_col else None
        })
    return out

# ---------- Sidebar (Navigation) ----------
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio("Go to", ["Home: Popular", "Collaborative Filtering"])

# ---------- Load data (with helpful error) ----------
try:
    popular_df, pt, books, similarity_scores = load_pickles(DATA_DIR)
    load_ok = True
except FileNotFoundError as e:
    load_ok = False
    st.error("Pickle files not found. Please place **popular.pkl**, **pt.pkl**, **books.pkl**, **similarity_scores.pkl** inside the `data/` folder next to `app.py`.")
    st.info("Expected structure: \n\n"
            "```\nbook-recommender-streamlit/\n  app.py\n  data/\n    popular.pkl\n    pt.pkl\n    books.pkl\n    similarity_scores.pkl\n```\n")
except Exception as e:
    load_ok = False
    st.exception(e)
    st.warning("If this looks like a NumPy/Sklearn version issue, try matching versions between Colab and local (see README).")

# ---------- Pages ----------
if page == "Home: Popular":
    st.title("📖 Popular Book Recommendations")
    if not load_ok:
        st.stop()

    # Map likely column names
    title_col  = _find_col(popular_df, ['Book-Title', 'Title'])
    author_col = _find_col(popular_df, ['Book-Author', 'Author'])
    img_col    = _find_col(popular_df, ['Image-URL-M', 'Image-URL', 'Image_URL', 'ImageURL', 'Image'])
    avg_col    = _find_col(popular_df, ['avg_rating', 'Average-Rating', 'Avg-Rating', 'Rating'])
    num_col    = _find_col(popular_df, ['num_ratings', 'Ratings-Count', 'Rating-Count', 'Votes'])

    if not all([title_col, author_col, img_col]):
        st.warning("Couldn't auto-detect expected columns in `popular_df`. Showing a preview so you can rename columns in your preprocessing.")
        st.dataframe(popular_df.head(20))
        st.stop()

    n = st.slider("How many to show", min_value=5, max_value=50, value=25, step=5)
    cols_per_row = 5
    rows = (n + cols_per_row - 1) // cols_per_row

    items = popular_df.head(n).to_dict(orient="records")
    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row)
        for c in cols:
            if idx >= len(items):
                break
            item = items[idx]
            with c:
                st.image(safe_image(item.get(img_col)), use_container_width=True)
                st.markdown(f"**{item.get(title_col, '')}**")
                st.caption(f"✍️ {item.get(author_col, 'Unknown')}")
                meta = []
                if avg_col and pd.notna(item.get(avg_col)):
                    meta.append(f"⭐ {float(item.get(avg_col)):.2f}")
                if num_col and pd.notna(item.get(num_col)):
                    meta.append(f"{int(item.get(num_col))} ratings")
                if meta:
                    st.write(" | ".join(meta))
            idx += 1

elif page == "Collaborative Filtering":
    st.title("🔍 Find Similar Books")
    if not load_ok:
        st.stop()

    # Select box with search
    st.write("Select a book you like to get similar recommendations.")
    titles_index = pt.index.values.tolist()
    selected = st.selectbox("Book", options=titles_index, index=None, placeholder="Start typing a title...")

    k = st.slider("How many recommendations?", 3, 10, 5)

    if st.button("Recommend", type="primary", use_container_width=False, disabled=(selected is None)):
        try:
            recs = recommend(selected, pt, books, similarity_scores, k=k)
            cols = st.columns(min(5, k))
            for i, rec in enumerate(recs):
                with cols[i % len(cols)]:
                    if rec["img"]:
                        st.image(safe_image(rec["img"]), use_container_width=True)
                    st.markdown(f"**{rec['title']}**")
                    st.caption(f"✍️ {rec['author']}")
        except Exception as e:
            st.exception(e)
            st.warning("If this is an index/version error, ensure your pickles were created from the same data and compatible library versions.")
