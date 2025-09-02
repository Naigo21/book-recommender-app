# Book Recommender — Streamlit UI

This is a minimal Streamlit UI for your Book Recommendation project with two pages:

- **Home: Popular** — shows popular (trending) books
- **Collaborative Filtering** — pick a book and get similar recommendations

## 1) Project structure

```
book-recommender-streamlit/
  app.py
  requirements.txt
  README.md
  data/
    popular.pkl
    pt.pkl
    books.pkl
    similarity_scores.pkl
```

> Put your pickles inside the `data/` folder.

---

## 2) Export your artifacts from Colab

Open your existing Colab notebook and make sure you have these Python variables *already computed*:

- `popular_df`  (DataFrame with columns like: Book-Title, Book-Author, Image-URL(-M), avg_rating, num_ratings)
- `pt`          (Pivot table / index of book titles used in collaborative filtering)
- `books`       (Full books metadata DataFrame)
- `similarity_scores` (2D similarity matrix, e.g., cosine similarities)

Then run this cell in Colab to export them as pickles and download:

```python
import pickle, os
from google.colab import files

os.makedirs("export", exist_ok=True)
with open("export/popular.pkl", "wb") as f: pickle.dump(popular_df, f)
with open("export/pt.pkl", "wb") as f: pickle.dump(pt, f)
with open("export/books.pkl", "wb") as f: pickle.dump(books, f)
with open("export/similarity_scores.pkl", "wb") as f: pickle.dump(similarity_scores, f)

for fn in ["popular.pkl", "pt.pkl", "books.pkl", "similarity_scores.pkl"]:
    files.download(f"export/{fn}")
```

Copy the four `.pkl` files into the `data/` folder locally.

---

## 3) Open in PyCharm (Windows)

1. **Extract** this folder somewhere (e.g., `D:\Projects\book-recommender-streamlit`).
2. **Open PyCharm** → *Open* → select this folder.
3. When prompted, **create a virtual environment** (recommended), Python 3.10+.
4. In the PyCharm *Terminal* (bottom panel), run:
   ```bash
   pip install -r requirements.txt
   ```

---

## 4) Run the app in PyCharm

### Option A — via Terminal
```bash
streamlit run app.py
```

### Option B — via Run Configuration
- Run → *Edit Configurations…*
- Click **+** → **Python**
- **Name:** Streamlit
- **Script path:** (browse to) the `streamlit` executable inside your venv, e.g.  
  `...\venv\Scripts\streamlit.exe`
- **Parameters:** `run app.py`
- **Working directory:** the project folder
- Apply → OK → **Run**.

Open the local URL shown (usually `http://localhost:8501`).

---

## 5) Troubleshooting

- **FileNotFoundError / Missing pickles**  
  Ensure the four pickle files are inside the `data/` folder with the exact names.

- **NumPy / sklearn version mismatch when loading pickles**  
  Pickle files are sensitive to the library versions they were created with.  
  In Colab, print versions:
  ```python
  import numpy as np, sklearn
  print(np.__version__, sklearn.__version__)
  ```
  Then pin the same versions locally by editing `requirements.txt` and reinstalling:
  ```bash
  pip install --upgrade --force-reinstall numpy==<colab_version> scikit-learn==<colab_version>
  ```

- **Images not showing**  
  Check your image column name in the data (`Image-URL-M` vs `Image-URL`). The app tries common names automatically.

- **Port blocked**  
  If port 8501 is busy, run:  
  `streamlit run app.py --server.port 8502`

---

## 6) Customize

- To change how many popular books show by default, adjust the slider on the page.
- To change layout, update the `cols_per_row` value inside `app.py`.
- If your column names differ, either rename them in preprocessing or edit the `_find_col` candidate lists.
