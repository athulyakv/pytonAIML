import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("Books Recommender - Content Based")

books = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\reccomentation\\books.csv")
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

book_titles = books['Title'].tolist()
selected = st.selectbox("Choose a Book", book_titles)
index = books[books['Title'] == selected].index[0]

if st.button("Recommend"):
    similar_books = content_similarity[index].argsort()[::-1][1:4]
    st.write("### Recommended Books:")
    for i in similar_books:
        st.write("- " + books.iloc[i]['Title'])



import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

st.title("Books Recommender - Collaborative Based")

# Load datasets
books = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\reccomentation\\books.csv")
ratings = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\reccomentation\\ratings.csv")

# Ensure consistent column names
books.columns = books.columns.str.strip()
ratings.columns = ratings.columns.str.strip()

# Create user-item matrix
user_item_matrix = ratings.pivot_table(index='User_ID', columns='Book_ID', values='Rating').fillna(0)

# Compute user similarity matrix
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

# Collaborative recommendation function
def get_collaborative_recommendations(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return []

    # Get similar users (excluding self)
    similar_users = user_similarity_df.loc[user_id].drop(user_id)

    # Get ratings of similar users as a matrix
    similar_users_ratings = user_item_matrix.loc[similar_users.index]

    # Weighted sum of ratings
    weighted_scores = similar_users_ratings.T.dot(similar_users)
    normalization = similar_users.sum()

    # Avoid division by zero
    if normalization == 0:
        return []

    predicted_ratings = weighted_scores / normalization

    # Filter out books already rated by the user
    user_rated = user_item_matrix.loc[user_id]
    unrated_books = predicted_ratings[user_rated == 0]

    # Get top N recommendations
    top_books = unrated_books.sort_values(ascending=False).head(top_n).index.tolist()
    return top_books

# Streamlit UI
user_ids = user_item_matrix.index.tolist()
selected_user = st.selectbox("Choose a User ID", user_ids)

if st.button("Collaborative Recommend"):
    recommended_ids = get_collaborative_recommendations(selected_user)

    st.write("### Recommended Books:")
    for book_id in recommended_ids:
        match = books[books["Book_ID"] == book_id]
        if not match.empty:
            st.write("- " + match["Title"].values[0])
        else:
            st.write(f"- Book ID {book_id} (title not found)")



import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📚 Hybrid Book Recommender")

# --- Load Data ---
books = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\reccomentation\\books.csv")
ratings = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\reccomentation\\ratings.csv")

books.columns = books.columns.str.strip()
ratings.columns = ratings.columns.str.strip()

# --- Content-Based Setup ---
books['features'] = books['Title'] + " " + books['Author'] + " " + books['Genre']
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(books['features'])
content_similarity = cosine_similarity(tfidf_matrix)

book_id_to_index = pd.Series(books.index, index=books['Book_ID'])

# --- Collaborative Filtering Setup ---
user_item_matrix = ratings.pivot_table(index='User_ID', columns='Book_ID', values='Rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


# --- Hybrid Recommendation Function ---
def hybrid_recommend(user_id, book_title, top_n=5):
    # Content-based part
    if book_title not in books['Title'].values:
        return []

    content_idx = books[books['Title'] == book_title].index[0]
    content_scores = list(enumerate(content_similarity[content_idx]))

    # Collaborative part
    if user_id in user_item_matrix.index:
        similar_users = user_similarity_df.loc[user_id].drop(user_id)
        sim_users_ratings = user_item_matrix.loc[similar_users.index]
        weighted_scores = sim_users_ratings.T.dot(similar_users)
        normalization = similar_users.sum()

        if normalization != 0:
            collaborative_scores = weighted_scores / normalization
        else:
            collaborative_scores = pd.Series(0, index=user_item_matrix.columns)
    else:
        collaborative_scores = pd.Series(0, index=user_item_matrix.columns)

    # Combine scores
    combined_scores = []
    for i, score in content_scores:
        book_id = books.iloc[i]['Book_ID']
        content_score = score
        collaborative_score = collaborative_scores.get(book_id, 0)
        final_score = (content_score + collaborative_score) / 2  # Weight can be adjusted
        combined_scores.append((i, final_score))

    combined_scores = sorted(combined_scores, key=lambda x: x[1], reverse=True)

    recommended_books = []
    for idx, _ in combined_scores:
        book = books.iloc[idx]
        if book['Title'] != book_title and book['Book_ID'] not in ratings[ratings['User_ID'] == user_id][
            'Book_ID'].values:
            recommended_books.append(book['Title'])
        if len(recommended_books) >= top_n:
            break

    return recommended_books


# --- Streamlit UI ---
user_ids = user_item_matrix.index.tolist()
book_titles = books['Title'].tolist()

st.subheader("📘 Choose a Book and User")
selected_user = st.selectbox("Select User ID", user_ids)
selected_book = st.selectbox("Select Book Title", book_titles)

if st.button("🔍 Recommend (Hybrid)"):
    recommendations = hybrid_recommend(selected_user, selected_book)
    st.write("### ✅ Hybrid Recommendations:")
    for title in recommendations:
        st.write(f"- {title}")