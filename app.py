"""
Team Rocket — Food.com Recipe Recommender
Streamlit UI  |  CSC 577
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
import ast
import os

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Recipe Recommender · Team Rocket",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — clean dark-food editorial theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:        #0f0e0c;
    --surface:   #1a1814;
    --border:    #2e2b26;
    --accent:    #e8a838;
    --accent2:   #c0392b;
    --text:      #f0ebe3;
    --muted:     #8a8278;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.stApp { background-color: var(--bg); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Headings */
h1 { font-family: 'Playfair Display', serif; font-size: 2.4rem !important;
     font-weight: 900; letter-spacing: -1px; color: var(--text) !important; }
h2 { font-family: 'Playfair Display', serif; font-size: 1.5rem !important;
     color: var(--accent) !important; }
h3 { font-family: 'DM Sans', sans-serif; font-size: 1rem !important;
     font-weight: 500; color: var(--muted) !important;
     text-transform: uppercase; letter-spacing: 2px; }

/* Recipe card */
.recipe-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 14px;
    transition: border-color 0.2s;
}
.recipe-card:hover { border-color: var(--accent); }
.recipe-rank {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--accent);
    opacity: 0.4;
    float: right;
    margin-top: -4px;
}
.recipe-name {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 6px;
}
.recipe-meta {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 10px;
}
.tag {
    display: inline-block;
    background: var(--border);
    color: var(--muted);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.72rem;
    margin: 2px 2px 2px 0;
}
.tag-missing {
    background: rgba(192,57,43,0.2);
    color: #e74c3c;
}
.tag-ok {
    background: rgba(39,174,96,0.15);
    color: #2ecc71;
}
.pill-score {
    display: inline-block;
    background: rgba(232,168,56,0.15);
    color: var(--accent);
    border: 1px solid rgba(232,168,56,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 500;
}
.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}
.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--accent);
}
.stat-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

/* Buttons */
.stButton > button {
    background: var(--accent) !important;
    color: #0f0e0c !important;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; }

/* Inputs */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* Slider */
.stSlider > div { color: var(--text) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    border-bottom: 1px solid var(--border) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--muted) !important;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* Expander — recipe card style */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 12px !important;
}
[data-testid="stExpander"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: var(--text) !important;
    padding: 14px 18px !important;
}
[data-testid="stExpander"] summary:hover {
    color: var(--accent) !important;
}
[data-testid="stExpander"] svg {
    color: var(--accent) !important;
    fill: var(--accent) !important;
}
[data-testid="stExpanderDetails"] {
    border-top: 1px solid var(--border) !important;
    padding: 16px 18px !important;
}
/* Metric boxes inside expander */
[data-testid="stMetric"] {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 10px !important;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Playfair Display', serif !important; }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────
DATA_DIR = "data/processed"

@st.cache_data(show_spinner=False)
def load_data():
    interactions = pd.read_csv(f"{DATA_DIR}/interactions_clean.csv")
    recipes      = pd.read_csv(f"{DATA_DIR}/recipes_clean.csv")

    # Parse ingredients back into sets
    recipes["ingredient_set"] = recipes["ingredients"].apply(
        lambda x: frozenset(i.strip().lower() for i in x.split("|"))
        if isinstance(x, str) else frozenset()
    )
    recipe_ing_map = dict(zip(recipes["recipe_id"], recipes["ingredient_set"]))
    recipe_info    = recipes.set_index("recipe_id")
    return interactions, recipes, recipe_ing_map, recipe_info


@st.cache_resource(show_spinner=False)
def build_models(interactions):
    """
    Build recommendation models keeping everything sparse.
    Never materialises the full dense matrix — too large for RAM.
    Predictions are computed per-user on demand in get_top_n().
    """
    all_users   = sorted(interactions["user_id"].unique())
    all_recipes = sorted(interactions["recipe_id"].unique())

    user2idx   = {u: i for i, u in enumerate(all_users)}
    recipe2idx = {r: i for i, r in enumerate(all_recipes)}
    idx2recipe = {i: r for r, i in recipe2idx.items()}

    n_users   = len(all_users)
    n_recipes = len(all_recipes)

    interactions = interactions.copy()
    interactions["u_idx"] = interactions["user_id"].map(user2idx)
    interactions["r_idx"] = interactions["recipe_id"].map(recipe2idx)

    # ── Build sparse rating matrix (stays sparse) ──────────────────────
    R_sparse = csr_matrix(
        (interactions["rating"].astype(np.float64).values,
         (interactions["u_idx"].values, interactions["r_idx"].values)),
        shape=(n_users, n_recipes)
    )

    # ── User means from sparse (no .toarray()) ─────────────────────────
    # Sum of ratings per user / count of ratings per user
    rating_sums   = np.array(R_sparse.sum(axis=1)).flatten()
    rating_counts = np.diff(R_sparse.indptr)   # nnz per row (CSR)
    rating_counts = np.maximum(rating_counts, 1)   # avoid /0
    user_means    = rating_sums / rating_counts    # shape: (n_users,)

    # ── Mean-centre sparse matrix (subtract user mean from nonzero entries) ──
    # We do this by scaling each row's data values
    R_centered = R_sparse.copy().astype(np.float64)
    for i in range(n_users):
        start, end = R_centered.indptr[i], R_centered.indptr[i + 1]
        R_centered.data[start:end] -= user_means[i]

    # ── SVD (works directly on sparse) ────────────────────────────────
    k = min(50, n_users - 1, n_recipes - 1)
    U, sigma, Vt = svds(R_centered, k=k)
    # svds returns ascending — reverse to descending
    U, sigma, Vt = U[:, ::-1], sigma[::-1], Vt[::-1, :]
    # Store factors — small matrices, never reconstruct full R_pred

    # ── UB-CF: user similarity matrix (n_users × n_users) ─────────────
    # This is ~9k × 9k = manageable (~330 MB float32)
    # Compute row norms from sparse
    row_norms = np.sqrt(np.array(R_centered.multiply(R_centered).sum(axis=1))).flatten()
    row_norms[row_norms == 0] = 1e-9

    # Normalise rows: R_norm = R_centered / row_norms (still sparse)
    from scipy.sparse import diags
    D_inv    = diags(1.0 / row_norms)
    R_normed = D_inv @ R_centered   # sparse normalised matrix

    # User-user cosine similarity — (n_users × n_users), computed as dense
    # since n_users^2 << n_users × n_recipes
    user_sim = (R_normed @ R_normed.T).toarray().astype(np.float32)
    np.fill_diagonal(user_sim, 0)

    # Rated mask as sparse (1 where rated, 0 elsewhere)
    rated_mask_sparse = R_sparse.copy()
    rated_mask_sparse.data = np.ones_like(rated_mask_sparse.data)

    return {
        "user2idx":          user2idx,
        "recipe2idx":        recipe2idx,
        "idx2recipe":        idx2recipe,
        "R_sparse":          R_sparse,          # original sparse ratings
        "R_centered":        R_centered,        # mean-centred sparse
        "rated_mask_sparse": rated_mask_sparse, # 1/0 sparse
        "user_means":        user_means,        # (n_users,)
        "U":                 U,                 # SVD factor
        "sigma":             sigma,             # SVD singular values
        "Vt":                Vt,                # SVD factor
        "user_sim":          user_sim,          # (n_users, n_users) float32
        "all_users":         all_users,
        "all_recipes":       all_recipes,
        "n_users":           n_users,
        "n_recipes":         n_recipes,
    }


def get_top_n(user_id, model_name, models, top_n=20):
    """
    Compute predicted ratings for ONE user on demand (no full matrix in memory).
    Returns list of (recipe_id, score) sorted by score descending.
    """
    if user_id not in models["user2idx"]:
        return []

    u_idx = models["user2idx"][user_id]

    # Recipes this user has already rated — exclude from recommendations
    already_rated = set(models["R_sparse"][u_idx].indices)

    if model_name == "SVD":
        # pred_row = U[u_idx] @ diag(sigma) @ Vt + user_mean
        # Shape: (n_recipes,) — computed from small factor matrices
        user_vec = models["U"][u_idx] * models["sigma"]   # (k,)
        pred_row = user_vec @ models["Vt"]                # (n_recipes,)
        pred_row = np.clip(pred_row + models["user_means"][u_idx], 1.0, 5.0)

    else:  # User-Based CF
        # Vectorized per-user prediction using stored similarity row
        sim_row  = models["user_sim"][u_idx].astype(np.float64)   # (n_users,)
        sim_pos  = np.maximum(sim_row, 0)

        # numerator: sim_pos @ R_centered  → shape (n_recipes,)
        # computed as sparse matrix-vector product (fast)
        numerator = sim_pos @ models["R_centered"]                 # (n_recipes,)
        if hasattr(numerator, "A1"):
            numerator = numerator.A1   # convert matrix to 1D if needed

        # denominator: sum of sim weights per recipe (only over raters)
        denom = sim_pos @ models["rated_mask_sparse"]
        if hasattr(denom, "A1"):
            denom = denom.A1

        with np.errstate(invalid="ignore", divide="ignore"):
            pred_row = np.where(denom > 0, numerator / denom, 0.0)
        pred_row = np.clip(pred_row + models["user_means"][u_idx], 1.0, 5.0)

    # Exclude already-rated recipes
    pred_row[list(already_rated)] = -np.inf

    top_idx = np.argsort(pred_row)[::-1][:top_n]
    return [
        (models["idx2recipe"][i], float(pred_row[i]))
        for i in top_idx if pred_row[i] > -np.inf
    ]


def apply_constraints(candidates, pantry, recipe_ing_map, mode, max_missing=3):
    """Filter and re-rank candidates by ingredient availability."""
    results = []
    for recipe_id, score in candidates:
        if recipe_id not in recipe_ing_map:
            continue
        missing = recipe_ing_map[recipe_id] - pantry
        n_miss  = len(missing)
        if mode == "Hard" and n_miss > 0:
            continue
        if mode == "Soft" and n_miss > max_missing:
            continue
        results.append((recipe_id, score, n_miss, missing))

    if mode == "Soft":
        results.sort(key=lambda x: (x[2], -x[1]))
    return results


def render_recipe_card(rank, recipe_id, score, n_missing, missing_ings, recipe_info):
    """Render a styled recipe card with expandable detail view."""
    if recipe_id not in recipe_info.index:
        return

    row = recipe_info.loc[recipe_id]
    name        = row.get("name", f"Recipe {recipe_id}").title()
    minutes     = int(row.get("minutes", 0))
    n_ingr      = int(row.get("n_ingredients", 0))
    tags_raw    = row.get("tags", "")
    tags        = [t.strip() for t in tags_raw.split("|")] if isinstance(tags_raw, str) else []
    ing_raw     = row.get("ingredients", "")
    all_ings    = [i.strip() for i in ing_raw.split("|")] if isinstance(ing_raw, str) else []
    description = str(row.get("description", "")).strip()
    n_steps     = int(row.get("n_steps", 0))

    # Nutrition
    calories    = row.get("calories", None)
    protein     = row.get("protein_pdv", None)
    carbs       = row.get("carbs_pdv", None)
    fat         = row.get("total_fat_pdv", None)

    hours      = f"{minutes//60}h {minutes%60}m" if minutes >= 60 else f"{minutes}m"
    miss_label = "✓ All ingredients available" if n_missing == 0 else f"{n_missing} ingredient{'s' if n_missing>1 else ''} missing"
    miss_class = "tag-ok" if n_missing == 0 else "tag-missing"

    tag_html  = "".join(f'<span class="tag">{t}</span>' for t in tags[:6])
    miss_html = (
        "".join(f'<span class="tag tag-missing">{i}</span>' for i in list(missing_ings)[:8])
        if missing_ings else ""
    )

    # ── Expander label (acts as the clickable card header) ─────────────
    label = f"#{rank}  ·  {name}  ·  ⏱ {hours}  ·  ⭐ {score:.2f}"

    with st.expander(label, expanded=False):
        # ── Top row: score pill + constraint badge ──────────────────
        st.markdown(
            f'<span class="pill-score">⭐ {score:.2f} predicted</span>'
            f'&nbsp;<span class="tag {miss_class}">{miss_label}</span>',
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Two-column layout ───────────────────────────────────────
        col_left, col_right = st.columns([1, 1])

        with col_left:
            # Tags
            if tags:
                st.markdown("**🏷️ Tags**")
                st.markdown(tag_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            # Ingredients — highlight available vs missing
            if all_ings:
                st.markdown("**🧂 Ingredients**")
                for ing in all_ings:
                    is_missing = ing.lower() in {m.lower() for m in missing_ings}
                    icon  = "❌" if is_missing else "✅"
                    color = "#e74c3c" if is_missing else "#2ecc71"
                    st.markdown(
                        f'<span style="color:{color};font-size:0.88rem">{icon} {ing}</span><br>',
                        unsafe_allow_html=True
                    )

        with col_right:
            # Nutrition box
            if any(v is not None for v in [calories, protein, carbs, fat]):
                st.markdown("**📊 Nutrition (per serving)**")
                nutr_cols = st.columns(2)
                if calories is not None:
                    nutr_cols[0].metric("Calories", f"{calories:.0f} kcal")
                if protein is not None:
                    nutr_cols[1].metric("Protein", f"{protein:.0f}% DV")
                if carbs is not None:
                    nutr_cols[0].metric("Carbs", f"{carbs:.0f}% DV")
                if fat is not None:
                    nutr_cols[1].metric("Total Fat", f"{fat:.0f}% DV")
                st.markdown("<br>", unsafe_allow_html=True)

            # Recipe stats
            st.markdown("**📋 Recipe Info**")
            st.markdown(
                f'<div class="recipe-meta">'
                f'⏱ Cook time: <strong style="color:var(--text)">{hours}</strong><br>'
                f'🧂 Ingredients: <strong style="color:var(--text)">{n_ingr}</strong><br>'
                f'📝 Steps: <strong style="color:var(--text)">{n_steps}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )

            # Description
            if description and description.lower() != "nan" and len(description) > 5:
                st.markdown("<br>**📖 Description**")
                st.markdown(
                    f'<p style="font-size:0.85rem;color:var(--muted);line-height:1.6">'
                    f'{description[:400]}{"..." if len(description) > 400 else ""}'
                    f'</p>',
                    unsafe_allow_html=True
                )

        # Missing ingredients row (if any)
        if miss_html:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(
                f'<span style="font-size:0.8rem;color:var(--muted)">🛒 Still need: </span>{miss_html}',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
def main():
    # ── Header ──────────────────────────────
    st.markdown("<h1>🍳 Recipe Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<h3>CSC 577 · Team Rocket</h3>", unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # ── Load data ───────────────────────────
    if not os.path.exists(f"{DATA_DIR}/interactions_clean.csv"):
        st.error(
            "❌ Processed data not found. Run notebooks 01–02 first to generate "
            f"`{DATA_DIR}/interactions_clean.csv` and `{DATA_DIR}/recipes_clean.csv`."
        )
        return

    with st.spinner("Loading data and building models..."):
        interactions, recipes, recipe_ing_map, recipe_info = load_data()
        models = build_models(interactions)

    # ── Sidebar controls ─────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        cf_model = st.radio(
            "Recommendation model",
            ["SVD", "User-Based CF"],
            help="SVD captures latent taste patterns. UB-CF uses neighbor similarity."
        )

        constraint_mode = st.radio(
            "Ingredient constraint",
            ["None", "Hard (0 missing)", "Soft (allow some missing)"],
        )

        max_missing = 3
        if constraint_mode == "Soft (allow some missing)":
            max_missing = st.slider("Max missing ingredients", 1, 5, 3)

        top_n = st.slider("Recipes to show", 5, 20, 10)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f"**Dataset stats**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{interactions['user_id'].nunique():,}</div>
                <div class="stat-label">Users</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{interactions['recipe_id'].nunique():,}</div>
                <div class="stat-label">Recipes</div>
            </div>""", unsafe_allow_html=True)

    # ── Main tabs ───────────────────────────
    tab1, tab2 = st.tabs(["🧂 By Ingredients", "👤 By User ID"])

    # ── Dietary restriction definitions ─────
    DIETARY_RESTRICTIONS = {
        "🥩 No Pork (Halal-friendly)": [
            "pork", "bacon", "ham", "prosciutto", "salami", "pancetta",
            "lard", "pork belly", "pork chops", "pork loin", "pork ribs",
            "pork shoulder", "sausage", "chorizo", "pepperoni", "hot dog",
            "pork rinds", "pork tenderloin", "ground pork", "spare ribs",
            "canadian bacon", "pork stock", "pork broth",
        ],
        "🐄 No Red Meat": [
            "beef", "steak", "ground beef", "lamb", "veal", "venison",
            "bison", "mutton", "hamburger", "meatball", "roast beef",
            "beef broth", "beef stock", "chuck", "sirloin", "ribeye",
            "flank steak", "brisket", "short ribs", "rack of lamb",
            "lamb chops", "ground lamb", "corned beef", "beef tenderloin",
        ],
        "🌱 Vegetarian (No Meat)": [
            "chicken", "beef", "pork", "lamb", "turkey", "fish", "shrimp",
            "salmon", "tuna", "crab", "lobster", "scallops", "clams",
            "mussels", "anchovies", "bacon", "ham", "sausage", "pepperoni",
            "ground beef", "steak", "veal", "venison", "duck", "bison",
        ],
        "🌿 Vegan (No Animal Products)": [
            "chicken", "beef", "pork", "lamb", "turkey", "fish", "shrimp",
            "salmon", "tuna", "eggs", "milk", "butter", "cheese", "cream",
            "yogurt", "honey", "gelatin", "lard", "bacon", "ham",
            "sour cream", "heavy cream", "whipped cream", "parmesan",
            "mozzarella", "cheddar", "cream cheese", "mayonnaise",
        ],
        "🌾 Gluten-Free": [
            "flour", "wheat", "bread", "pasta", "breadcrumbs", "soy sauce",
            "barley", "rye", "semolina", "couscous", "bulgur", "farro",
            "wheat germ", "wheat bran", "all-purpose flour", "bread flour",
            "cake flour", "whole wheat flour", "self-rising flour",
        ],
        "🥛 Dairy-Free": [
            "milk", "butter", "cheese", "cream", "yogurt", "sour cream",
            "heavy cream", "whipped cream", "parmesan", "mozzarella",
            "cheddar", "cream cheese", "half and half", "buttermilk",
            "ice cream", "ghee", "condensed milk", "evaporated milk",
        ],
    }

    # ── TAB 1: Ingredient-based ─────────────
    with tab1:
        st.markdown("## What's in your pantry?")
        st.markdown("Select your available ingredients and dietary preferences below.")

        # ── Dietary Restrictions ─────────────────────────────────────
        st.markdown("### 🚫 Dietary Restrictions")
        st.markdown("<small style='color:var(--muted)'>Select restrictions to automatically exclude recipes containing those ingredients.</small>", unsafe_allow_html=True)

        restriction_cols = st.columns(3)
        selected_restrictions = []
        for i, (label, _) in enumerate(DIETARY_RESTRICTIONS.items()):
            with restriction_cols[i % 3]:
                if st.checkbox(label, key=f"restrict_{i}"):
                    selected_restrictions.append(label)

        # Build the set of banned ingredient keywords from selected restrictions
        banned_keywords = set()
        for r in selected_restrictions:
            banned_keywords.update(DIETARY_RESTRICTIONS[r])

        if banned_keywords:
            st.markdown(
                f"<small style='color:var(--accent)'>⚠ Excluding recipes containing "
                f"{len(banned_keywords)} restricted ingredients.</small>",
                unsafe_allow_html=True
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # ── Ingredient selector ───────────────────────────────────────
        st.markdown("### 🧂 Your Available Ingredients")

        # Get all unique ingredients across the dataset for the dropdown
        all_ingredients_list = sorted(set(
            ing
            for ing_set in recipe_ing_map.values()
            for ing in ing_set
        ))

        # Quick preset buttons
        PRESETS = {
            "🍗 Chicken & Basics":   ["chicken breast", "garlic", "olive oil", "salt", "pepper", "onion", "lemon juice", "butter", "flour", "eggs"],
            "🌿 Vegetarian Staples": ["tomatoes", "garlic", "olive oil", "onion", "bell pepper", "basil", "oregano", "salt", "pepper", "pasta"],
            "🍣 Seafood Pantry":     ["salmon", "garlic", "olive oil", "lemon juice", "butter", "salt", "pepper", "dill", "capers", "white wine"],
            "🧁 Baking Essentials":  ["flour", "sugar", "butter", "eggs", "milk", "baking powder", "vanilla extract", "salt", "brown sugar"],
            "🥘 Slow Cook Staples":  ["beef", "onion", "garlic", "tomatoes", "beef broth", "carrots", "potatoes", "thyme", "bay leaves"],
        }

        preset_cols = st.columns(len(PRESETS))
        for col, (label, ings) in zip(preset_cols, PRESETS.items()):
            with col:
                if st.button(label, key=f"preset_{label}"):
                    # Only add preset ingredients that exist in the dataset
                    valid = [i for i in ings if i in all_ingredients_list]
                    existing = st.session_state.get("selected_ings", [])
                    st.session_state["selected_ings"] = list(set(existing + valid))

        # Multiselect dropdown — full ingredient list from the dataset
        selected_ings = st.multiselect(
            "Search and select ingredients",
            options=all_ingredients_list,
            default=st.session_state.get("selected_ings", []),
            placeholder="Type to search ingredients...",
            key="ing_multiselect",
        )
        # Sync to session state
        st.session_state["selected_ings"] = selected_ings

        if selected_ings:
            st.markdown(
                f"<small style='color:var(--muted)'>{len(selected_ings)} ingredient(s) selected</small>",
                unsafe_allow_html=True
            )

        if st.button("🔍 Find Recipes", key="btn_ing"):
            if not selected_ings:
                st.warning("Please select at least one ingredient.")
            else:
                pantry = frozenset(i.lower() for i in selected_ings)
                st.markdown(f"**Pantry:** {len(pantry)} ingredients loaded")

                # Apply dietary restriction filter first
                def is_allowed(recipe_id):
                    if not banned_keywords:
                        return True
                    ing_set = recipe_ing_map.get(recipe_id, frozenset())
                    return not any(
                        any(banned in ing for banned in banned_keywords)
                        for ing in ing_set
                    )

                # Pick the most active user to drive CF recommendations
                user_counts = interactions["user_id"].value_counts()
                demo_user   = user_counts.index[0]

                candidates = get_top_n(demo_user, cf_model, models, top_n=200)

                # Filter by dietary restrictions
                candidates = [(rid, score) for rid, score in candidates if is_allowed(rid)]

                # Apply ingredient constraints
                mode_key = (
                    "None" if constraint_mode == "None"
                    else "Hard" if "Hard" in constraint_mode
                    else "Soft"
                )

                if mode_key == "None":
                    results = []
                    for recipe_id, score in candidates:
                        if recipe_id in recipe_ing_map:
                            missing = recipe_ing_map[recipe_id] - pantry
                            results.append((recipe_id, score, len(missing), missing))
                    results = results[:top_n]
                else:
                    results = apply_constraints(
                        candidates, pantry, recipe_ing_map, mode_key, max_missing
                    )[:top_n]

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                if not results:
                    st.warning(
                        "No recipes found matching your constraint. "
                        "Try relaxing the constraint or adding more ingredients."
                    )
                else:
                    # Summary stats
                    n_zero  = sum(1 for r in results if r[2] == 0)
                    avg_missing = np.mean([r[2] for r in results])
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f'<div class="stat-box"><div class="stat-num">{len(results)}</div><div class="stat-label">Recipes found</div></div>', unsafe_allow_html=True)
                    c2.markdown(f'<div class="stat-box"><div class="stat-num">{n_zero}</div><div class="stat-label">Fully makeable</div></div>', unsafe_allow_html=True)
                    c3.markdown(f'<div class="stat-box"><div class="stat-num">{avg_missing:.1f}</div><div class="stat-label">Avg missing</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    for rank, (recipe_id, score, n_miss, missing_ings) in enumerate(results, 1):
                        render_recipe_card(rank, recipe_id, score, n_miss, missing_ings, recipe_info)

    # ── TAB 2: User ID-based ─────────────────
    with tab2:
        st.markdown("## Browse by User")
        st.markdown("Enter a user ID from the dataset to see their personalized recommendations.")

        # Sample user IDs for convenience
        sample_users = interactions["user_id"].value_counts().head(20).index.tolist()

        col_x, col_y = st.columns([2, 1])
        with col_x:
            user_input = st.text_input(
                "User ID",
                placeholder=f"e.g. {sample_users[0]}",
            )
        with col_y:
            st.markdown("**Or pick a sample user**")
            selected_sample = st.selectbox(
                "Active users",
                options=["— select —"] + [str(u) for u in sample_users[:10]],
                label_visibility="collapsed",
            )
            if selected_sample != "— select —":
                user_input = selected_sample

        if st.button("🔍 Get Recommendations", key="btn_user"):
            try:
                user_id = int(user_input)
            except (ValueError, TypeError):
                st.error("Please enter a valid numeric User ID.")
                st.stop()

            if user_id not in models["user2idx"]:
                st.error(f"User {user_id} not found in the training data.")
                st.stop()

            # User stats
            user_history = interactions[interactions["user_id"] == user_id]
            avg_rating   = user_history["rating"].mean()
            n_rated      = len(user_history)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="stat-box"><div class="stat-num">{n_rated}</div><div class="stat-label">Recipes rated</div></div>', unsafe_allow_html=True)
            c2.markdown(f'<div class="stat-box"><div class="stat-num">{avg_rating:.1f}★</div><div class="stat-label">Avg rating</div></div>', unsafe_allow_html=True)
            c3.markdown(f'<div class="stat-box"><div class="stat-num">{cf_model}</div><div class="stat-label">Model used</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            candidates = get_top_n(user_id, cf_model, models, top_n=200)

            # Apply dietary restriction filter
            def is_allowed_tab2(recipe_id):
                if not banned_keywords:
                    return True
                ing_set = recipe_ing_map.get(recipe_id, frozenset())
                return not any(
                    any(banned in ing for banned in banned_keywords)
                    for ing in ing_set
                )
            candidates = [(rid, score) for rid, score in candidates if is_allowed_tab2(rid)]

            # Apply constraint if selected
            mode_key = (
                "None" if constraint_mode == "None"
                else "Hard" if "Hard" in constraint_mode
                else "Soft"
            )

            if mode_key == "None":
                # No pantry available for user-browse mode — show all with n/a missing
                results = [(rid, score, 0, frozenset()) for rid, score in candidates[:top_n]]
            else:
                # Simulate the user's pantry from their history
                user_recipes  = user_history["recipe_id"].unique()
                all_ings      = set()
                for rid in user_recipes:
                    if rid in recipe_ing_map:
                        all_ings.update(recipe_ing_map[rid])
                rng    = np.random.default_rng(42)
                n_keep = max(1, int(len(all_ings) * 0.7))
                pantry = frozenset(rng.choice(list(all_ings), size=n_keep, replace=False))

                results = apply_constraints(
                    candidates, pantry, recipe_ing_map, mode_key, max_missing
                )[:top_n]

                if mode_key != "None":
                    st.info(f"Using simulated pantry of {len(pantry)} ingredients based on this user's rating history.")

            if not results:
                st.warning("No recipes found. Try relaxing the ingredient constraint.")
            else:
                for rank, (recipe_id, score, n_miss, missing_ings) in enumerate(results, 1):
                    render_recipe_card(rank, recipe_id, score, n_miss, missing_ings, recipe_info)


if __name__ == "__main__":
    main()