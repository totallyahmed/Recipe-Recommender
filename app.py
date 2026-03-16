"""
Team Rocket — Food.com Recipe Recommender
Streamlit UI  |  CSC 577
"""

import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, diags
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
# Custom CSS — dark-food editorial theme
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
.pill-match {
    display: inline-block;
    background: rgba(39,174,96,0.15);
    color: #2ecc71;
    border: 1px solid rgba(39,174,96,0.3);
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

    # Parse ingredients into sets
    recipes["ingredient_set"] = recipes["ingredients"].apply(
        lambda x: frozenset(i.strip().lower() for i in x.split("|"))
        if isinstance(x, str) else frozenset()
    )
    recipe_ing_map = dict(zip(recipes["recipe_id"], recipes["ingredient_set"]))
    recipe_info    = recipes.set_index("recipe_id")

    # Precompute recipe quality for ingredient-based ranking
    stats = interactions.groupby("recipe_id").agg(
        avg_rating=("rating", "mean"),
        n_ratings=("rating", "count"),
    ).reset_index()
    recipe_quality  = dict(zip(stats["recipe_id"], stats["avg_rating"]))
    recipe_n_ratings = dict(zip(stats["recipe_id"], stats["n_ratings"]))

    return interactions, recipes, recipe_ing_map, recipe_info, recipe_quality, recipe_n_ratings


@st.cache_resource(show_spinner=False)
def build_models(interactions):
    """Build CF models from interactions. Used for the By User tab."""
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

    R_sparse = csr_matrix(
        (interactions["rating"].astype(np.float64).values,
         (interactions["u_idx"].values, interactions["r_idx"].values)),
        shape=(n_users, n_recipes)
    )

    rating_sums   = np.array(R_sparse.sum(axis=1)).flatten()
    rating_counts = np.diff(R_sparse.indptr)
    rating_counts = np.maximum(rating_counts, 1)
    user_means    = rating_sums / rating_counts

    R_centered = R_sparse.copy().astype(np.float64)
    for i in range(n_users):
        s, e = R_centered.indptr[i], R_centered.indptr[i + 1]
        R_centered.data[s:e] -= user_means[i]

    k = min(50, n_users - 1, n_recipes - 1)
    U, sigma, Vt = svds(R_centered, k=k)
    U, sigma, Vt = U[:, ::-1], sigma[::-1], Vt[::-1, :]

    row_norms = np.sqrt(np.array(R_centered.multiply(R_centered).sum(axis=1))).flatten()
    row_norms[row_norms == 0] = 1e-9
    R_normed = diags(1.0 / row_norms) @ R_centered
    user_sim = (R_normed @ R_normed.T).toarray().astype(np.float32)
    np.fill_diagonal(user_sim, 0)

    rated_mask_sparse = R_sparse.copy()
    rated_mask_sparse.data = np.ones_like(rated_mask_sparse.data)

    return {
        "user2idx": user2idx, "recipe2idx": recipe2idx, "idx2recipe": idx2recipe,
        "R_sparse": R_sparse, "R_centered": R_centered,
        "rated_mask_sparse": rated_mask_sparse, "user_means": user_means,
        "U": U, "sigma": sigma, "Vt": Vt, "user_sim": user_sim,
        "n_users": n_users, "n_recipes": n_recipes,
    }


# ─────────────────────────────────────────────
# Ingredient matching (substring-based)
# ─────────────────────────────────────────────

def ingredient_covered(recipe_ingredient, pantry_terms):
    """
    Check if a recipe ingredient is covered by the user's pantry.
    Uses substring matching: if the user selected 'chicken', it covers
    'chicken breast', 'boneless skinless chicken thighs', 'chicken stock', etc.
    Also works the other way: if user selected 'boneless chicken breast'
    it covers a recipe asking for 'chicken'.
    """
    ri = recipe_ingredient.lower()
    for term in pantry_terms:
        if term in ri or ri in term:
            return True
    return False


def compute_missing(ing_set, pantry_terms):
    """Return the set of recipe ingredients NOT covered by pantry (substring match)."""
    return frozenset(ing for ing in ing_set if not ingredient_covered(ing, pantry_terms))


# ─────────────────────────────────────────────
# Recommendation functions
# ─────────────────────────────────────────────

def get_top_n(user_id, model_name, models, top_n=20):
    """CF predictions for one user. Used by the By User tab."""
    if user_id not in models["user2idx"]:
        return []

    u_idx = models["user2idx"][user_id]
    already_rated = set(models["R_sparse"][u_idx].indices)

    if model_name == "SVD":
        user_vec = models["U"][u_idx] * models["sigma"]
        pred_row = user_vec @ models["Vt"]
        pred_row = np.clip(pred_row + models["user_means"][u_idx], 1.0, 5.0)
    else:
        sim_pos = np.maximum(models["user_sim"][u_idx].astype(np.float64), 0)
        numerator = sim_pos @ models["R_centered"]
        if hasattr(numerator, "A1"):
            numerator = numerator.A1
        denom = sim_pos @ models["rated_mask_sparse"]
        if hasattr(denom, "A1"):
            denom = denom.A1
        with np.errstate(invalid="ignore", divide="ignore"):
            pred_row = np.where(denom > 0, numerator / denom, 0.0)
        pred_row = np.clip(pred_row + models["user_means"][u_idx], 1.0, 5.0)

    pred_row[list(already_rated)] = -np.inf
    top_idx = np.argsort(pred_row)[::-1][:top_n]
    return [
        (models["idx2recipe"][i], float(pred_row[i]))
        for i in top_idx if pred_row[i] > -np.inf
    ]


def find_by_ingredients(pantry, recipe_ing_map, recipe_quality, recipe_n_ratings,
                        banned_keywords, mode, max_missing, top_n):
    """
    Ingredient-driven recipe search. Ranks by ingredient coverage × recipe quality.
    
    Uses substring matching: selecting 'chicken' covers 'chicken breast',
    'boneless skinless chicken thighs', 'chicken stock', etc.
    
    Score = 0.6 × coverage + 0.4 × (bayesian_avg / 5)
    """
    pantry_terms = [i.lower() for i in pantry]
    results = []

    for recipe_id, ing_set in recipe_ing_map.items():
        if not ing_set:
            continue

        # Dietary check
        if banned_keywords:
            if any(any(b in ing for b in banned_keywords) for ing in ing_set):
                continue

        missing = compute_missing(ing_set, pantry_terms)
        n_missing = len(missing)

        # Constraint
        if mode == "Hard" and n_missing > 0:
            continue
        if mode == "Soft" and n_missing > max_missing:
            continue

        coverage = 1.0 - (n_missing / len(ing_set))

        # Bayesian average (prior=4.6, weight=5)
        avg_r = recipe_quality.get(recipe_id, 3.0)
        n_r   = recipe_n_ratings.get(recipe_id, 0)
        bayesian = (n_r * avg_r + 5 * 4.6) / (n_r + 5)

        score = coverage * 0.6 + (bayesian / 5.0) * 0.4
        results.append((recipe_id, round(score, 4), n_missing, missing))

    results.sort(key=lambda x: (x[2], -x[1]))
    return results[:top_n]


def apply_constraints(candidates, pantry, recipe_ing_map, mode, max_missing=3):
    """Filter CF candidates by ingredient availability. Substring matching."""
    pantry_terms = [p.lower() for p in pantry]
    results = []
    for recipe_id, score in candidates:
        if recipe_id not in recipe_ing_map:
            continue
        missing = compute_missing(recipe_ing_map[recipe_id], pantry_terms)
        n_miss  = len(missing)
        if mode == "Hard" and n_miss > 0:
            continue
        if mode == "Soft" and n_miss > max_missing:
            continue
        results.append((recipe_id, score, n_miss, missing))

    if mode == "Soft":
        results.sort(key=lambda x: (x[2], -x[1]))
    return results


# ─────────────────────────────────────────────
# Recipe card renderer
# ─────────────────────────────────────────────

def render_recipe_card(rank, recipe_id, score, n_missing, missing_ings, recipe_info,
                       score_label="predicted", pantry=None):
    if recipe_id not in recipe_info.index:
        return

    row = recipe_info.loc[recipe_id]
    name    = row.get("name", f"Recipe {recipe_id}").title()
    minutes = int(row.get("minutes", 0))
    n_ingr  = int(row.get("n_ingredients", 0))
    tags_raw = row.get("tags", "")
    tags     = [t.strip() for t in tags_raw.split("|")] if isinstance(tags_raw, str) else []
    ing_raw  = row.get("ingredients", "")
    all_ings = [i.strip() for i in ing_raw.split("|")] if isinstance(ing_raw, str) else []
    description = str(row.get("description", "")).strip()
    n_steps = int(row.get("n_steps", 0))

    calories = row.get("calories", None)
    protein  = row.get("protein_pdv", None)
    carbs    = row.get("carbs_pdv", None)
    fat      = row.get("total_fat_pdv", None)

    hours = f"{minutes // 60}h {minutes % 60}m" if minutes >= 60 else f"{minutes}m"
    miss_label = ("✓ All ingredients available" if n_missing == 0
                  else f"{n_missing} ingredient{'s' if n_missing > 1 else ''} missing")
    miss_class = "tag-ok" if n_missing == 0 else "tag-missing"

    # Coverage in header for ingredient tab
    if pantry is not None and len(all_ings) > 0:
        pantry_terms = [p.lower() for p in pantry]
        n_have = sum(1 for i in all_ings if ingredient_covered(i, pantry_terms))
        cov_text = f"  ·  🧂 {n_have}/{len(all_ings)} ingredients ({n_have / len(all_ings) * 100:.0f}%)"
    else:
        cov_text = ""

    if score_label == "predicted":
        label = f"#{rank}  ·  {name}  ·  ⏱ {hours}  ·  ⭐ {score:.2f}"
    else:
        label = f"#{rank}  ·  {name}  ·  ⏱ {hours}{cov_text}"

    tag_html = "".join(f'<span class="tag">{t}</span>' for t in tags[:6])
    miss_html = (
        "".join(f'<span class="tag tag-missing">{i}</span>' for i in list(missing_ings)[:8])
        if missing_ings else ""
    )

    with st.expander(label, expanded=False):
        if score_label == "predicted":
            pill = f'<span class="pill-score">⭐ {score:.2f} predicted</span>'
        else:
            pill = f'<span class="pill-match">🎯 {score:.0%} match</span>'

        st.markdown(
            f'{pill}&nbsp;<span class="tag {miss_class}">{miss_label}</span>',
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        col_left, col_right = st.columns([1, 1])

        with col_left:
            if tags:
                st.markdown("**🏷️ Tags**")
                st.markdown(tag_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

            if all_ings:
                st.markdown("**🧂 Ingredients**")
                pantry_terms = [p.lower() for p in pantry] if pantry else []
                missing_lower = {m.lower() for m in missing_ings} if missing_ings else set()
                for ing in all_ings:
                    # Check if this ingredient is in the missing set,
                    # or fall back to substring match against pantry
                    if missing_lower:
                        is_missing = ing.lower() in missing_lower
                    elif pantry_terms:
                        is_missing = not ingredient_covered(ing, pantry_terms)
                    else:
                        is_missing = False
                    icon  = "❌" if is_missing else "✅"
                    color = "#e74c3c" if is_missing else "#2ecc71"
                    st.markdown(
                        f'<span style="color:{color};font-size:0.88rem">{icon} {ing}</span><br>',
                        unsafe_allow_html=True
                    )

        with col_right:
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

            st.markdown("**📋 Recipe Info**")
            st.markdown(
                f'<div class="recipe-meta">'
                f'⏱ Cook time: <strong style="color:var(--text)">{hours}</strong><br>'
                f'🧂 Ingredients: <strong style="color:var(--text)">{n_ingr}</strong><br>'
                f'📝 Steps: <strong style="color:var(--text)">{n_steps}</strong>'
                f'</div>',
                unsafe_allow_html=True
            )

            if description and description.lower() != "nan" and len(description) > 5:
                st.markdown("<br>**📖 Description**")
                st.markdown(
                    f'<p style="font-size:0.85rem;color:var(--muted);line-height:1.6">'
                    f'{description[:400]}{"..." if len(description) > 400 else ""}'
                    f'</p>',
                    unsafe_allow_html=True
                )

        if miss_html:
            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            st.markdown(
                f'<span style="font-size:0.8rem;color:var(--muted)">🛒 Still need: </span>{miss_html}',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# Dietary restrictions
# ─────────────────────────────────────────────

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


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
def main():
    st.markdown("<h1>🍳 Recipe Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<h3>CSC 577 · Team Rocket</h3>", unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    if not os.path.exists(f"{DATA_DIR}/interactions_clean.csv"):
        st.error(
            "❌ Processed data not found. Run the notebook first to generate "
            f"`{DATA_DIR}/interactions_clean.csv` and `{DATA_DIR}/recipes_clean.csv`."
        )
        return

    with st.spinner("Loading data and building models..."):
        interactions, recipes, recipe_ing_map, recipe_info, recipe_quality, recipe_n_ratings = load_data()
        models = build_models(interactions)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        cf_model = st.radio(
            "CF model (User tab)",
            ["SVD", "User-Based CF"],
            help="Applies to the 'By User ID' tab. The ingredient tab uses ingredient matching."
        )

        constraint_mode = st.radio(
            "Ingredient constraint",
            ["None", "Hard (0 missing)", "Soft (allow some missing)"],
        )

        max_missing = 3
        if constraint_mode == "Soft (allow some missing)":
            max_missing = st.slider("Max missing ingredients", 1, 10, 3)

        top_n = st.slider("Recipes to show", 5, 20, 10)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**Dataset**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{interactions['user_id'].nunique():,}</div>
                <div class="stat-label">Users</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-num">{interactions['recipe_id'].nunique():,}</div>
                <div class="stat-label">Recipes</div>
            </div>""", unsafe_allow_html=True)

    mode_key = (
        "None" if constraint_mode == "None"
        else "Hard" if "Hard" in constraint_mode
        else "Soft"
    )

    tab1, tab2 = st.tabs(["🧂 By Ingredients", "👤 By User ID"])

    # ══════════════════════════════════════════
    # TAB 1: By Ingredients
    # ══════════════════════════════════════════
    with tab1:
        st.markdown("## What's in your pantry?")
        st.markdown("Select your available ingredients — we'll find recipes you can actually make, ranked by how well they match what you have.")

        # Dietary restrictions
        st.markdown("### 🚫 Dietary Restrictions")
        st.markdown(
            "<small style='color:var(--muted)'>Exclude recipes containing restricted ingredients.</small>",
            unsafe_allow_html=True
        )

        rcols = st.columns(3)
        sel_r = []
        for i, (lab, _) in enumerate(DIETARY_RESTRICTIONS.items()):
            with rcols[i % 3]:
                if st.checkbox(lab, key=f"r_{i}"):
                    sel_r.append(lab)

        banned = set()
        for r in sel_r:
            banned.update(DIETARY_RESTRICTIONS[r])

        if banned:
            st.markdown(
                f"<small style='color:var(--accent)'>⚠ Excluding recipes with "
                f"{len(banned)} restricted ingredients.</small>",
                unsafe_allow_html=True
            )

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 🧂 Your Available Ingredients")

        all_ingredients_list = sorted(set(
            ing for ing_set in recipe_ing_map.values() for ing in ing_set
        ))

        PRESETS = {
            "🍗 Chicken & Basics":   ["chicken breast", "garlic", "olive oil", "salt", "pepper", "onion", "lemon juice", "butter", "flour", "eggs"],
            "🌿 Vegetarian Staples": ["tomatoes", "garlic", "olive oil", "onion", "bell pepper", "basil", "oregano", "salt", "pepper", "pasta", "rice", "canned tomatoes"],
            "🍣 Seafood Pantry":     ["salmon", "garlic", "olive oil", "lemon juice", "butter", "salt", "pepper", "dill", "capers", "white wine"],
            "🧁 Baking Essentials":  ["flour", "sugar", "butter", "eggs", "milk", "baking powder", "vanilla extract", "salt", "brown sugar", "baking soda"],
            "🥘 Slow Cook Staples":  ["beef", "onion", "garlic", "tomatoes", "beef broth", "carrots", "potatoes", "thyme", "bay leaves", "salt", "pepper"],
        }

        pcols = st.columns(len(PRESETS))
        for col, (lab, ings) in zip(pcols, PRESETS.items()):
            with col:
                if st.button(lab, key=f"p_{lab}"):
                    valid = [i for i in ings if i in all_ingredients_list]
                    existing = st.session_state.get("sel_ings", [])
                    st.session_state["sel_ings"] = list(set(existing + valid))

        selected = st.multiselect(
            "Search and select ingredients",
            options=all_ingredients_list,
            default=st.session_state.get("sel_ings", []),
            placeholder="Type to search ingredients...",
            key="ms_ings",
        )
        st.session_state["sel_ings"] = selected

        if selected:
            st.markdown(
                f"<small style='color:var(--muted)'>{len(selected)} ingredient(s) selected</small>",
                unsafe_allow_html=True
            )

        if st.button("🔍 Find Recipes", key="btn_ing"):
            if not selected:
                st.warning("Please select at least one ingredient.")
            else:
                results = find_by_ingredients(
                    selected, recipe_ing_map, recipe_quality, recipe_n_ratings,
                    banned, mode_key, max_missing, top_n
                )

                st.markdown('<hr class="divider">', unsafe_allow_html=True)

                if not results:
                    st.warning("No recipes found. Try adding more ingredients or switching to Soft constraint.")
                else:
                    n_perf = sum(1 for r in results if r[2] == 0)
                    avg_m  = np.mean([r[2] for r in results])

                    s1, s2, s3 = st.columns(3)
                    s1.markdown(
                        f'<div class="stat-box"><div class="stat-num">{len(results)}</div>'
                        f'<div class="stat-label">Recipes found</div></div>', unsafe_allow_html=True)
                    s2.markdown(
                        f'<div class="stat-box"><div class="stat-num">{n_perf}</div>'
                        f'<div class="stat-label">Fully makeable</div></div>', unsafe_allow_html=True)
                    s3.markdown(
                        f'<div class="stat-box"><div class="stat-num">{avg_m:.1f}</div>'
                        f'<div class="stat-label">Avg missing</div></div>', unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    for rank, (rid, score, nm, miss) in enumerate(results, 1):
                        render_recipe_card(rank, rid, score, nm, miss, recipe_info,
                                           score_label="match", pantry=selected)

    # ══════════════════════════════════════════
    # TAB 2: By User ID
    # ══════════════════════════════════════════
    with tab2:
        st.markdown("## Browse by User")
        st.markdown("Enter a user ID to get personalized CF recommendations based on their rating history.")

        sample_users = interactions["user_id"].value_counts().head(20).index.tolist()

        cx, cy = st.columns([2, 1])
        with cx:
            user_input = st.text_input("User ID", placeholder=f"e.g. {sample_users[0]}")
        with cy:
            st.markdown("**Or pick a sample user**")
            sel_sample = st.selectbox(
                "Active users",
                options=["— select —"] + [str(u) for u in sample_users[:10]],
                label_visibility="collapsed",
            )
            if sel_sample != "— select —":
                user_input = sel_sample

        # Dietary restrictions for tab 2
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("### 🚫 Dietary Restrictions")
        rcols2 = st.columns(3)
        sel_r2 = []
        for i, (lab, _) in enumerate(DIETARY_RESTRICTIONS.items()):
            with rcols2[i % 3]:
                if st.checkbox(lab, key=f"r2_{i}"):
                    sel_r2.append(lab)

        banned2 = set()
        for r in sel_r2:
            banned2.update(DIETARY_RESTRICTIONS[r])

        if st.button("🔍 Get Recommendations", key="btn_user"):
            try:
                user_id = int(user_input)
            except (ValueError, TypeError):
                st.error("Please enter a valid numeric User ID.")
                st.stop()

            if user_id not in models["user2idx"]:
                st.error(f"User {user_id} not found in the training data.")
                st.stop()

            user_hist = interactions[interactions["user_id"] == user_id]
            avg_r = user_hist["rating"].mean()
            n_rated = len(user_hist)

            st.markdown('<hr class="divider">', unsafe_allow_html=True)
            s1, s2, s3 = st.columns(3)
            s1.markdown(
                f'<div class="stat-box"><div class="stat-num">{n_rated}</div>'
                f'<div class="stat-label">Recipes rated</div></div>', unsafe_allow_html=True)
            s2.markdown(
                f'<div class="stat-box"><div class="stat-num">{avg_r:.1f}★</div>'
                f'<div class="stat-label">Avg rating</div></div>', unsafe_allow_html=True)
            s3.markdown(
                f'<div class="stat-box"><div class="stat-num">{cf_model}</div>'
                f'<div class="stat-label">Model</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            candidates = get_top_n(user_id, cf_model, models, top_n=200)

            # Dietary filter
            if banned2:
                def ok(rid):
                    ings = recipe_ing_map.get(rid, frozenset())
                    return not any(any(b in ing for b in banned2) for ing in ings)
                candidates = [(rid, s) for rid, s in candidates if ok(rid)]

            if mode_key == "None":
                results = [(rid, sc, 0, frozenset()) for rid, sc in candidates[:top_n]]
            else:
                # Simulate pantry from user's history
                urecs = user_hist["recipe_id"].unique()
                all_i = set()
                for rid in urecs:
                    if rid in recipe_ing_map:
                        all_i.update(recipe_ing_map[rid])
                rng = np.random.default_rng(42)
                nk = max(1, int(len(all_i) * 0.7))
                pantry = frozenset(rng.choice(list(all_i), size=nk, replace=False))

                results = apply_constraints(
                    candidates, pantry, recipe_ing_map, mode_key, max_missing
                )[:top_n]

                st.info(f"Simulated pantry: {len(pantry)} ingredients from this user's history.")

            if not results:
                st.warning("No recipes found. Try relaxing the ingredient constraint.")
            else:
                for rank, (rid, sc, nm, miss) in enumerate(results, 1):
                    render_recipe_card(rank, rid, sc, nm, miss, recipe_info,
                                       score_label="predicted")


if __name__ == "__main__":
    main()