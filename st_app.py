import streamlit as st
import numpy as np
import plotly.express as px
import time

# ---- Core Attention Functions ----
def compute_qkv(X, Wq, Wk, Wv):
    Q = X @ Wq
    K = X @ Wk
    V = X @ Wv
    return Q, K, V

def split_heads(X, num_heads):
    batch_size, seq_len, d_model = X.shape
    d_k = d_model // num_heads
    reshaped = X.reshape(batch_size, seq_len, num_heads, d_k)
    return reshaped.transpose(0, 2, 1, 3)  # [batch, heads, seq, d_k]

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(0,1,3,2) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    out = weights @ V
    return scores, weights, out

def combine_heads(X):
    batch, heads, seq, d_k = X.shape
    concat = X.transpose(0,2,1,3).reshape(batch, seq, heads * d_k)
    return concat

def create_causal_mask(seq_len, num_heads):
    mask = np.tril(np.ones((seq_len, seq_len), dtype=np.int32))
    return mask[np.newaxis, np.newaxis, :, :]

# ---- Explanations for a 12-year-old ----
EXPLANATIONS = [
    """**Step 1 - The Input Matrix X**: 
    🎒 Think of this like a school notebook where each row is a different word, and each column tells us something about that word (like its meaning, grammar, etc.). 
    
    📊 **Simple Math**: We start with a matrix X that has shape `(sequence_length × d_model)`. Each number in this grid represents a feature of a word at that position.""",
    
    """**Step 2 - Creating Q, K, V (Query, Key, Value)**:
    🔍 Imagine you're looking for your friend in a crowded room:
    - **Q (Query)**: "What am I looking for?" - This is like your search question
    - **K (Key)**: "What can be found?" - These are like name tags on everyone  
    - **V (Value)**: "What information do I get?" - This is the actual information about each person
    
    📊 **Simple Math**: We multiply our input X by three different "recipe cards" (weight matrices):
    - `Q = X × Wq` (Query matrix)
    - `K = X × Wk` (Key matrix)  
    - `V = X × Wv` (Value matrix)""",
    
    """**Step 3 - Splitting into Multiple Heads**:
    👀 Instead of using just one pair of eyes, we give our model multiple pairs of eyes (heads) to look at different things!
    
    🍕 Think of cutting a pizza into slices - we cut our Q, K, V matrices into smaller pieces so each "head" can focus on different patterns.
    
    📊 **Simple Math**: We reshape each matrix from `(seq_len × d_model)` into `(num_heads × seq_len × d_k)` where `d_k = d_model ÷ num_heads`. It's like dividing candy equally among friends!""",
    
    """**Step 4 - The Causal Mask (Looking Back Only)**:
    🚫 This is like a rule in a game: "You can only look at cards that were played before yours, not the future ones!"
    
    📊 **Simple Math**: We create a triangular mask filled with 1s below the diagonal and 0s above:
    ```
    [1, 0, 0]    ← Position 1 can only see itself
    [1, 1, 0]    ← Position 2 can see positions 1 & 2  
    [1, 1, 1]    ← Position 3 can see all previous positions
    ```""",
    
    """**Step 5 - Attention Scores and Weights**:
    🎯 Now we calculate how much each word should "pay attention" to every other word:
    
    **Part A - Calculating Scores**: 
    📊 `Scores = (Q × K^T) ÷ √d_k`
    - We multiply Query by Key (like checking if your search matches the name tags)
    - We divide by √d_k to keep numbers from getting too big (like turning down the volume)
    
    **Part B - Making Probabilities**:
    📊 `Weights = softmax(Scores)` 
    - We use softmax to turn scores into probabilities that add up to 1 (like dividing a pie into percentages)
    
    **Part C - Getting the Result**:
    📊 `Output = Weights × V`
    - We multiply these attention weights by the Values to get our final answer!""",
    
    """**Step 6 - Combining All Heads Together**:
    🧩 Remember how we split everything into multiple heads? Now we put all the puzzle pieces back together!
    
    📊 **Simple Math**: 
    1. **Concatenate heads**: We stick all head outputs side by side like combining separate drawings into one big picture
    2. **Final transformation**: `Final_Output = Combined_Heads × Wo`
    3. This gives us our final result that combines insights from all the different "pairs of eyes" (heads)!
    
    🎉 **The Result**: Each word now has new, smarter representations that consider what all other words were saying!"""
]

# ---- Streamlit App ----
st.set_page_config(layout="wide")
st.title("Interactive Multi-Head Attention Walkthrough 🎥")

# Sidebar controls
seq_len = st.sidebar.slider("Sequence Length", 2, 10, 5)
d_model = st.sidebar.selectbox("Model Dimension (d_model)", [16, 32, 64, 128], index=1)
num_heads = st.sidebar.slider("Number of Heads", 1, 8, 2)
causal = st.sidebar.checkbox("Causal Masking", value=True)
play = st.sidebar.button("▶ Play Animation")
step = st.sidebar.slider("Step", 1, 6, 1)

# Initialize random weights
Wq = np.random.randn(d_model, d_model)
Wk = np.random.randn(d_model, d_model)
Wv = np.random.randn(d_model, d_model)
Wo = np.random.randn(d_model, d_model)

# Generate random input
X = np.random.randn(1, seq_len, d_model)

# Precompute frames
frames = []
labels = [f"{i+1}) {txt.split(':')[0]}" for i, txt in enumerate(EXPLANATIONS)]

# Frame visuals
# 1) Input
fig0 = px.imshow(X[0], labels={'x':'Feature','y':'Position'}).update_layout(title=labels[0])
frames.append((labels[0], [fig0]))
# 2) Q, K, V
Q, K, V = compute_qkv(X, Wq, Wk, Wv)
fig1_Q = px.imshow(Q[0], labels={'x':'Feature','y':'Position'}).update_layout(title='Q')
fig1_K = px.imshow(K[0], labels={'x':'Feature','y':'Position'}).update_layout(title='K')
fig1_V = px.imshow(V[0], labels={'x':'Feature','y':'Position'}).update_layout(title='V')
frames.append((labels[1], [fig1_Q, fig1_K, fig1_V]))
# 3) Split & Transpose
Qh = split_heads(Q, num_heads)
fig2 = px.imshow(Qh[0,0], labels={'x':'Feature','y':'Position'}).update_layout(title=labels[2])
frames.append((labels[2], [fig2]))
# 4) Mask
mask = create_causal_mask(seq_len, num_heads) if causal else None
mask_mat = mask[0,0] if mask is not None else np.ones((seq_len, seq_len))
fig3 = px.imshow(mask_mat, labels={'x':'Key Pos','y':'Query Pos'}).update_layout(title=labels[3])
frames.append((labels[3], [fig3]))
# 5) Attention scores & weights
scores, weights, out_h = scaled_dot_product_attention(Qh, split_heads(K, num_heads), split_heads(V, num_heads), mask)
fig4_scores = px.imshow(scores[0,0], labels={'x':'Key Pos','y':'Query Pos'}).update_layout(title='Scores')
fig4_weights = px.imshow(weights[0,0], labels={'x':'Key Pos','y':'Query Pos'}).update_layout(title='Weights')
frames.append((labels[4], [fig4_scores, fig4_weights]))
# 6) Concat & Output
concat = combine_heads(out_h)
out = concat @ Wo
fig5 = px.imshow(out[0], labels={'x':'Feature','y':'Position'}).update_layout(title=labels[5])
frames.append((labels[5], [fig5]))

# Display logic
if play:
    st.markdown("## 🎬 Animation Playing...")
    for idx, (label, figs) in enumerate(frames):
        # Create containers for this step
        step_container = st.container()
        
        with step_container:
            st.markdown(f"### {label}")
            
            # Show visualizations
            cols = st.columns(len(figs))
            for jdx, fig in enumerate(figs):
                cols[jdx].plotly_chart(fig, use_container_width=True, key=f"play_{idx}_{jdx}")
            
            # Show explanation for this step
            st.markdown("---")
            st.markdown("### 📚 What's Happening:")
            st.markdown(EXPLANATIONS[idx])
            st.markdown("---")
            
            # Give time to read
            time.sleep(3)  # Increased time to read
            
        # Clear this step before moving to next
        step_container.empty()
    
    st.success("🎉 Animation Complete! Use the slider to review any step.")
    step = 1

# Static display for selected step
label, figs = frames[step-1]
st.markdown(f"### {label}")

# Show visualizations
cols = st.columns(len(figs))
for jdx, fig in enumerate(figs):
    cols[jdx].plotly_chart(fig, use_container_width=True, key=f"static_{step-1}_{jdx}")

# Show explanation in a clearly separated section
st.markdown("---")
st.markdown("### 📚 Understanding This Step:")
st.markdown(EXPLANATIONS[step-1])
st.markdown("---")

st.markdown("---")
st.markdown(r"**Attention Formula:** $\mathrm{Attention}(Q,K,V)=\mathrm{softmax}(\frac{QK^T}{\sqrt{d_k}})V$")
