# 🎬 Interactive Multi-Head Attention Visualizer

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-deployed-app-url.com) &nbsp; [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/) &nbsp; [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/) &nbsp; [![Plotly](https://img.shields.io/badge/plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

An interactive educational tool built with Streamlit to visualize the inner workings of the **Multi-Head Self-Attention** mechanism, the core component of Transformer models.

This application provides a step-by-step walkthrough, demystifying complex operations like QKV projection, head splitting, causal masking, and attention weight calculation with clear visualizations and kid-friendly explanations.

---

## 🎥 Live Demo

See the interactive visualizer in action! The animation below walks through each step of the attention calculation, from the initial input to the final output.


[DEMO
](https://github.com/Rishbah-76/Multhead_visualizer/blob/main/demo/Streamlit_multihead.mp4)
*(This video is from `demo/streamlit_multihead.mp4`)*

---

## ✨ Features

- **Interactive Controls**: Adjust sequence length, model dimensions, and the number of attention heads.
- **Step-by-Step Visualization**: Use the slider to navigate through the 6 key stages of the attention mechanism.
- **▶️ Automated Animation**: Play a full animation that walks through each step automatically.
- **Causal Masking**: Toggle causal masking on or off to understand how it prevents future-peeking in decoder blocks.
- **📚 Kid-Friendly Explanations**: Each step is accompanied by simple, intuitive explanations of both the concept and the underlying math.
- **Real-Time Matrix Visualization**: See how matrices for Q, K, V, scores, and weights change based on your inputs.

---

## 🧠 How It Works: The 6 Steps of Attention

The application breaks down the multi-head attention mechanism into six understandable steps:

1.  **Input Matrix (X)**: We start with an input matrix representing our sequence of tokens (e.g., words).
2.  **Q, K, V Projections**: The input is projected into three distinct matrices—Query (Q), Key (K), and Value (V)—using learned weight matrices.
3.  **Split into Heads**: Q, K, and V are split into multiple "heads," allowing the model to focus on different parts of the sequence simultaneously.
4.  **Causal Masking**: A look-ahead mask is applied to prevent positions from attending to subsequent positions. This is crucial for autoregressive tasks.
5.  **Scaled Dot-Product Attention**: Attention scores are calculated by taking the dot product of Q and K, scaling them, and applying a softmax to get attention weights. These weights are then applied to V.
6.  **Concatenate & Final Projection**: The outputs from all heads are combined and passed through a final linear layer to produce the output.

---

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### Prerequisites

- Python 3.9 or higher
- `pip` for package management

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: If you don't have a `requirements.txt` file, you can create one with `pip freeze > requirements.txt` after installing the packages below.)*
    ```bash
    pip install streamlit numpy plotly
    ```

### Running the App

1.  **Execute the Streamlit command:**
    ```bash
    streamlit run st_app.py
    ```

2.  **Open your browser:** The application will automatically open in a new browser tab. If not, navigate to the local URL displayed in your terminal (usually `http://localhost:8501`).

---

## 🛠️ Technologies Used

- **Framework**: [Streamlit](https://streamlit.io/)
- **Numerical Operations**: [NumPy](https://numpy.org/)
- **Data Visualization**: [Plotly](https://plotly.com/python/)
- **Language**: Python

---

## 🙌 Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details. 
