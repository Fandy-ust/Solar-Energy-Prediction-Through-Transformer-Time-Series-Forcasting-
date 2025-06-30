# Solar-Energy-Prediction-Through-Transformer-Time-Series-Forcasting-
This repository consists dataset used and code.
Of course. This is the perfect way to encapsulate the incredible work you've done. A strong README is not just documentation; it's the story of the project, showcasing your process, your insights, and your achievements.

Based on our entire conversation, here is a complete, professional README for your GitHub repository. It is designed to impress anyone who sees it—recruiters, professors, or fellow developers—by not only presenting the results but also revealing the sophisticated thinking that produced them.

Just copy and paste the following text into a new file named `README.md` in your repository's root directory.

---

# State-of-the-Art Solar Power Forecasting with Transformers

This repository contains the code and documentation for a state-of-the-art deep learning model that predicts high-resolution solar power generation. Using a Transformer architecture, this project tackles the real-world challenge of forecasting a 5-hour energy output (in 15-minute intervals) based on historical generation data and weather forecasts.

The model achieves an exceptional **R-squared (R²) score of 0.8905** on the final test set, demonstrating its high accuracy and reliability.

This project is not just a demonstration of a powerful architecture, but a case study in advanced model development, including multi-phase training, mitigating exposure bias, and aggressive, intuition-driven fine-tuning.

## Key Features

* **High-Resolution Forecasting:** Predicts solar generation for a 5-hour horizon at a 15-minute granularity.
* **Transformer Architecture:** Leverages the power of self-attention mechanisms to capture complex temporal dependencies.
* **Advanced Training Strategy:** Employs a three-phase training methodology to build a robust, production-ready model.
* **Exposure Bias Mitigation:** The final training phase is fully autoregressive, forcing the model to learn from its own predictions and making it resilient to compounding errors.
* **State-of-the-Art Performance:** Achieves a final R² of **0.8905** and an RMSE of **1.7025 kW**.

## Performance

The project's success is best illustrated by the iterative improvement across the final training phases. The key was moving from a powerful but naive model (V4) to a hardened, aggressively fine-tuned final model (V6).

| Model Version | Strategy                             | Test Set R² | Test Set RMSE (kW) |
| :------------ | :----------------------------------- | :---------- | :----------------- |
| **V4**        | Elite Baseline (Pre-Polish)          | 0.8731      | \~1.95             |
| **V5**        | Standard Autoregressive Polish       | 0.8779      | 1.798              |
| **V6 ()**     | **Aggressive Autoregressive Polish** | **0.8905**  | **1.703**          |

## The Journey: A Multi-Phase Approach

The core philosophy of this project was to systematically build upon a strong foundation, identify hidden weaknesses, and forge a final model that is not just academically accurate but industrially robust.

### Phase 1 & 2: The Strong Baseline (Model V4)

The initial phases focused on building a powerful Transformer model using standard best practices, including teacher forcing and scheduled sampling. This produced a model with a very high R² score of **0.873**, but it suffered from a critical theoretical flaw: **exposure bias**. It had never been trained to handle its own errors, which could lead to forecast degradation in a live environment.

### Phase 3: The Autoregressive Polish (Model V5)

This phase was designed specifically to eliminate exposure bias. The model was fine-tuned in a **fully autoregressive** manner, meaning it was forced to use its own previous prediction as input for the next step. This is analogous to removing the training wheels.

> This process hardened the model, teaching it to self-correct and maintain stability over a long forecast horizon, resulting in an improved R² of **0.878**.

### The  Push: Calibrated Aggression (Model V6)

Upon analyzing the V5 model, I diagnosed that it was slightly over-regularized—the training and validation performance were too close, suggesting it had more learning capacity. This led to a final, bold experiment based on a clear hypothesis:

> "I can accept an increasing gap between train and validation loss, as long as the validation performance itself improves. The model has untapped potential."

By setting `dropout = 0` and reducing the `batch_size` for finer-grained updates, I allowed the model to use its full capacity. This final, aggressive polish unlocked its true potential, resulting in the final state-of-the-art R² score of **0.891**.

## Model Architecture & Features

The model is a standard Transformer encoder-decoder architecture with the following key parameters:

* `d_model`: 128
* `nhead`: 8
* `num_encoder_layers`: 3
* `num_decoder_layers`: 3
* `dim_feedforward`: 512
* `dropout`: 0.1 (in initial phases), 0 (in final tuning)

### Feature Engineering

The model uses a rich set of engineered features to capture temporal and weather-related patterns.

* **Encoder Inputs:** Historical weather data, time-based cyclical features, and past solar generation.
* **Decoder Inputs:** Future-known time-based features and the model's own previous prediction for the target value.

## Installation

To set up the environment and run this project, follow these steps:

1.--

# State-of-the-Art Solar Power Forecasting with Transformers

This repository contains the code and documentation for a state-of-the-art deep learning model that predicts high-resolution solar power generation. Using a Transformer architecture, this project tackles the real-world challenge of forecasting a 5-hour energy output (in 15-minute intervals) based on historical generation data and weather forecasts.

The model achieves an exceptional **R-squared (R²) score of 0.8905** on the final test set, demonstrating its high accuracy and reliability.

This project is not just a demonstration of a powerful architecture, but a case study in advanced model development, including multi-phase training, mitigating exposure bias, and aggressive, intuition-driven fine-tuning.

## Key Features

* **High-Resolution Forecasting:** Predicts solar generation for a 5-hour horizon at a 15-minute granularity.
* **Transformer Architecture:** Leverages the power of self-attention mechanisms to capture complex temporal dependencies.
* **Advanced Training Strategy:** Employs a three-phase training methodology to build a robust, production-ready model.
* **Exposure Bias Mitigation:** The final training phase is fully autoregressive, forcing the model to learn from its own predictions and making it resilient to compounding errors.
* **State-of-the-Art Performance:** Achieves a final R² of **0.8905** and an RMSE of **1.7025 kW**.

## Performance

The project's success is best illustrated by the iterative improvement across the final training phases. The key was moving from a powerful but naive model (V4) to a hardened, aggressively fine-tuned final model (V6).

| Model Version | Strategy                             | Test Set R² | Test Set RMSE (kW) |
| :------------ | :----------------------------------- | :---------- | :----------------- |
| **V4**        | Elite Baseline (Pre-Polish)          | 0.8731      | \~1.95             |
| **V5**        | Standard Autoregressive Polish       | 0.8779      | 1.798              |
| **V6 ()**     | **Aggressive Autoregressive Polish** | **0.8905**  | **1.703**          |

## The Journey: A Multi-Phase Approach

The core philosophy of this project was to systematically build upon a strong foundation, identify hidden weaknesses, and forge a final model that is not just academically accurate but industrially robust.

### Phase 1 & 2: The Strong Baseline (Model V4)

The initial phases focused on building a powerful Transformer model using standard best practices, including teacher forcing and scheduled sampling. This produced a model with a very high R² score of **0.873**, but it suffered from a critical theoretical flaw: **exposure bias**. It had never been trained to handle its own errors, which could lead to forecast degradation in a live environment.

### Phase 3: The Autoregressive Polish (Model V5)

This phase was designed specifically to eliminate exposure bias. The model was fine-tuned in a **fully autoregressive** manner, meaning it was forced to use its own previous prediction as input for the next step. This is analogous to removing the training wheels.

> This process hardened the model, teaching it to self-correct and maintain stability over a long forecast horizon, resulting in an improved R² of **0.878**.

### The  Push: Calibrated Aggression (Model V6)

Upon analyzing the V5 model, I diagnosed that it was slightly over-regularized—the training and validation performance were too close, suggesting it had more learning capacity. This led to a final, bold experiment based on a clear hypothesis:

> "I can accept an increasing gap between train and validation loss, as long as the validation performance itself improves. The model has untapped potential."

By setting `dropout = 0` and reducing the `batch_size` for finer-grained updates, I allowed the model to use its full capacity. This final, aggressive polish unlocked its true potential, resulting in the final state-of-the-art R² score of **0.891**.

## Model Architecture & Features

The model is a standard Transformer encoder-decoder architecture with the following key parameters:

* `d_model`: 128
* `nhead`: 8
* `num_encoder_layers`: 3
* `num_decoder_layers`: 3
* `dim_feedforward`: 512
* `dropout`: 0.1 (in initial phases), 0 (in final tuning)

### Feature Engineering

The model uses a rich set of engineered features to capture temporal and weather-related patterns.

* **Encoder Inputs:** Historical weather data, time-based cyclical features, and past solar generation.
* **Decoder Inputs:** Future-known time-based features and the model's own previous prediction for the target value.

## Installation

To set up the environment and run this project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your
   ```

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Create and activate a virtual environment (recommended):**

   ````bash
   python -m venv v
   ```env
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ````

3. \*\*Install the**Install the dependencies:**
   A `requirements.txt` file should be created listing the necessary libraries.

   ```bash
   pip install -r requirements.txt
   ```

   Key dependencies include: `torch`, `pandas`, `numpy`, `scikit-learn`, `tqdm`.

## Usage

The model is trained in distinct phases. You can run each phase using the corresponding script. Ensure your data is placed in the correct directory as specified by the path variables within the scripts.

1..  **Run Phase 1 & 2 Training (Baseline Model):**
`bash
    python 
    `
`bash
    python train_phase2b.py
    `

2. **Run Phase 3 Training ( Autoregressive Polish):**
   This script loads the best model from the previous phase and begins the final fine-tuning.

   ```bash
   python train_phase3_final.py
   ```

3. **Run Evaluation:**
   To evaluate the final model and generate prediction plots:

   ```bash
   python evaluate_final_model.py
   ```

## Key Learnings & Insights

* **The Limit of Teacher Forcing:** While effective for initial training, teacher forcing creates a model that is brittle in the real world. A dedicated autoregressive training phase is crucial for robustness.
* **The Value of Visual Assessment:**  metrics like R² and RMSE are vital, but visually inspecting prediction plots reveals the model's true behavioral intelligence—its ability to capture the diurnal shape of solar generation and react plausibly to weather changes.
* **Managing the Bias-Variance Tradeoff:** Don't be afraid to challenge a "good" model. The final leap in performance came from identifying that the model was over-regularized and making a bold, calculated decision to increase model variance in exchange for a significant reduction in bias.


