## SHAP Interpretation Guide

This guide explains how to read the main SHAP plots used in the project and how to communicate them to the rest of the team (and non‑technical stakeholders).

- **Audience**: intermediate data scientists / ML engineers
- **Model type**: tree‑based multiclass classifier for obesity (7 classes)
- **Goal**: understand global feature importance and patient‑level explanations

---

## 1. SHAP Summary Plot (Beeswarm)

### What it shows

- **Global view of feature importance and effect direction** across many patients.
- Each **dot** is one patient–feature pair.
- Features are **sorted by importance** (mean absolute SHAP value).

### How to read it

- **Y‑axis**: features (top = most important globally).
- **X‑axis**: SHAP value for that feature (impact on the model output).  
  - Values **> 0** push the prediction **up** (toward higher risk / target score).  
  - Values **< 0** push the prediction **down**.
- **Color**: feature value for that patient.  
  - Typically, **red = high value**, **blue = low value**.

Key reading patterns:

- A feature with many points far from 0 (left/right) is **globally influential**.
- If **red points** cluster on the **right**, high values of that feature **increase** the predicted risk.
- If **red points** cluster on the **left**, high values **decrease** the predicted risk.
- The **spread** horizontally indicates how variable the effect is across patients.

### Key insights you can extract

- **Ranked list of most important features** for the model.
- **Monotonic vs non‑monotonic effects**:  
  - Monotonic: higher value consistently pushes in one direction.  
  - Non‑monotonic: red and blue dots appear on both sides → effect depends on context.
- **Feature interaction hints**: thick “bands” or multimodal structures often indicate interactions with other features.

### Common mistakes to avoid

- **Confusing SHAP values with feature values**:  
  - X‑axis is **impact on prediction**, not the raw feature.
- **Interpreting colors backwards**: always confirm the color map (red = high, blue = low).
- **Over‑interpreting noisy tails**: a few extreme dots may come from very rare combinations; check sample counts.
- **Assuming causality**: SHAP explains the model, not the real‑world causal mechanism.

---

## 2. SHAP Bar Plot (Global Feature Importance)

### What it shows

- **Ranked global importance** of features based on **mean |SHAP value|**.
- No sign / direction, only **magnitude of contribution**.

### How to read it

- **Y‑axis**: features, sorted from most to least important.
- **X‑axis**: mean(|SHAP value|), i.e. average absolute impact on the prediction.
- **Bars**: longer bars = features that on average move predictions more.

### Key insights you can extract

- **Clean ranking** of global feature importance.
- Good for:
  - **Feature selection** discussions.
  - Prioritizing which features to inspect with dependence or local plots.
  - Slide‑friendly “top‑k feature” summaries.

### Common mistakes to avoid

- **Ignoring direction**: bar plots hide whether high values increase or decrease risk; combine with beeswarm or dependence plots.
- **Dropping mid‑importance features too early**: some medium features may be critical for fairness or specific subgroups.
- **Comparing values across different models**: SHAP magnitudes are model‑specific; don’t compare bar lengths from different runs without care.

---

## 3. Waterfall Plot (Individual Explanation)

### What it shows

- **Patient‑level explanation** of one prediction.
- Starts from the **base value** (average model output on the training data) and adds contributions from each feature to reach the final prediction.
- Visual **“cascade”** of positive and negative contributions.

### How to read it

- **Leftmost bar**: **base value** (expected model output).
- Subsequent bars:
  - **Red bars**: features that **increase** the prediction for this patient (positive SHAP).
  - **Blue bars**: features that **decrease** the prediction (negative SHAP).
- **Rightmost bar**: final model **prediction** for the patient (for a given class or log‑odds / score, depending on the setup).
- Features are usually **sorted by |SHAP value|** (largest contributors shown first).

### Key insights you can extract

- **Why this patient obtained this prediction** instead of the average.
- **Top drivers** increasing risk (red) and **protective factors** decreasing risk (blue).
- Easy narrative for clinicians:  
  “Because the patient has high Weight and low physical activity, the model pushes the prediction toward the severe obesity class, while their younger Age slightly pulls it down.”

### Common mistakes to avoid

- **Misinterpreting the scale**: know whether the model output is probability, log‑odds, or raw score.
- **Ignoring omitted features**: only the top contributors are shown; many small effects can accumulate.
- **Reading absolute bar length without context**: check the final prediction value and compare across patients.

---

## 4. Force Plot (Individual or Group Explanation)

### What it shows

- Another way to visualize **how features push the prediction** from the base value to the final value.
- Often used interactively (HTML) for a single patient.

### How to read it

- **Baseline (center)**: expected model output (base value).
- **Red segments**: features that **push the prediction up** (toward higher risk).
- **Blue segments**: features that **push the prediction down**.
- The **length** of each segment is proportional to the magnitude of its SHAP value.
- For a single patient, you see a single “bar” with contributions stacked left and right around the baseline.

### Key insights you can extract

- Same as waterfall plot, but:
  - More **compact** view.
  - Good for interactive dashboards / tooltips.
- Shows **balance** between risk‑increasing and risk‑decreasing factors.

### Common mistakes to avoid

- **Using static screenshots only**: force plots are more powerful when interactive (hover, zoom).
- **Overloading with too many features**: keep them to the top contributors to avoid clutter.
- **Ignoring direction**: always check sign and color together.

---

## 5. Dependence Plot

### What it shows

- Relationship between a **single feature’s value** and its **SHAP value** across many patients.
- Often also shows a **second feature (color)** to reveal interactions.

### How to read it

- **X‑axis**: raw feature value (e.g. `Weight`, `Age`, `FAF`).
- **Y‑axis**: SHAP value for that feature (impact on model output).
- **Color**: value of an interaction feature (optional), often chosen automatically or specified.

Reading patterns:

- An **upward trend**: higher feature values tend to **increase** the prediction.
- A **downward trend**: higher feature values **decrease** the prediction.
- **Non‑linear shapes** (S‑curves, thresholds): the model may behave very differently below/above certain values.
- **Color gradients**: suggest interactions (e.g. effect of Weight depends on physical activity).

### Key insights you can extract

- **Partial effect** of a feature with uncertainty bands implicitly encoded in the point cloud.
- **Thresholds** where risk sharply increases/decreases.
- **Interactions**: when points with similar X differ strongly in Y depending on the color feature.

### Common mistakes to avoid

- **Reading it as a causal dose–response curve**: SHAP reflects the model, not necessarily the real world.
- **Ignoring sample density**: areas with few points can be misleading; look at how many observations support a pattern.
- **Assuming monotonicity**: tree models can learn complex non‑monotonic relationships.

---

## 6. Quick‑Reference Table

| Plot type              | Scope         | Axes / encoding                                                                 | What to use it for                                               | Watch out for                                                   |
|------------------------|--------------|----------------------------------------------------------------------------------|------------------------------------------------------------------|-----------------------------------------------------------------|
| Summary (beeswarm)     | Global       | Y: features; X: SHAP value; Color: feature value                                 | Global importance + direction; spotting interactions             | Confusing impact (X) with feature values; assuming causality    |
| Bar (mean |SHAP|)      | Global       | Y: features; X: mean(|SHAP|)                                                     | Clean ranking of features; feature selection                     | No direction info; over‑pruning medium features                 |
| Waterfall              | Individual   | Bars: contributions from base value to prediction                                | Patient‑level explanation; storytelling with clinicians          | Misreading output scale; forgetting hidden small contributors   |
| Force                  | Individual   | Baseline vs red/blue pushes (segment lengths)                                    | Compact patient explanation; interactive exploration             | Too many features; static screenshots lose interactivity        |
| Dependence             | Global/local | X: feature value; Y: SHAP value; Color: optional interaction feature             | Shape of effect; thresholds; interaction patterns                | Treating it as causal; over‑interpreting sparse regions         |

Use this guide as a checklist when creating plots for reports or the Streamlit/Flask UI:  
for each plot, ask **“What is the question I want to answer?”** and pick the SHAP visualization that matches that question.

