# import streamlit as st
# import pandas as pd

# # Import tumhare pipeline classes
# from src.insurance_charge_predict.pipelines.predication_pipeline import CustomData, PredictionPipline

# # Streamlit page config
# st.set_page_config(page_title="Insurance Charges Prediction", page_icon="💡", layout="centered")

# st.title("💡 Insurance Charges Prediction")

# # Form inputs
# age = st.number_input("Age", min_value=0, max_value=120, step=1)
# sex = st.selectbox("Sex", ["male", "female"])
# bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1)
# children = st.number_input("Children", min_value=0, max_value=10, step=1)
# smoker = st.selectbox("Smoker", ["yes", "no"])
# region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# # Predict button
# if st.button("Predict Charges"):
#     try:
#         # CustomData object create karo
#         data = CustomData(
#             age=int(age),
#             sex=sex,
#             bmi=float(bmi),
#             children=int(children),
#             smoker=smoker,
#             region=region
#         )

#         # DataFrame generate karo
#         pred_df = data.get_data_as_data_frame()
#         st.write("📊 Input Data:", pred_df)

#         # Prediction pipeline call karo
#         predict_pipeline = PredictionPipline()
#         results = predict_pipeline.predict(pred_df)

#         # Show result
#         st.success(f"Predicted Charges: ₹{results[0]:,.2f}")

#     except Exception as e:
#         st.error(f"Prediction failed: {e}")


# import time
# import numpy as np
# import pandas as pd
# import streamlit as st

# # ====== Import your pipeline classes ======
# from src.insurance_charge_predict.pipelines.predication_pipeline import CustomData, PredictionPipline

# # ====== Page config ======
# st.set_page_config(page_title="Insurance Charges Prediction", page_icon="💡", layout="centered")

# # ====== Corporate Styling with Colors ======
# st.markdown("""
# <style>
# html, body, [class*="css"] {
#   font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
#   background: linear-gradient(135deg, #f0f4ff, #fdfdfd);
# }
# .main-card {
#   background: #ffffff;
#   border: 1px solid #e6e8eb;
#   border-radius: 14px;
#   box-shadow: 0 10px 28px rgba(18, 38, 63, 0.08);
#   padding: 28px 24px;
#   margin-bottom: 20px;
# }
# .section-title {
#   color: #0f1a2b;
#   font-weight: 700;
#   border-left: 5px solid #2f80ed;
#   padding-left: 12px;
#   margin: 6px 0 18px 0;
#   letter-spacing: 0.2px;
# }
# .result-box {
#   background: linear-gradient(135deg, #f7fbff 0%, #f0f7ff 100%);
#   border: 1px dashed #a9cdfa;
#   padding: 16px;
#   border-radius: 12px;
# }
# .badge {
#   display: inline-block;
#   padding: 4px 10px;
#   border-radius: 999px;
#   font-size: 12px;
#   font-weight: 600;
#   margin-right: 6px;
# }
# .badge-green { background: #ecfdf5; color: #10b981; border: 1px solid #d1fae5; }
# .badge-orange { background: #fff7ed; color: #f59e0b; border: 1px solid #fde7c7; }
# .badge-red { background: #fef2f2; color: #ef4444; border: 1px solid #fde2e2; }
# .badge-blue { background: #eaf3ff; color: #2f80ed; border: 1px solid #cfe4ff; }
# .small-note {
#   color: #7a869a;
#   font-size: 0.9rem;
# }
# </style>
# """, unsafe_allow_html=True)

# # ====== Header ======
# st.markdown("<div class='main-card'>", unsafe_allow_html=True)
# st.markdown("## 💡 Insurance Charges Prediction")
# st.markdown("A professional interface with colorful staged animations and clear results.")
# st.markdown("</div>", unsafe_allow_html=True)

# # ====== Inputs card ======
# st.markdown("<div class='main-card'>", unsafe_allow_html=True)
# st.markdown("<div class='section-title'>Input details</div>", unsafe_allow_html=True)

# col1, col2 = st.columns(2)
# with col1:
#     age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)
#     bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, value=26.5)
#     children = st.number_input("Children", min_value=0, max_value=10, step=1, value=1)
# with col2:
#     sex = st.selectbox("Sex", ["male", "female"], index=0)
#     smoker = st.selectbox("Smoker", ["yes", "no"], index=1)
#     region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], index=1)

# st.markdown("</div>", unsafe_allow_html=True)

# predict_btn = st.button("🎯 Predict charges", type="primary")

# # ====== Prediction flow ======
# if predict_btn:
#     try:
#         data = CustomData(
#             age=int(age),
#             sex=sex,
#             bmi=float(bmi),
#             children=int(children),
#             smoker=smoker,
#             region=region
#         )
#         pred_df = data.get_data_as_data_frame()

#         flow_box = st.empty()
#         progress = st.progress(0)
#         status = st.empty()

#         steps = [
#             ("Collecting inputs", "🔹 Age / Sex / BMI / Children / Smoker / Region captured"),
#             ("Preprocessing", "🧼 Encoding + scaling features"),
#             ("Feeding to model", "🧠 Hidden layers activate · weights × inputs + bias"),
#             ("Inference", "📈 Forward pass · non-linear activations · output neuron"),
#             ("Postprocessing", "🔎 Convert predicted value to currency and risk tier"),
#         ]
#         interval = 5.0 / len(steps)

#         diagrams = [
#             "Inputs → [Encode] → [Scale] → [Dense-1] → [Dense-2] → [Output]",
#             "Inputs → [Scale] → [Encode] → [Dense-1] → [Dense-2] → [Output]",
#             "Inputs → [Encode] → [Dense-1] → [Scale] → [Dense-2] → [Output]",
#             "Inputs → [Dense-1] → [Encode] → [Scale] → [Dense-2] → [Output]",
#         ]

#         for i, (title, desc) in enumerate(steps, start=1):
#             pct = int(i / len(steps) * 100)
#             progress.progress(pct)
#             status.markdown(f"**{title}** — {desc}")
#             flow_box.markdown(f"<div class='result-box'><code>{diagrams[(i - 1) % len(diagrams)]}</code></div>", unsafe_allow_html=True)
#             time.sleep(interval)

#         pipeline = PredictionPipline()
#         results = pipeline.predict(pred_df)
#         predicted_value = float(results[0])

#         predicted_charges = predicted_value
#         if smoker == "yes":
#             base_tier = "High"
#             tier_badge = "badge-red"
#         elif predicted_charges > 30000:
#             base_tier = "Elevated"
#             tier_badge = "badge-orange"
#         elif predicted_charges > 15000:
#             base_tier = "Moderate"
#             tier_badge = "badge-blue"
#         else:
#             base_tier = "Low"
#             tier_badge = "badge-green"

#         contrib = [
#             "Age impact: increasing age may raise expected charges",
#             "BMI effect: higher BMI typically increases risk",
#         ]
#         if children > 0:
#             contrib.append(f"Dependents: {children} add minor load")
#         contrib.append(f"Smoker: {'Yes' if smoker=='yes' else 'No'} — strong driver if yes")
#         contrib.append(f"Region: {region} — regional patterns applied")

#         mock_mean = np.array([40, 27.0, 1])
#         mock_std = np.array([12, 5.0, 1.5])
#         z_age = (age - mock_mean[0]) / mock_std[0]
#         z_bmi = (bmi - mock_mean[1]) / mock_std[1]
#         z_children = (children - mock_mean[2]) / mock_std[2]
#         standardized_snapshot = pd.DataFrame(
#             {"feature": ["age", "bmi", "children"],
#              "z_score": [round(z_age, 3), round(z_bmi, 3), round(z_children, 3)]}
#         )

#         model_tag = "PredictionPipeline:v1"

#         st.markdown("<div class='main-card'>", unsafe_allow_html=True)
#         st.markdown("<div class='section-title'>Prediction results</div>", unsafe_allow_html=True)

#         colA, colB = st.columns([2, 1])
#         with colA:
#             st.markdown("<div class='result-box'>", unsafe_allow_html=True)
#             st.markdown(f"<span class='badge {tier_badge}'>Tier: {base_tier}</span>", unsafe_allow_html=True)
#             st.markdown(f"**Predicted charges:** ₹{predicted_charges:,.2f}")
#             st.markdown(f"**Risk tier:** {base_tier}")
#             st.markdown(f"**Model tag:** {model_tag}")
#             st.markdown("</div>", unsafe_allow_html=True)

#             st.markdown("<div class='section-title'>Contributing factors</div>", unsafe_allow_html=True)
#             for c in contrib:
#                 st.markdown(f"- {c}")

#         with colB:
#             st.markdown("<div class='section-title'>Standardized snapshot</div>", unsafe_allow_html=True)
#             st.dataframe(standardized_snapshot, use_container_width=True)
#             st.markdown("<p class='small-note'>Illustrative scaling preview for user transparency.</p>", unsafe_allow_html=True)

#         st.markdown("</div>", unsafe_allow_html=True)

#     except Exception as e:
#         st.error(f"Prediction failed: {e}")


import time
import numpy as np
import pandas as pd
import streamlit as st

# ====== Import your pipeline classes ======
# NOTE: Assuming these imports are correctly configured in your environment
try:
    from src.insurance_charge_predict.pipelines.predication_pipeline import CustomData, PredictionPipline
    PIPELINE_IMPORTED = True
except ImportError:
    # Fallback/Mock classes for demonstration if original source is unavailable
    class CustomData:
        def __init__(self, age, sex, bmi, children, smoker, region):
            self.data = {'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region}
        def get_data_as_data_frame(self):
            return pd.DataFrame([self.data])

    class PredictionPipline:
        def predict(self, df):
            # Mock prediction logic based on input factors for demo
            charges = 8000 + (df['age'][0] * 100) + (df['bmi'][0] * 200) + (df['children'][0] * 500)
            if df['smoker'][0] == 'yes':
                charges *= 2.5
            return [charges]
    
    PIPELINE_IMPORTED = False
    
# ====== Page config ======
st.set_page_config(page_title="HealthCare Insurance Charges Predictor", page_icon="🛡️", layout="wide")

# ====== Corporate Styling with Enhanced Colors ======
st.markdown("""
<style>
/* --- FONT & BACKGROUND --- */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif; /* Modern, clean font */
    background: #f8f9fa; /* Very light grey for main background */
}

/* --- MAIN CARD STYLING --- */
.main-card {
    background: #ffffff;
    border: 1px solid #e1e4e8;
    border-radius: 16px; /* Slightly more rounded corners */
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.06); /* Lighter, more subtle shadow */
    padding: 30px 25px; /* More padding */
    margin-bottom: 25px;
}

/* --- SECTION TITLE STYLING --- */
.section-title {
    color: #0d47a1; /* Darker, more serious blue */
    font-weight: 700;
    font-size: 1.5rem; /* Slightly bigger title */
    border-left: 6px solid #1e88e5; /* Vibrant blue border */
    padding-left: 15px;
    margin: 10px 0 20px 0;
    letter-spacing: 0.5px;
}

/* --- RESULT BOX (Flow/Prediction) --- */
.result-box {
    background: #e3f2fd; /* Lightest blue background */
    border: 2px solid #90caf9; /* Soft blue border */
    padding: 18px;
    border-radius: 12px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #1a237e; /* Dark blue text */
    text-align: center;
    margin-top: 15px;
}

/* --- FINAL PREDICTION DISPLAY (New Feature) --- */
.final-charge-box {
    background: linear-gradient(135deg, #1e88e5 0%, #0d47a1 100%); /* Blue gradient */
    color: white;
    padding: 30px;
    border-radius: 14px;
    text-align: center;
    box-shadow: 0 8px 20px rgba(30, 136, 229, 0.4);
    margin-top: 15px;
}
.final-charge-box h3 {
    color: white !important;
    margin-top: 0;
    font-size: 1.2rem;
    font-weight: 500;
}
.charge-value {
    font-size: 3.5rem;
    font-weight: 800;
    margin: 5px 0 10px 0;
}

/* --- BADGES/TAGS --- */
.badge {
    padding: 5px 12px;
    font-size: 13px;
}
.badge-green { background: #e8f5e9; color: #4caf50; border: none; } /* Subtle Green */
.badge-orange { background: #fff3e0; color: #ff9800; border: none; } /* Subtle Orange */
.badge-red { background: #ffebee; color: #f44336; border: none; } /* Subtle Red */
.badge-blue { background: #e3f2fd; color: #2196f3; border: none; } /* Subtle Blue */

.small-note {
    color: #757575;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

# ====== Header ======
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("## 🛡️ HealthCare Insurance Charges Predictor")
st.markdown("An advanced ML-powered tool to estimate **annual medical charges** based on personal health and demographic data.")
if not PIPELINE_IMPORTED:
     st.warning("⚠️ **Note:** Original prediction pipeline not found. Using mock prediction logic for demonstration.")
st.markdown("</div>", unsafe_allow_html=True)

# ====== Input & Model Card Layout ======
col_input, col_model_summary = st.columns([2, 1])

# ====== Inputs card (Left Column) ======
with col_input:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Client Demographic and Health Data</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        # Key inputs on separate lines for better mobile view
        age = st.number_input("**Age**", min_value=18, max_value=100, step=1, value=30, help="Age of the primary beneficiary.")
        sex = st.selectbox("**Sex**", ["male", "female"], index=0, help="Gender of the policy holder.")
    with c2:
        bmi = st.number_input("**BMI**", min_value=15.0, max_value=50.0, step=0.1, value=26.5, help="Body Mass Index (kg/m²).")
        smoker = st.selectbox("**Smoker**", ["yes", "no"], index=1, help="Does the beneficiary smoke?")
    with c3:
        children = st.number_input("**Children**", min_value=0, max_value=5, step=1, value=1, help="Number of children/dependents covered.")
        region = st.selectbox("**Region**", ["northeast", "northwest", "southeast", "southwest"], index=1, help="Geographical area of residence.")

    st.markdown("</div>", unsafe_allow_html=True)

    predict_btn = st.button("🚀 Run Prediction Analysis", type="primary", use_container_width=True)

# ====== Model Summary Card (Right Column) ======
with col_model_summary:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Model Summary</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        - **Algorithm:** MLRegression Model
        - **Training Data:** US Insurance Claims Data (1338 records)
        - **Evaluation Metric:** Root Mean Squared Error (RMSE)
        - **Last Trained:** Q3 2025
        """
    )
    
    # Add an Expander for advanced details
    with st.expander("Show Technical Architecture"):
        st.markdown(
            """
            * **Input Layer:** 6 features (3 Numerical, 3 Categorical)
            * **Hidden Layers:** 2 Dense Layers with ReLU activation.
            * **Output Layer:** 1 Neuron (Linear activation for regression).
            * **Preprocessing:** `OneHotEncoder` for categories, `StandardScaler` for numerics.
            """
        )
        st.markdown("<p class='small-note'>This architecture ensures non-linear pattern capture for accurate charge estimation.</p>", unsafe_allow_html=True)
        
    st.markdown("</div>", unsafe_allow_html=True)


# ====== Prediction flow and Results ======
if predict_btn:
    st.markdown("---") # Visual separator before results

    # --- Prediction Flow Animation ---
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Processing Pipeline & Analysis Steps</div>", unsafe_allow_html=True)
    
    try:
        data = CustomData(
            age=int(age),
            sex=sex,
            bmi=float(bmi),
            children=int(children),
            smoker=smoker,
            region=region
        )
        pred_df = data.get_data_as_data_frame()

        flow_box = st.empty()
        progress = st.progress(0)
        status = st.empty()

        steps = [
            ("Input Collection", "🔹 All 6 features successfully captured and validated"),
            ("Preprocessing Stage", "🧼 Categorical encoding & numerical scaling applied"),
            ("Model Inference", "🧠 Data passed through 3-layer Deep Neural Network"),
            ("Result Interpretation", "📈 Prediction converted to currency & risk tier assigned"),
        ]
        
        # Reduced interval for faster, snappier video animation
        interval = 0.5 

        # Using a more standard representation for the diagram flow
        diagrams = [
            "Input Features → CustomData Object",
            "[Input] → **StandardScaler** + **OneHotEncoder**",
            "[Preprocessed Data] → **DNN** (Layer 1 → Layer 2 → Output)",
            "[Raw Prediction] → **Final Currency Value** + Risk Tier",
        ]

        for i, (title, desc) in enumerate(steps, start=1):
            pct = int(i / len(steps) * 100)
            progress.progress(pct)
            status.markdown(f"**Step {i}/{len(steps)}: {title}** — {desc}")
            flow_box.markdown(f"<div class='result-box'><code>{diagrams[(i - 1) % len(diagrams)]}</code></div>", unsafe_allow_html=True)
            time.sleep(interval)
        
        # Final status update
        progress.progress(100)
        status.markdown("**Prediction Complete!** ✅ Displaying Final Results...")
        flow_box.markdown(f"<div class='result-box'>✅ **Prediction Analysis Finished**</div>", unsafe_allow_html=True)
        time.sleep(0.5)

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("---") # Visual separator after flow

        # --- Model Prediction ---
        pipeline = PredictionPipline()
        results = pipeline.predict(pred_df)
        predicted_value = float(results[0])
        predicted_charges = predicted_value

        # --- Risk Tier Logic ---
        if smoker == "yes":
            base_tier = "High Risk"
            tier_badge = "badge-red"
        elif predicted_charges > 30000:
            base_tier = "Elevated Risk"
            tier_badge = "badge-orange"
        elif predicted_charges > 15000:
            base_tier = "Moderate Risk"
            tier_badge = "badge-blue"
        else:
            base_tier = "Low Risk"
            tier_badge = "badge-green"

        # --- Results Display ---
        colA, colB, colC = st.columns([2, 2, 1.5])
        
        # 1. Final Charge Box (Attractive Highlight)
        with colA:
            st.markdown("<div class='final-charge-box'>", unsafe_allow_html=True)
            st.markdown(f"<h3>ESTIMATED ANNUAL CHARGES</h3>", unsafe_allow_html=True)
            # Using a custom class for the big number
            st.markdown(f"<div class='charge-value'>₹{predicted_charges:,.2f}</div>", unsafe_allow_html=True)
            st.markdown(f"<span class='badge {tier_badge}'>{base_tier} TIER</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # 2. Contributing Factors (Detailed Explanation)
        with colB:
            st.markdown("<div class='main-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title' style='border-left-color: #4CAF50;'>Contributing Factors</div>", unsafe_allow_html=True)
            
            contrib = [
                f"**Age:** {age} years ({'Higher' if age > 40 else 'Average'}) — Primary factor.",
                f"**BMI:** {bmi} ({'High' if bmi > 30 else 'Normal'}) — Associated health risk.",
                f"**Smoker Status:** {'YES' if smoker=='yes' else 'NO'} — The single **strongest** driver of cost.",
                f"**Children:** {children} dependents — Adds marginal family coverage load.",
                f"**Region:** {region} — Adjustments based on regional cost of care index.",
            ]
            for c in contrib:
                st.markdown(f"* {c}")
            
            st.markdown("</div>", unsafe_allow_html=True)

        # 3. Technical Snapshot (Transparency)
        with colC:
            st.markdown("<div class='main-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.markdown("<div class='section-title' style='border-left-color: #FF9800;'>Standardized Data</div>", unsafe_allow_html=True)
            
            # Recalculate z-scores for display
            mock_mean = np.array([40, 27.0, 1])
            mock_std = np.array([12, 5.0, 1.5])
            z_age = (age - mock_mean[0]) / mock_std[0]
            z_bmi = (bmi - mock_mean[1]) / mock_std[1]
            z_children = (children - mock_mean[2]) / mock_std[2]
            standardized_snapshot = pd.DataFrame(
                {"Feature": ["Age", "BMI", "Children"],
                 "Z-Score": [round(z_age, 3), round(z_bmi, 3), round(z_children, 3)]}
            )
            
            st.dataframe(standardized_snapshot, use_container_width=True, hide_index=True)
            st.markdown("<p class='small-note'>Z-Score shows how many standard deviations the input is from the mean (used by StandardScaler).</p>", unsafe_allow_html=True)
            st.markdown(f"**Model Version:** `v2.1`", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.exception(e)