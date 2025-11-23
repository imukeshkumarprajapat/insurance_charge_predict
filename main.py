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


import time
import numpy as np
import pandas as pd
import streamlit as st

# ====== Import your pipeline classes ======
from src.insurance_charge_predict.pipelines.predication_pipeline import CustomData, PredictionPipline

# ====== Page config ======
st.set_page_config(page_title="Insurance Charges Prediction", page_icon="💡", layout="centered")

# ====== Corporate Styling with Colors ======
st.markdown("""
<style>
html, body, [class*="css"] {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #f0f4ff, #fdfdfd);
}
.main-card {
  background: #ffffff;
  border: 1px solid #e6e8eb;
  border-radius: 14px;
  box-shadow: 0 10px 28px rgba(18, 38, 63, 0.08);
  padding: 28px 24px;
  margin-bottom: 20px;
}
.section-title {
  color: #0f1a2b;
  font-weight: 700;
  border-left: 5px solid #2f80ed;
  padding-left: 12px;
  margin: 6px 0 18px 0;
  letter-spacing: 0.2px;
}
.result-box {
  background: linear-gradient(135deg, #f7fbff 0%, #f0f7ff 100%);
  border: 1px dashed #a9cdfa;
  padding: 16px;
  border-radius: 12px;
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
  margin-right: 6px;
}
.badge-green { background: #ecfdf5; color: #10b981; border: 1px solid #d1fae5; }
.badge-orange { background: #fff7ed; color: #f59e0b; border: 1px solid #fde7c7; }
.badge-red { background: #fef2f2; color: #ef4444; border: 1px solid #fde2e2; }
.badge-blue { background: #eaf3ff; color: #2f80ed; border: 1px solid #cfe4ff; }
.small-note {
  color: #7a869a;
  font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ====== Header ======
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("## 💡 Insurance Charges Prediction")
st.markdown("A professional interface with colorful staged animations and clear results.")
st.markdown("</div>", unsafe_allow_html=True)

# ====== Inputs card ======
st.markdown("<div class='main-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Input details</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, step=1, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, value=26.5)
    children = st.number_input("Children", min_value=0, max_value=10, step=1, value=1)
with col2:
    sex = st.selectbox("Sex", ["male", "female"], index=0)
    smoker = st.selectbox("Smoker", ["yes", "no"], index=1)
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"], index=1)

st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("🎯 Predict charges", type="primary")

# ====== Prediction flow ======
if predict_btn:
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
            ("Collecting inputs", "🔹 Age / Sex / BMI / Children / Smoker / Region captured"),
            ("Preprocessing", "🧼 Encoding + scaling features"),
            ("Feeding to model", "🧠 Hidden layers activate · weights × inputs + bias"),
            ("Inference", "📈 Forward pass · non-linear activations · output neuron"),
            ("Postprocessing", "🔎 Convert predicted value to currency and risk tier"),
        ]
        interval = 5.0 / len(steps)

        diagrams = [
            "Inputs → [Encode] → [Scale] → [Dense-1] → [Dense-2] → [Output]",
            "Inputs → [Scale] → [Encode] → [Dense-1] → [Dense-2] → [Output]",
            "Inputs → [Encode] → [Dense-1] → [Scale] → [Dense-2] → [Output]",
            "Inputs → [Dense-1] → [Encode] → [Scale] → [Dense-2] → [Output]",
        ]

        for i, (title, desc) in enumerate(steps, start=1):
            pct = int(i / len(steps) * 100)
            progress.progress(pct)
            status.markdown(f"**{title}** — {desc}")
            flow_box.markdown(f"<div class='result-box'><code>{diagrams[(i - 1) % len(diagrams)]}</code></div>", unsafe_allow_html=True)
            time.sleep(interval)

        pipeline = PredictionPipline()
        results = pipeline.predict(pred_df)
        predicted_value = float(results[0])

        predicted_charges = predicted_value
        if smoker == "yes":
            base_tier = "High"
            tier_badge = "badge-red"
        elif predicted_charges > 30000:
            base_tier = "Elevated"
            tier_badge = "badge-orange"
        elif predicted_charges > 15000:
            base_tier = "Moderate"
            tier_badge = "badge-blue"
        else:
            base_tier = "Low"
            tier_badge = "badge-green"

        contrib = [
            "Age impact: increasing age may raise expected charges",
            "BMI effect: higher BMI typically increases risk",
        ]
        if children > 0:
            contrib.append(f"Dependents: {children} add minor load")
        contrib.append(f"Smoker: {'Yes' if smoker=='yes' else 'No'} — strong driver if yes")
        contrib.append(f"Region: {region} — regional patterns applied")

        mock_mean = np.array([40, 27.0, 1])
        mock_std = np.array([12, 5.0, 1.5])
        z_age = (age - mock_mean[0]) / mock_std[0]
        z_bmi = (bmi - mock_mean[1]) / mock_std[1]
        z_children = (children - mock_mean[2]) / mock_std[2]
        standardized_snapshot = pd.DataFrame(
            {"feature": ["age", "bmi", "children"],
             "z_score": [round(z_age, 3), round(z_bmi, 3), round(z_children, 3)]}
        )

        model_tag = "PredictionPipeline:v1"

        st.markdown("<div class='main-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Prediction results</div>", unsafe_allow_html=True)

        colA, colB = st.columns([2, 1])
        with colA:
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.markdown(f"<span class='badge {tier_badge}'>Tier: {base_tier}</span>", unsafe_allow_html=True)
            st.markdown(f"**Predicted charges:** ₹{predicted_charges:,.2f}")
            st.markdown(f"**Risk tier:** {base_tier}")
            st.markdown(f"**Model tag:** {model_tag}")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='section-title'>Contributing factors</div>", unsafe_allow_html=True)
            for c in contrib:
                st.markdown(f"- {c}")

        with colB:
            st.markdown("<div class='section-title'>Standardized snapshot</div>", unsafe_allow_html=True)
            st.dataframe(standardized_snapshot, use_container_width=True)
            st.markdown("<p class='small-note'>Illustrative scaling preview for user transparency.</p>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")