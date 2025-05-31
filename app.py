import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Chargement du modèle et du scaler
model = joblib.load("LogisticR.pkl")
scaler = joblib.load("scaler.pkl")

feature_names = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                 'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography', 'Gender']

st.set_page_config(page_title="Prédiction de Churn Client", page_icon="🔍", layout="centered")
st.title("🔮 Prédiction de départ client (Churn)")
st.markdown("Cette application prédit si un client est susceptible de quitter la banque.")

st.sidebar.header("🧾 Informations du client")

def user_input():
    credit_score = st.sidebar.slider('Credit Score', 300, 900, 650)
    age = st.sidebar.slider('Âge', 18, 100, 40)
    tenure = st.sidebar.slider('Ancienneté (en années)', 0, 10, 5)
    balance = st.sidebar.number_input('Solde du compte (€)', value=10000.0)
    num_of_products = st.sidebar.selectbox('Nombre de produits bancaires', [1, 2, 3, 4])
    has_cr_card = st.sidebar.selectbox('Carte de crédit ?', ['Oui', 'Non'])
    is_active_member = st.sidebar.selectbox('Client actif ?', ['Oui', 'Non'])
    estimated_salary = st.sidebar.number_input('Salaire estimé (€)', value=50000.0)
    geography = st.sidebar.selectbox('Pays', ['France', 'Germany', 'Spain'])
    gender = st.sidebar.selectbox('Genre', ['Male', 'Female'])

    # Traitement des données
    has_cr_card = 1 if has_cr_card == 'Oui' else 0
    is_active_member = 1 if is_active_member == 'Oui' else 0
    gender = 1 if gender == 'Male' else 0

    geo_encoding = {'France': 0.5, 'Germany': 0.3, 'Spain': 0.2}
    geography = geo_encoding.get(geography, 0.5)

    data = np.array([[credit_score, age, tenure, balance, num_of_products,
                      has_cr_card, is_active_member, estimated_salary, geography, gender]])
    return data, credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary, geography, gender

input_data, credit_score, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary, geography, gender = user_input()

if st.button("🔍 Prédire le churn"):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    pred_proba = model.predict_proba(input_scaled)[0][1]

    st.subheader("🧠 Résultat de la prédiction")
    if prediction == 1:
        st.error(f"⚠️ Le client est à risque de quitter. (Probabilité: {pred_proba:.2f})")
    else:
        st.success(f"✅ Le client est fidèle. (Probabilité de churn: {pred_proba:.2f})")

    st.markdown("---")
    st.markdown("### 📝 Détails du client")
    st.write(f"- Score de crédit : {credit_score}")
    st.write(f"- Âge : {age}")
    st.write(f"- Solde : {balance} €")
    st.write(f"- Ancienneté : {tenure} ans")
    st.write(f"- Nombre de produits : {num_of_products}")
    st.write(f"- Carte de crédit : {'Oui' if has_cr_card else 'Non'}")
    st.write(f"- Client actif : {'Oui' if is_active_member else 'Non'}")
    st.write(f"- Salaire estimé : {estimated_salary} €")
    st.write(f"- Pays : {geography}")
    st.write(f"- Genre : {'Homme' if gender else 'Femme'}")

    st.markdown("---")
    st.markdown("### 📈 Probabilité de churn")
    st.progress(int(pred_proba * 100))
    st.metric("Probabilité (%)", f"{pred_proba*100:.2f} %")

    st.markdown("---")
    if st.button("💾 Sauvegarder cette prédiction"):
        data_to_save = pd.DataFrame(input_data, columns=feature_names)
        data_to_save["prediction"] = prediction
        data_to_save["probability"] = pred_proba
        data_to_save.to_csv("historique_predictions.csv", mode='a', header=False, index=False)
        st.success("✅ Prédiction sauvegardée dans historique_predictions.csv")

    if st.checkbox("📂 Voir l’historique des prédictions"):
        try:
            df_history = pd.read_csv("historique_predictions.csv", header=None)
            df_history.columns = feature_names + ["prediction", "probability"]
            st.dataframe(df_history.tail(10))
        except FileNotFoundError:
            st.warning("Aucune prédiction enregistrée pour le moment.")

    st.markdown("---")
    st.markdown("**Modèle utilisé : Logistic Regression**")
    st.markdown("- Accuracy : **82%**")
    st.markdown("- AUC : **0.77**")
    st.markdown("- Source : Application interne Banque Zénith 🏦")

    st.markdown("---")
    
