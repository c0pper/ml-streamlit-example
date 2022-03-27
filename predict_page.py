import streamlit as st
import pickle

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
model = data["model"]
le = data["le"]
ctv = data["ctv"]


def show_predict_page():
    st.title("AMICI di lorenzo valitutto COME PRIMA")

    st.write("""### Scrivi un testo e indovinerò quale amico come prima sei""")
    st.write("""##### Tra Ciro, Ambrogio, Giuseppe, Martin, Lorenzo""")

    text = st.text_area("Testo")

    ok = st.button("Indovina")
    if ok:
        X = ctv.transform([text])

        person = model.predict(X)
        probs = model.predict_proba(X)[0]*100
        formatted_maxprob = "{:.2f}".format(max(probs))
        st.subheader(f"Sono sicuro al al {formatted_maxprob}% che a scrivere sia stato {le.inverse_transform(person)[0]}")
        st.write("""##### Livello di confidenza:""")
        for idx, p in enumerate(le.classes_):
            st.write(f"###### {p}:")
            st.write(f"###### {'{:.2f}'.format(probs[idx])}%")
            st.progress(int(probs[idx]))
