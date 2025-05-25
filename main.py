import streamlit as st
from transformer.app import FrenchTextHumanizer, NLP_FR, download_nltk_resources
from nltk.tokenize import word_tokenize  # Only if needed for stats; else remove

def main():
    """
    Streamlit app to humanize French text into a formal academic style:
    - Adds French academic transitions
    - Optionally replaces words with synonyms (limited by dictionary)
    """
    # No NLTK downloads needed for French, but keep for compatibility
    download_nltk_resources()

    st.set_page_config(
        page_title="Humaniser le texte généré par IA",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# Cette application permet d'humaniser du texte généré par IA en français."
        }
    )

    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 2em;
            font-weight: bold;
            margin-top: 0.5em;
        }
        .intro {
            text-align: left;
            line-height: 1.6;
            margin-bottom: 1.2em;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='title'>🧔🏻‍♂️Humaniser le texte IA🤖 généré</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='intro'>
        <p><b>Cette application transforme votre texte en style académique formel en français :</b><br>
        • Ajoute des transitions académiques<br>
        • <em>Optionnellement</em> remplace certains mots par des synonymes<br>
        </p><hr></div>
        """,
        unsafe_allow_html=True
    )

    # Checkbox for synonym replacement
    use_synonyms = st.checkbox("Activer le remplacement par synonymes", value=False)

    # Text input area
    user_text = st.text_area("Entrez votre texte ici :")

    # File upload option
    uploaded_file = st.file_uploader("Ou téléversez un fichier .txt :", type=["txt"])
    if uploaded_file is not None:
        file_text = uploaded_file.read().decode("utf-8", errors="ignore")
        user_text = file_text

    if st.button("Transformer en style académique"):
        if not user_text.strip():
            st.warning("Veuillez entrer ou téléverser un texte à transformer.")
        else:
            with st.spinner("Transformation en cours..."):
                # Input stats
                # Word tokenization in French (optional)
                input_word_count = len(user_text.split())
                doc_input = NLP_FR(user_text)
                input_sentence_count = len(list(doc_input.sents))

                # Instantiate humanizer
                humanizer = FrenchTextHumanizer(
                    p_synonym_replacement=0.3,
                    p_academic_transition=0.4
                )

                # Transform text
                transformed = humanizer.humanize_text(
                    user_text,
                    use_synonyms=use_synonyms
                )

                # Output transformed text
                st.subheader("Texte transformé :")
                st.write(transformed)

                # Output stats
                output_word_count = len(transformed.split())
                doc_output = NLP_FR(transformed)
                output_sentence_count = len(list(doc_output.sents))

                st.markdown(
                    f"**Nombre de mots (entrée)**: {input_word_count} "
                    f"| **Nombre de phrases (entrée)**: {input_sentence_count}  "
                    f"| **Nombre de mots (sortie)**: {output_word_count} "
                    f"| **Nombre de phrases (sortie)**: {output_sentence_count}"
                )

    st.markdown("---")


if __name__ == "__main__":
    main()
