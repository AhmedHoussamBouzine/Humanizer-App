import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from transformer.app import EnhancedFrenchTextHumanizer, NLP_FR
import time
import io


def main():
    """
    Enhanced Streamlit app to humanize French text using the improved humanizer:
    - Multiple tone options (academic/casual)
    - Advanced text analysis and visualization
    - Configurable parameters
    - Batch processing support
    """
    
    st.set_page_config(
        page_title="Humaniseur de Texte IA Avancé",
        page_icon="🧔🏻‍♂️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "# Application avancée pour humaniser du texte généré par IA en français avec analyse détaillée."
        }
    )

    # Custom CSS
    st.markdown(
        """
        <style>
        .title {
            text-align: center;
            font-size: 2.5em;
            font-weight: bold;
            margin-top: 0.5em;
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .intro {
            text-align: left;
            line-height: 1.8;
            margin-bottom: 1.5em;
            padding: 1em;
            background-color: #f0f2f6;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .feature-box {
            background-color: #f8f9fa;
            padding: 1em;
            border-radius: 8px;
            margin: 0.5em 0;
            border: 1px solid #dee2e6;
        }
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1em;
            border-radius: 10px;
            text-align: center;
            margin: 0.5em;
        }
        .stButton > button {
            background: linear-gradient(90deg, #1f77b4, #ff7f0e);
            color: white;
            border: none;
            border-radius: 20px;
            padding: 0.5em 2em;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header
    st.markdown("<div class='title'>🧔🏻‍♂️ Humaniseur de Texte IA Avancé 🤖</div>", unsafe_allow_html=True)
    
    st.markdown(
        """
        <div class='intro'>
        <h4>🚀 Transformez votre texte IA en contenu naturel et humain</h4>
        <p><b>Fonctionnalités avancées :</b></p>
        <ul>
        <li>🎯 <b>Tons multiples</b> : Académique ou décontracté</li>
        <li>🔄 <b>Remplacement intelligent</b> : Synonymes contextuels</li>
        <li>📊 <b>Analyse détaillée</b> : Métriques de complexité</li>
        <li>⚙️ <b>Paramètres configurables</b> : Contrôle précis des transformations</li>
        <li>📁 <b>Traitement par lots</b> : Plusieurs fichiers simultanément</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for advanced configuration
    with st.sidebar:
        st.header("⚙️ Configuration Avancée")
        
        # Tone selection
        tone = st.selectbox(
            "🎨 Ton du texte",
            options=["academic", "casual"],
            format_func=lambda x: "🎓 Académique" if x == "academic" else "💬 Décontracté",
            help="Choisissez le style de transformation souhaité"
        )
        
        st.markdown("---")
        
        # Advanced parameters
        st.subheader("🔧 Paramètres de Transformation")
        
        p_synonym = st.slider(
            "Remplacement de synonymes",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Probabilité de remplacer un mot par un synonyme"
        )
        
        p_restructure = st.slider(
            "Restructuration de phrases",
            min_value=0.0,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Probabilité de restructurer une phrase"
        )
        
        p_transition = st.slider(
            "Ajout de transitions",
            min_value=0.0,
            max_value=0.5,
            value=0.15,
            step=0.05,
            help="Probabilité d'ajouter des mots de transition"
        )
        
        p_errors = st.slider(
            "Imperfections naturelles",
            min_value=0.0,
            max_value=0.2,
            value=0.05,
            step=0.01,
            help="Probabilité d'ajouter de petites imperfections naturelles"
        )
        
        preserve_meaning = st.checkbox(
            "🔒 Préserver le sens strict",
            value=True,
            help="Utilise la similarité sémantique pour un meilleur choix de synonymes"
        )
        
        show_analysis = st.checkbox(
            "📊 Afficher l'analyse détaillée",
            value=True,
            help="Montre les métriques et graphiques d'analyse"
        )

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📝 Texte d'Entrée")
        
        # Text input methods
        input_method = st.radio(
            "Méthode d'entrée :",
            ["Saisie directe", "Fichier unique", "Fichiers multiples"],
            horizontal=True
        )
        
        user_text = ""
        files_data = []
        
        if input_method == "Saisie directe":
            user_text = st.text_area(
                "Entrez votre texte ici :",
                height=200,
                placeholder="Collez votre texte généré par IA ici..."
            )
            
        elif input_method == "Fichier unique":
            uploaded_file = st.file_uploader(
                "Téléversez un fichier :",
                type=["txt", "md"],
                help="Formats supportés : .txt, .md"
            )
            if uploaded_file is not None:
                user_text = uploaded_file.read().decode("utf-8", errors="ignore")
                st.text_area("Aperçu du fichier :", value=user_text[:500] + "..." if len(user_text) > 500 else user_text, height=100)
                
        else:  # Multiple files
            uploaded_files = st.file_uploader(
                "Téléversez plusieurs fichiers :",
                type=["txt", "md"],
                accept_multiple_files=True,
                help="Sélectionnez plusieurs fichiers pour un traitement par lots"
            )
            if uploaded_files:
                for file in uploaded_files:
                    content = file.read().decode("utf-8", errors="ignore")
                    files_data.append({"name": file.name, "content": content})
                st.success(f"✅ {len(files_data)} fichier(s) chargé(s)")

    with col2:
        st.subheader("✨ Résultat Transformé")
        
        if st.button("🚀 Transformer le Texte", type="primary"):
            if input_method != "Fichiers multiples" and not user_text.strip():
                st.warning("⚠️ Veuillez entrer ou téléverser un texte à transformer.")
            elif input_method == "Fichiers multiples" and not files_data:
                st.warning("⚠️ Veuillez téléverser au moins un fichier.")
            else:
                with st.spinner("🔄 Transformation en cours..."):
                    try:
                        # Initialize humanizer with custom parameters
                        humanizer = EnhancedFrenchTextHumanizer(
                            p_synonym_replacement=p_synonym,
                            p_sentence_restructure=p_restructure,
                            p_transition_add=p_transition,
                            p_minor_errors=p_errors,
                            tone=tone,
                            seed=42  # For reproducible results
                        )
                        
                        if input_method != "Fichiers multiples":
                            # Single text processing
                            process_single_text(user_text, humanizer, preserve_meaning, show_analysis)
                        else:
                            # Batch processing
                            process_multiple_files(files_data, humanizer, preserve_meaning, show_analysis)
                            
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la transformation : {str(e)}")
                        st.info("💡 Assurez-vous que le modèle SpaCy français est installé : `python -m spacy download fr_core_news_sm`")


def process_single_text(text, humanizer, preserve_meaning, show_analysis):
    """Process a single text input"""
    start_time = time.time()
    
    # Get original text analysis
    original_analysis = humanizer.analyze_text_features(text)
    
    # Transform text
    transformed = humanizer.humanize_text(text, preserve_meaning=preserve_meaning)
    
    # Get transformed text analysis
    transformed_analysis = humanizer.analyze_text_features(transformed)
    
    processing_time = time.time() - start_time
    
    # Display results
    st.markdown("### 📋 Texte Transformé")
    st.markdown(f"```\n{transformed}\n```")
    
    # Download button
    st.download_button(
        label="💾 Télécharger le résultat",
        data=transformed,
        file_name="texte_humanise.txt",
        mime="text/plain"
    )
    
    # Statistics
    display_statistics(original_analysis, transformed_analysis, processing_time)
    
    # Detailed analysis
    if show_analysis:
        display_detailed_analysis(original_analysis, transformed_analysis)


def process_multiple_files(files_data, humanizer, preserve_meaning, show_analysis):
    """Process multiple files in batch"""
    st.markdown("### 📁 Traitement par Lots")
    
    results = []
    progress_bar = st.progress(0)
    
    for i, file_data in enumerate(files_data):
        st.markdown(f"**Traitement de : {file_data['name']}**")
        
        # Transform text
        transformed = humanizer.humanize_text(file_data['content'], preserve_meaning=preserve_meaning)
        
        results.append({
            "filename": file_data['name'],
            "original": file_data['content'],
            "transformed": transformed
        })
        
        # Show preview
        with st.expander(f"Aperçu : {file_data['name']}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original :**")
                st.text_area("", value=file_data['content'][:300] + "...", height=100, key=f"orig_{i}")
            with col2:
                st.markdown("**Transformé :**")
                st.text_area("", value=transformed[:300] + "...", height=100, key=f"trans_{i}")
        
        progress_bar.progress((i + 1) / len(files_data))
    
    # Create downloadable zip or combined file
    combined_results = "\n\n" + "="*50 + "\n\n".join([
        f"FICHIER: {result['filename']}\n\n{result['transformed']}" 
        for result in results
    ])
    
    st.download_button(
        label="💾 Télécharger tous les résultats",
        data=combined_results,
        file_name="textes_humanises_batch.txt",
        mime="text/plain"
    )
    
    st.success(f"✅ {len(results)} fichier(s) traité(s) avec succès !")


def display_statistics(original_analysis, transformed_analysis, processing_time):
    """Display comparison statistics"""
    st.markdown("### 📊 Statistiques Comparatives")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Mots",
            transformed_analysis['word_count'],
            delta=transformed_analysis['word_count'] - original_analysis['word_count']
        )
    
    with col2:
        st.metric(
            "Phrases",
            transformed_analysis['sentence_count'],
            delta=transformed_analysis['sentence_count'] - original_analysis['sentence_count']
        )
    
    with col3:
        st.metric(
            "Diversité lexicale",
            f"{transformed_analysis['lexical_diversity']:.3f}",
            delta=f"{transformed_analysis['lexical_diversity'] - original_analysis['lexical_diversity']:.3f}"
        )
    
    with col4:
        st.metric(
            "Temps de traitement",
            f"{processing_time:.2f}s"
        )


def display_detailed_analysis(original_analysis, transformed_analysis):
    """Display detailed analysis with charts"""
    st.markdown("### 🔍 Analyse Détaillée")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Longueur Moyenne des Phrases")
        comparison_data = pd.DataFrame({
            'Version': ['Original', 'Transformé'],
            'Longueur Moyenne': [
                original_analysis['avg_sentence_length'],
                transformed_analysis['avg_sentence_length']
            ]
        })
        
        fig = px.bar(
            comparison_data, 
            x='Version', 
            y='Longueur Moyenne',
            color='Version',
            title="Comparaison de la longueur des phrases"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 🏷️ Distribution des Classes Grammaticales")
        
        # Combine POS data
        all_pos = set(original_analysis['pos_distribution'].keys()) | set(transformed_analysis['pos_distribution'].keys())
        pos_data = []
        
        for pos in all_pos:
            pos_data.extend([
                {'POS': pos, 'Count': original_analysis['pos_distribution'].get(pos, 0), 'Version': 'Original'},
                {'POS': pos, 'Count': transformed_analysis['pos_distribution'].get(pos, 0), 'Version': 'Transformé'}
            ])
        
        pos_df = pd.DataFrame(pos_data)
        
        fig = px.bar(
            pos_df, 
            x='POS', 
            y='Count', 
            color='Version',
            barmode='group',
            title="Distribution des classes grammaticales"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Complexity comparison
    st.markdown("#### 🧠 Score de Complexité")
    complexity_data = pd.DataFrame({
        'Métrique': ['Complexité'],
        'Original': [original_analysis['complexity_score']],
        'Transformé': [transformed_analysis['complexity_score']]
    })
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Original', x=complexity_data['Métrique'], y=complexity_data['Original']))
    fig.add_trace(go.Bar(name='Transformé', x=complexity_data['Métrique'], y=complexity_data['Transformé']))
    fig.update_layout(title="Comparaison de la complexité textuelle")
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()