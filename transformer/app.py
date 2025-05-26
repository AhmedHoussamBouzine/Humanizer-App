import ssl
import random
import warnings
import re
from typing import List, Dict, Optional, Tuple
import json

import spacy
from sentence_transformers import SentenceTransformer, util
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)



def load_french_model():
    try:
        return spacy.load("fr_core_news_sm")
    except OSError:
        raise RuntimeError("French SpaCy model is required. Make sure it's installed.")

NLP_FR = load_french_model()

class EnhancedFrenchTextHumanizer:
    """
    Enhanced French text humanizer that transforms AI-generated text to appear more human-like.
    Features:
    - Improved synonym replacement with French synonym dictionary
    - Better sentence structure variation
    - Academic and casual tone options
    - Error injection for natural imperfections
    - Contextual transformations
    """

    def __init__(
        self,
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        p_synonym_replacement=0.25,
        p_sentence_restructure=0.2,
        p_transition_add=0.15,
        p_minor_errors=0.05,
        tone='academic',
        seed=None
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        if NLP_FR is None:
            raise RuntimeError("French SpaCy model is required")
            
        self.nlp = NLP_FR
        
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load sentence transformer: {e}")
            self.model = None

        self.p_synonym_replacement = p_synonym_replacement
        self.p_sentence_restructure = p_sentence_restructure
        self.p_transition_add = p_transition_add
        self.p_minor_errors = p_minor_errors
        self.tone = tone

        self._load_resources()

    def _load_resources(self):
        """Load linguistic resources and dictionaries"""
        
        # Enhanced French synonyms dictionary
        self.french_synonyms = {
            # Common verbs
            'dire': ['affirmer', 'déclarer', 'exprimer', 'mentionner', 'soutenir'],
            'faire': ['réaliser', 'effectuer', 'accomplir', 'exécuter'],
            'voir': ['observer', 'constater', 'remarquer', 'percevoir'],
            'donner': ['fournir', 'offrir', 'accorder', 'procurer'],
            'prendre': ['saisir', 'adopter', 'choisir', 'capturer'],
            'venir': ['arriver', 'survenir', 'apparaître'],
            'aller': ['se rendre', 'se diriger', 'partir'],
            'savoir': ['connaître', 'maîtriser', 'comprendre'],
            'pouvoir': ['être capable', 'avoir la possibilité'],
            'vouloir': ['désirer', 'souhaiter', 'ambitionner'],
            
            # Common adjectives
            'grand': ['important', 'considérable', 'majeur', 'significatif'],
            'petit': ['réduit', 'modeste', 'limité', 'mineur'],
            'bon': ['excellent', 'satisfaisant', 'approprié', 'convenable'],
            'mauvais': ['défavorable', 'négatif', 'inadéquat'],
            'important': ['essentiel', 'crucial', 'fondamental', 'primordial'],
            'différent': ['distinct', 'varié', 'divers', 'autre'],
            'possible': ['réalisable', 'envisageable', 'potentiel'],
            'nouveau': ['récent', 'moderne', 'inédit', 'innovant'],
            'premier': ['initial', 'principal', 'primaire'],
            'dernier': ['final', 'ultime', 'récent'],
            
            # Common nouns
            'problème': ['difficulté', 'enjeu', 'défi', 'obstacle'],
            'solution': ['résolution', 'réponse', 'remède'],
            'question': ['interrogation', 'problématique', 'sujet'],
            'résultat': ['conséquence', 'effet', 'aboutissement'],
            'exemple': ['illustration', 'cas', 'modèle'],
            'raison': ['motif', 'cause', 'justification'],
            'temps': ['période', 'durée', 'époque'],
            'travail': ['activité', 'tâche', 'emploi', 'œuvre'],
            'vie': ['existence', 'vécu', 'expérience'],
            'monde': ['univers', 'société', 'planète'],
            
            # Common adverbs
            'très': ['extrêmement', 'particulièrement', 'fort', 'vraiment'],
            'bien': ['correctement', 'convenablement', 'parfaitement'],
            'mal': ['incorrectement', 'défavorablement'],
            'souvent': ['fréquemment', 'régulièrement', 'habituellement'],
            'toujours': ['constamment', 'perpétuellement', 'continuellement'],
            'jamais': ['nullement', 'en aucun cas'],
            'beaucoup': ['énormément', 'considérablement', 'grandement'],
            'peu': ['faiblement', 'légèrement', 'modérément']
        }

        # Academic transitions by category
        self.transitions = {
            'academic': {
                'addition': ['De plus,', 'En outre,', 'Par ailleurs,', 'De surcroît,', 'Qui plus est,'],
                'contrast': ['Néanmoins,', 'Toutefois,', 'Cependant,', 'En revanche,', 'Malgré cela,'],
                'consequence': ['Par conséquent,', 'Ainsi,', 'De ce fait,', 'Il en résulte que', 'Dès lors,'],
                'explanation': ['En effet,', 'Effectivement,', 'À vrai dire,', 'Autrement dit,'],
                'example': ['Par exemple,', 'Notamment,', 'À titre d\'illustration,', 'Ainsi,'],
                'conclusion': ['En conclusion,', 'Pour conclure,', 'Finalement,', 'En définitive,']
            },
            'casual': {
                'addition': ['Aussi,', 'Et puis,', 'En plus,', 'D\'ailleurs,'],
                'contrast': ['Mais,', 'Pourtant,', 'Quand même,', 'Malgré tout,'],
                'consequence': ['Du coup,', 'Alors,', 'Donc,', 'Résultat,'],
                'explanation': ['En fait,', 'Bon,', 'Disons que,', 'C\'est-à-dire,'],
                'example': ['Par exemple,', 'Tiens,', 'Comme,'],
                'conclusion': ['Bref,', 'Bon,', 'Voilà,', 'Au final,']
            }
        }

        # Common French filler words and expressions
        self.fillers = {
            'academic': ['il convient de noter', 'il est à souligner', 'on peut observer'],
            'casual': ['on peut dire', 'il faut dire', 'c\'est vrai que']
        }

        # Sentence starters for variety
        self.sentence_starters = {
            'academic': [
                'Il convient de souligner que',
                'Il est important de noter que',
                'On peut observer que',
                'Il apparaît que',
                'Force est de constater que'
            ],
            'casual': [
                'On peut dire que',
                'Il faut reconnaître que',
                'C\'est vrai que',
                'Bon, il faut dire que'
            ]
        }

    def humanize_text(self, text: str, preserve_meaning=True) -> str:
        """
        Main method to humanize French text
        
        Args:
            text: Input text to humanize
            preserve_meaning: Whether to preserve original meaning strictly
            
        Returns:
            Humanized text
        """
        if not text.strip():
            return text

        # Split into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        
        transformed_sentences = []
        
        for i, sentence in enumerate(sentences):
            transformed = sentence
            
            # Apply transformations with controlled randomness
            if random.random() < self.p_synonym_replacement:
                transformed = self._replace_synonyms(transformed, preserve_meaning)
            
            if random.random() < self.p_sentence_restructure:
                transformed = self._restructure_sentence(transformed)
            
            if random.random() < self.p_transition_add and i > 0:
                transformed = self._add_transition(transformed)
            
            # Occasionally add sentence starters for variety
            if random.random() < 0.1 and not any(transformed.startswith(t) for transitions in self.transitions[self.tone].values() for t in transitions):
                transformed = self._add_sentence_starter(transformed)
            
            # Add minor imperfections occasionally
            if random.random() < self.p_minor_errors:
                transformed = self._add_minor_imperfection(transformed)
            
            transformed_sentences.append(transformed)
        
        result = ' '.join(transformed_sentences)
        
        # Final cleanup
        result = self._cleanup_text(result)
        
        return result

    def _replace_synonyms(self, sentence: str, preserve_meaning: bool = True) -> str:
        """Replace words with appropriate synonyms"""
        doc = self.nlp(sentence)
        new_tokens = []
        
        for token in doc:
            word = token.text.lower()
            
            # Only replace content words
            if token.pos_ in {'NOUN', 'VERB', 'ADJ', 'ADV'} and len(word) > 3:
                lemma = token.lemma_.lower()
                
                if lemma in self.french_synonyms:
                    synonyms = self.french_synonyms[lemma]
                    
                    if preserve_meaning and self.model:
                        # Use semantic similarity for better synonym selection
                        best_synonym = self._select_best_synonym(word, synonyms)
                        if best_synonym:
                            new_tokens.append(best_synonym)
                        else:
                            new_tokens.append(token.text)
                    else:
                        # Random selection
                        if random.random() < 0.4:  # 40% chance to replace
                            new_tokens.append(random.choice(synonyms))
                        else:
                            new_tokens.append(token.text)
                else:
                    new_tokens.append(token.text)
            else:
                new_tokens.append(token.text)
        
        return self._reconstruct_sentence(doc, new_tokens)

    def _select_best_synonym(self, original_word: str, synonyms: List[str]) -> Optional[str]:
        """Select the most semantically similar synonym"""
        if not self.model or not synonyms:
            return None
            
        try:
            original_emb = self.model.encode(original_word, convert_to_tensor=True)
            synonym_embs = self.model.encode(synonyms, convert_to_tensor=True)
            
            cos_scores = util.cos_sim(original_emb, synonym_embs)[0]
            max_score_idx = cos_scores.argmax().item()
            max_score = cos_scores[max_score_idx].item()
            
            # Only use synonym if similarity is reasonable
            if max_score >= 0.6:
                return synonyms[max_score_idx]
        except Exception:
            pass
            
        return None

    def _restructure_sentence(self, sentence: str) -> str:
        """Apply simple sentence restructuring"""
        doc = self.nlp(sentence)
        
        # Simple transformations
        if random.random() < 0.5:
            # Move adverbial phrases
            sentence_lower = sentence.lower()
            if sentence_lower.startswith(('cependant', 'néanmoins', 'toutefois')):
                # Move transition to middle or end sometimes
                parts = sentence.split(',', 1)
                if len(parts) == 2:
                    return f"{parts[1].strip()}, {parts[0].lower()}"
        
        return sentence

    def _add_transition(self, sentence: str) -> str:
        """Add appropriate transition words"""
        transition_type = random.choice(list(self.transitions[self.tone].keys()))
        transition = random.choice(self.transitions[self.tone][transition_type])
        
        return f"{transition} {sentence}"

    def _add_sentence_starter(self, sentence: str) -> str:
        """Add variety with sentence starters"""
        starter = random.choice(self.sentence_starters[self.tone])
        
        # Make sure the sentence flows well
        if sentence.lower().startswith(('il', 'on', 'c\'est', 'ce')):
            return f"{starter} {sentence}"
        else:
            return f"{starter} {sentence}"

    def _add_minor_imperfection(self, sentence: str) -> str:
        """Add subtle imperfections to make text more human"""
        imperfections = [
            # Add casual filler
            lambda s: f"{random.choice(self.fillers[self.tone])}, {s.lower()}",
            # Minor repetition
            lambda s: s.replace(' et ', ' et puis ') if ' et ' in s else s,
            # Slight informality
            lambda s: s.replace(' très ', ' assez ' if random.random() < 0.5 else ' plutôt '),
        ]
        
        if random.random() < 0.3:  # Only apply occasionally
            imperfection = random.choice(imperfections)
            return imperfection(sentence)
        
        return sentence

    def _reconstruct_sentence(self, doc, new_tokens: List[str]) -> str:
        """Reconstruct sentence preserving punctuation and spacing"""
        result = []
        for i, token in enumerate(doc):
            if i < len(new_tokens):
                result.append(new_tokens[i])
                if token.whitespace_:
                    result.append(token.whitespace_)
        
        return ''.join(result)

    def _cleanup_text(self, text: str) -> str:
        """Final cleanup of the text"""
        # Fix double spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*', r'\1 ', text)
        
        # Ensure proper capitalization after periods
        sentences = text.split('. ')
        sentences = [s.strip().capitalize() if s else s for s in sentences]
        text = '. '.join(sentences)
        
        return text.strip()

    def analyze_text_features(self, text: str) -> Dict:
        """Analyze text features to help with detection avoidance"""
        doc = self.nlp(text)
        
        # Calculate various metrics
        sentences = list(doc.sents)
        words = [token for token in doc if not token.is_punct and not token.is_space]
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        unique_words = len(set(token.lemma_.lower() for token in words))
        lexical_diversity = unique_words / len(words) if words else 0
        
        # POS distribution
        pos_counts = {}
        for token in words:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1
        
        return {
            'sentence_count': len(sentences),
            'word_count': len(words),
            'avg_sentence_length': avg_sentence_length,
            'lexical_diversity': lexical_diversity,
            'pos_distribution': pos_counts,
            'complexity_score': self._calculate_complexity(doc)
        }

    def _calculate_complexity(self, doc) -> float:
        """Calculate text complexity score"""
        # Simple complexity based on sentence length variation and vocabulary
        sentences = list(doc.sents)
        if not sentences:
            return 0
        
        lengths = [len([t for t in sent if not t.is_punct]) for sent in sentences]
        length_variance = np.var(lengths) if len(lengths) > 1 else 0
        
        return min(length_variance / 10, 1.0)  # Normalize to 0-1