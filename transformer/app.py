import ssl
import random
import warnings

import spacy
from sentence_transformers import SentenceTransformer, util

warnings.filterwarnings("ignore", category=FutureWarning)

# Load French SpaCy model globally
NLP_FR = spacy.load("fr_core_news_sm")

def download_nltk_resources():
    """
    Dummy function for compatibility.
    No NLTK resources are needed here for French.
    """
    pass


class FrenchTextHumanizer:
    """
    Transforms French text into a more formal academic style:
      - Adds academic transitions in French
      - Optionally replaces words with synonyms (using simple WordNet fallback if any)
      - Uses SpaCy for POS tagging and lemmatization
    """

    def __init__(
        self,
        model_name='paraphrase-multilingual-MiniLM-L12-v2',
        p_synonym_replacement=0.3,
        p_academic_transition=0.3,
        seed=None
    ):
        if seed is not None:
            random.seed(seed)

        self.nlp = NLP_FR
        self.model = SentenceTransformer(model_name)

        self.p_synonym_replacement = p_synonym_replacement
        self.p_academic_transition = p_academic_transition

        self.academic_transitions = [
            "De plus,", "En outre,", "Par conséquent,", "Ainsi,", 
            "Néanmoins,", "Toutefois,", "D'autre part,", "En revanche,"
        ]

    def humanize_text(self, text, use_synonyms=False):
        doc = self.nlp(text)
        transformed_sentences = []

        for sent in doc.sents:
            sentence_str = sent.text.strip()

            # Possibly add academic transitions
            if random.random() < self.p_academic_transition:
                sentence_str = self.add_academic_transitions(sentence_str)

            # Optionally replace words with synonyms
            if use_synonyms and random.random() < self.p_synonym_replacement:
                sentence_str = self.replace_with_synonyms(sentence_str)

            transformed_sentences.append(sentence_str)

        return ' '.join(transformed_sentences)

    def add_academic_transitions(self, sentence):
        transition = random.choice(self.academic_transitions)
        return f"{transition} {sentence}"

    def replace_with_synonyms(self, sentence):
        doc = self.nlp(sentence)
        new_tokens = []

        for token in doc:
            word = token.text
            pos = token.pos_

            # Focus on adjectives, nouns, verbs, and adverbs
            if pos in {'ADJ', 'NOUN', 'VERB', 'ADV'}:
                synonyms = self._get_synonyms(token.lemma_)
                if synonyms and random.random() < 0.5:
                    best_synonym = self._select_closest_synonym(word, synonyms)
                    new_tokens.append(best_synonym if best_synonym else word)
                else:
                    new_tokens.append(word)
            else:
                new_tokens.append(word)

        return ' '.join(new_tokens)

    def _get_synonyms(self, lemma):
        # Simple placeholder: French WordNet access is limited,
        # so return an empty list or implement your own synonym dictionary.
        # You can integrate WOLF or any French synonym DB here.
        return []

    def _select_closest_synonym(self, original_word, synonyms):
        if not synonyms:
            return None
        original_emb = self.model.encode(original_word, convert_to_tensor=True)
        synonym_embs = self.model.encode(synonyms, convert_to_tensor=True)
        cos_scores = util.cos_sim(original_emb, synonym_embs)[0]
        max_score_index = cos_scores.argmax().item()
        max_score = cos_scores[max_score_index].item()
        if max_score >= 0.5:
            return synonyms[max_score_index]
        return None
