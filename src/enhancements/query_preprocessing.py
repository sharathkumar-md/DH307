"""
Query Preprocessing Module - Improves Robustness to Errors

Implements:
1. Spell correction
2. Medical abbreviation expansion
3. Query normalization
4. Medical term standardization

Author: Sharath Kumar MD
Date: 2025-10-25
"""

from typing import Dict, List, Tuple
import re
from textblob import TextBlob
from spellchecker import SpellChecker

####################################################################
#                    MEDICAL KNOWLEDGE BASE
####################################################################

# Medical abbreviations dictionary
MEDICAL_ABBREVIATIONS = {
    # Blood Pressure related
    'bp': 'blood pressure',
    'b.p': 'blood pressure',
    'sbp': 'systolic blood pressure',
    'dbp': 'diastolic blood pressure',

    # Diabetes
    'dm': 'diabetes mellitus',
    'd.m': 'diabetes mellitus',
    'fbs': 'fasting blood sugar',
    'ppbs': 'postprandial blood sugar',
    'hba1c': 'hemoglobin a1c',

    # Hypertension
    'htn': 'hypertension',
    'h.t.n': 'hypertension',

    # Body metrics
    'bmi': 'body mass index',
    'b.m.i': 'body mass index',

    # Diseases
    'ncd': 'non communicable disease',
    'n.c.d': 'non communicable disease',
    'cvd': 'cardiovascular disease',
    'chd': 'coronary heart disease',
    'copd': 'chronic obstructive pulmonary disease',

    # Healthcare roles
    'cho': 'community health officer',
    'c.h.o': 'community health officer',
    'phc': 'primary health center',
    'p.h.c': 'primary health center',

    # Medical terms
    'ecg': 'electrocardiogram',
    'echo': 'echocardiogram',
    'mi': 'myocardial infarction',
    'cad': 'coronary artery disease',

    # Measurements
    'mmhg': 'millimeters of mercury',
    'mg/dl': 'milligrams per deciliter',
    'kg/m2': 'kilogram per meter square',
}

# Common medical term misspellings
MEDICAL_SPELL_CORRECTIONS = {
    'diabetis': 'diabetes',
    'diabeties': 'diabetes',
    'diabets': 'diabetes',
    'hypertention': 'hypertension',
    'hypertenstion': 'hypertension',
    'obesaty': 'obesity',
    'obesety': 'obesity',
    'colesterol': 'cholesterol',
    'cholestrol': 'cholesterol',
    'cardio vascular': 'cardiovascular',
    'treatement': 'treatment',
    'symtoms': 'symptoms',
    'symptom': 'symptoms',
    'diagnois': 'diagnosis',
    'diagosis': 'diagnosis',
    'medicin': 'medicine',
    'preasure': 'pressure',
    'preassure': 'pressure',
    'hemorrage': 'hemorrhage',
    'hemorage': 'hemorrhage',
}

# Medical term synonyms (for query expansion)
MEDICAL_SYNONYMS = {
    'high blood pressure': 'hypertension',
    'sugar': 'diabetes',
    'blood sugar': 'diabetes',
    'heart attack': 'myocardial infarction',
    'stroke': 'cerebrovascular accident',
    'weight': 'obesity',
    'overweight': 'obesity',
}

####################################################################
#                    QUERY PREPROCESSOR CLASS
####################################################################

class QueryPreprocessor:
    """Preprocess queries to improve robustness"""

    def __init__(self,
                 enable_spell_correction: bool = True,
                 enable_abbreviation_expansion: bool = True,
                 enable_synonym_expansion: bool = False,
                 custom_medical_dictionary: Dict[str, str] = None):
        """
        Initialize query preprocessor

        Args:
            enable_spell_correction: Enable spelling correction
            enable_abbreviation_expansion: Expand medical abbreviations
            enable_synonym_expansion: Expand medical synonyms
            custom_medical_dictionary: Additional medical terms
        """
        self.enable_spell_correction = enable_spell_correction
        self.enable_abbreviation_expansion = enable_abbreviation_expansion
        self.enable_synonym_expansion = enable_synonym_expansion

        # Initialize spell checker with medical terms
        self.spell_checker = SpellChecker()

        # Add medical terms to dictionary
        medical_terms = set(MEDICAL_ABBREVIATIONS.values())
        medical_terms.update(['hypertension', 'diabetes', 'obesity', 'cholesterol',
                            'cardiovascular', 'blood', 'pressure', 'treatment',
                            'symptoms', 'diagnosis', 'medicine', 'patient'])
        self.spell_checker.word_frequency.load_words(medical_terms)

        # Load custom dictionary if provided
        if custom_medical_dictionary:
            MEDICAL_ABBREVIATIONS.update(custom_medical_dictionary)

    def preprocess(self, query: str, verbose: bool = False) -> Dict:
        """
        Main preprocessing pipeline

        Returns:
            Dict with original query, processed query, and modifications made
        """
        original_query = query
        modifications = []

        # Step 1: Normalize (lowercase, strip)
        query = query.strip()

        # Step 2: Expand abbreviations
        if self.enable_abbreviation_expansion:
            query, abbrev_mods = self._expand_abbreviations(query)
            modifications.extend(abbrev_mods)

        # Step 3: Apply medical spell corrections (domain-specific)
        query, medical_spell_mods = self._apply_medical_corrections(query)
        modifications.extend(medical_spell_mods)

        # Step 4: General spell correction
        if self.enable_spell_correction:
            query, spell_mods = self._spell_correct(query)
            modifications.extend(spell_mods)

        # Step 5: Synonym expansion (optional)
        if self.enable_synonym_expansion:
            query, synonym_mods = self._expand_synonyms(query)
            modifications.extend(synonym_mods)

        # Step 6: Final normalization
        query = self._normalize_text(query)

        result = {
            'original_query': original_query,
            'processed_query': query,
            'modifications': modifications,
            'num_modifications': len(modifications)
        }

        if verbose:
            print(f"Original:  {original_query}")
            print(f"Processed: {query}")
            print(f"Modifications: {len(modifications)}")
            for mod in modifications:
                print(f"  - {mod}")

        return result

    def _expand_abbreviations(self, query: str) -> Tuple[str, List[str]]:
        """Expand medical abbreviations"""
        modifications = []
        query_lower = query.lower()

        # Sort by length (longest first) to avoid partial replacements
        sorted_abbrevs = sorted(MEDICAL_ABBREVIATIONS.items(),
                               key=lambda x: len(x[0]),
                               reverse=True)

        for abbrev, expansion in sorted_abbrevs:
            # Match abbreviation as whole word (with word boundaries)
            pattern = r'\b' + re.escape(abbrev) + r'\b'

            if re.search(pattern, query_lower):
                # Replace while preserving other case
                query_lower = re.sub(pattern, expansion, query_lower)
                modifications.append(f"Expanded '{abbrev}' → '{expansion}'")

        return query_lower, modifications

    def _apply_medical_corrections(self, query: str) -> Tuple[str, List[str]]:
        """Apply medical-specific spell corrections"""
        modifications = []

        for misspell, correct in MEDICAL_SPELL_CORRECTIONS.items():
            if misspell in query.lower():
                query = re.sub(re.escape(misspell), correct, query, flags=re.IGNORECASE)
                modifications.append(f"Medical correction '{misspell}' → '{correct}'")

        return query, modifications

    def _spell_correct(self, query: str) -> Tuple[str, List[str]]:
        """Apply general spell correction using pyspellchecker"""
        modifications = []
        words = query.split()
        corrected_words = []

        for word in words:
            # Skip short words, numbers, punctuation
            if len(word) < 3 or not word.isalpha():
                corrected_words.append(word)
                continue

            # Check if misspelled
            word_lower = word.lower()
            if word_lower not in self.spell_checker:
                # Get correction
                correction = self.spell_checker.correction(word_lower)

                if correction and correction != word_lower:
                    corrected_words.append(correction)
                    modifications.append(f"Spell corrected '{word}' → '{correction}'")
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words), modifications

    def _expand_synonyms(self, query: str) -> Tuple[str, List[str]]:
        """Expand medical synonyms"""
        modifications = []
        query_lower = query.lower()

        for synonym, standard_term in MEDICAL_SYNONYMS.items():
            if synonym in query_lower:
                # Add standard term alongside synonym
                query_lower = query_lower + f" {standard_term}"
                modifications.append(f"Added synonym '{standard_term}' for '{synonym}'")

        return query_lower, modifications

    def _normalize_text(self, text: str) -> str:
        """Final text normalization"""
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove extra punctuation
        text = re.sub(r'([.,!?;:])\1+', r'\1', text)

        return text.strip()

    def batch_preprocess(self, queries: List[str], verbose: bool = False) -> List[Dict]:
        """Preprocess multiple queries"""
        return [self.preprocess(q, verbose) for q in queries]


####################################################################
#                    ALTERNATIVE: TEXTBLOB IMPLEMENTATION
####################################################################

class SimpleQueryPreprocessor:
    """Simpler version using TextBlob (if pyspellchecker not available)"""

    def __init__(self):
        pass

    def preprocess(self, query: str) -> str:
        """Simple preprocessing using TextBlob"""
        # Expand abbreviations
        query_lower = query.lower()
        for abbrev, expansion in MEDICAL_ABBREVIATIONS.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            query_lower = re.sub(pattern, expansion, query_lower)

        # Apply medical corrections
        for misspell, correct in MEDICAL_SPELL_CORRECTIONS.items():
            query_lower = query_lower.replace(misspell, correct)

        # Spell correction with TextBlob
        try:
            blob = TextBlob(query_lower)
            query_lower = str(blob.correct())
        except:
            pass  # Skip if TextBlob fails

        return query_lower.strip()


####################################################################
#                    TESTING & EXAMPLES
####################################################################

def test_preprocessor():
    """Test the preprocessor with various examples"""

    print("\n" + "="*70)
    print("QUERY PREPROCESSOR TESTING")
    print("="*70)

    preprocessor = QueryPreprocessor(
        enable_spell_correction=True,
        enable_abbreviation_expansion=True,
        enable_synonym_expansion=False
    )

    test_queries = [
        "What is HTN?",
        "How to treat diabetis?",
        "What is the normal range for BP?",
        "Symtoms of high blood preasure",
        "What is DM treatement?",
        "How to mesure BMI?",
        "What is colesterol?",
        "Hypertention management",
    ]

    print("\nTest Queries:\n")

    for query in test_queries:
        result = preprocessor.preprocess(query, verbose=False)

        print(f"Original:  {result['original_query']}")
        print(f"Processed: {result['processed_query']}")

        if result['modifications']:
            print(f"Changes:   {', '.join(result['modifications'])}")
        print()


def compare_with_without_preprocessing():
    """Compare retrieval performance with/without preprocessing"""
    from automated_evaluation_FIXED import (
        load_vector_db, build_bm25_index_standalone,
        hybrid_search_standalone, calculate_metrics
    )
    import json

    print("\n" + "="*70)
    print("COMPARING RETRIEVAL WITH/WITHOUT PREPROCESSING")
    print("="*70)

    # Load data
    vector_db = load_vector_db("data/vector_db")
    build_bm25_index_standalone(vector_db)

    with open("data/test_queries.json", 'r') as f:
        test_queries = json.load(f)

    preprocessor = QueryPreprocessor()

    # Test sample queries
    sample_size = 10

    without_preprocessing = []
    with_preprocessing = []

    for item in test_queries[:sample_size]:
        query = item['query']
        relevant_docs = item['relevant_docs']

        # Without preprocessing
        results_without = hybrid_search_standalone(query, vector_db, alpha=0.7, k=5)
        metrics_without = calculate_metrics(results_without, relevant_docs, k=5)
        without_preprocessing.append(metrics_without['f1@k'])

        # With preprocessing
        processed = preprocessor.preprocess(query)
        results_with = hybrid_search_standalone(
            processed['processed_query'], vector_db, alpha=0.7, k=5
        )
        metrics_with = calculate_metrics(results_with, relevant_docs, k=5)
        with_preprocessing.append(metrics_with['f1@k'])

    # Calculate average
    avg_without = sum(without_preprocessing) / len(without_preprocessing)
    avg_with = sum(with_preprocessing) / len(with_preprocessing)

    print(f"\nAverage F1@5 without preprocessing: {avg_without:.4f}")
    print(f"Average F1@5 with preprocessing:    {avg_with:.4f}")
    print(f"Improvement: {((avg_with - avg_without) / avg_without * 100):.2f}%")


####################################################################
#                    MAIN
####################################################################

if __name__ == "__main__":
    # Run tests
    test_preprocessor()

    # Optionally compare performance
    # Uncomment if you have vector_db loaded
    # compare_with_without_preprocessing()
