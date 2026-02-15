"""
Feedback Collection Component for CHO Training RAG System
Collects user feedback during pilot testing

Author: Sharath Kumar MD
Date: February 2026
"""

import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path

# Feedback storage path
FEEDBACK_DIR = Path("data/feedback")
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "pilot_feedback.json"


def load_feedback():
    """Load existing feedback from file"""
    if FEEDBACK_FILE.exists():
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []


def save_feedback(feedback_list):
    """Save feedback to file"""
    with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
        json.dump(feedback_list, f, indent=2, ensure_ascii=False)


def add_feedback(feedback_entry):
    """Add a new feedback entry"""
    feedback_list = load_feedback()
    feedback_list.append(feedback_entry)
    save_feedback(feedback_list)
    return len(feedback_list)


def render_quick_feedback(query: str, answer: str):
    """
    Render quick feedback buttons after each response
    Call this after displaying an answer in the chat
    """
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("Helpful", key=f"helpful_{hash(query)}"):
            add_feedback({
                'timestamp': datetime.now().isoformat(),
                'type': 'quick',
                'rating': 'helpful',
                'query': query,
                'answer_preview': answer[:200]
            })
            st.success("Thanks for your feedback!")

    with col2:
        if st.button("Not Helpful", key=f"not_helpful_{hash(query)}"):
            add_feedback({
                'timestamp': datetime.now().isoformat(),
                'type': 'quick',
                'rating': 'not_helpful',
                'query': query,
                'answer_preview': answer[:200]
            })
            st.warning("Thanks! We'll improve.")

    with col3:
        if st.button("Incorrect", key=f"incorrect_{hash(query)}"):
            add_feedback({
                'timestamp': datetime.now().isoformat(),
                'type': 'quick',
                'rating': 'incorrect',
                'query': query,
                'answer_preview': answer[:200]
            })
            st.error("Noted. We'll review this.")


def render_feedback_form():
    """
    Render detailed feedback form in sidebar
    Call this in the sidebar section of your app
    """
    st.subheader("Pilot Feedback")

    with st.form("feedback_form", clear_on_submit=True):
        # User info (optional)
        user_name = st.text_input(
            "Your Name (optional)",
            placeholder="e.g., CHO Pune District"
        )

        # Overall rating
        rating = st.select_slider(
            "How useful is this system?",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {
                1: "1 - Not useful",
                2: "2 - Slightly useful",
                3: "3 - Moderately useful",
                4: "4 - Very useful",
                5: "5 - Extremely useful"
            }[x]
        )

        # Accuracy rating
        accuracy = st.select_slider(
            "How accurate are the answers?",
            options=[1, 2, 3, 4, 5],
            value=3,
            format_func=lambda x: {
                1: "1 - Often incorrect",
                2: "2 - Sometimes incorrect",
                3: "3 - Mostly accurate",
                4: "4 - Very accurate",
                5: "5 - Always accurate"
            }[x]
        )

        # Language preference
        language_needed = st.multiselect(
            "Which languages do you need?",
            options=["English", "Marathi", "Hindi", "Other"],
            default=["English"]
        )

        # Feature requests
        features_needed = st.multiselect(
            "What features would help you?",
            options=[
                "Voice input",
                "Offline mode",
                "Mobile app",
                "More detailed answers",
                "Shorter answers",
                "More sources/references",
                "Drug dosage calculator",
                "Emergency protocols quick access"
            ]
        )

        # Open feedback
        comments = st.text_area(
            "Additional comments or suggestions",
            placeholder="Tell us what works well and what needs improvement..."
        )

        # Specific issues
        issues = st.text_area(
            "Any incorrect or missing information?",
            placeholder="Please describe any errors you noticed..."
        )

        submitted = st.form_submit_button("Submit Feedback")

        if submitted:
            feedback_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': 'detailed',
                'user_name': user_name,
                'rating': rating,
                'accuracy': accuracy,
                'language_needed': language_needed,
                'features_needed': features_needed,
                'comments': comments,
                'issues': issues
            }

            count = add_feedback(feedback_entry)
            st.success(f"Thank you! Feedback #{count} recorded.")


def render_feedback_stats():
    """
    Render feedback statistics (for admin/researcher view)
    """
    feedback = load_feedback()

    if not feedback:
        st.info("No feedback collected yet.")
        return

    st.subheader(f"Feedback Summary ({len(feedback)} responses)")

    # Quick feedback stats
    quick_feedback = [f for f in feedback if f.get('type') == 'quick']
    if quick_feedback:
        helpful = len([f for f in quick_feedback if f.get('rating') == 'helpful'])
        not_helpful = len([f for f in quick_feedback if f.get('rating') == 'not_helpful'])
        incorrect = len([f for f in quick_feedback if f.get('rating') == 'incorrect'])

        col1, col2, col3 = st.columns(3)
        col1.metric("Helpful", helpful)
        col2.metric("Not Helpful", not_helpful)
        col3.metric("Incorrect", incorrect)

    # Detailed feedback stats
    detailed_feedback = [f for f in feedback if f.get('type') == 'detailed']
    if detailed_feedback:
        avg_rating = sum(f.get('rating', 3) for f in detailed_feedback) / len(detailed_feedback)
        avg_accuracy = sum(f.get('accuracy', 3) for f in detailed_feedback) / len(detailed_feedback)

        st.metric("Average Usefulness", f"{avg_rating:.1f}/5")
        st.metric("Average Accuracy", f"{avg_accuracy:.1f}/5")

        # Language needs
        all_languages = []
        for f in detailed_feedback:
            all_languages.extend(f.get('language_needed', []))

        if all_languages:
            st.write("**Language Needs:**")
            for lang in set(all_languages):
                count = all_languages.count(lang)
                st.write(f"- {lang}: {count} requests")

        # Feature requests
        all_features = []
        for f in detailed_feedback:
            all_features.extend(f.get('features_needed', []))

        if all_features:
            st.write("**Feature Requests:**")
            for feat in set(all_features):
                count = all_features.count(feat)
                st.write(f"- {feat}: {count} requests")


# Google Form template for external feedback
GOOGLE_FORM_TEMPLATE = """
## Google Form Template for CHO RAG Pilot Feedback

Create a Google Form with these questions:

### Section 1: Basic Information
1. Your Name/ID (optional) - Short answer
2. District/Location - Short answer
3. Date of feedback - Date

### Section 2: System Usability
4. How useful is the CHO RAG system for your daily work?
   - Scale: 1-5 (1 = Not useful, 5 = Extremely useful)

5. How accurate are the answers provided?
   - Scale: 1-5 (1 = Often incorrect, 5 = Always accurate)

6. How easy is the system to use?
   - Scale: 1-5 (1 = Very difficult, 5 = Very easy)

### Section 3: Language & Features
7. Which languages do you need? (Multiple choice)
   - English
   - Marathi
   - Hindi
   - Other (specify)

8. What features would be most helpful? (Multiple choice)
   - Voice input
   - Offline mode
   - Mobile app
   - More detailed answers
   - Shorter answers
   - Drug dosage calculator
   - Emergency protocols quick access

### Section 4: Specific Feedback
9. What topics does the system answer well? - Long answer

10. What topics need improvement? - Long answer

11. Have you found any incorrect information? If yes, please describe. - Long answer

12. Any other suggestions for improvement? - Long answer

### Form Settings:
- Allow response editing
- Collect email addresses (optional)
- Send confirmation email
"""


def get_google_form_template():
    """Return the Google Form template"""
    return GOOGLE_FORM_TEMPLATE


if __name__ == "__main__":
    # Test the feedback component
    st.set_page_config(page_title="Feedback Test", layout="wide")
    st.title("Feedback Component Test")

    # Sidebar feedback form
    with st.sidebar:
        render_feedback_form()

    # Main area - quick feedback test
    st.subheader("Quick Feedback Test")
    render_quick_feedback("What is hypertension?", "Hypertension is high blood pressure...")

    # Stats
    st.divider()
    render_feedback_stats()

    # Google Form template
    st.divider()
    st.subheader("Google Form Template")
    st.code(GOOGLE_FORM_TEMPLATE, language="markdown")
