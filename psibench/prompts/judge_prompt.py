"""Judge prompts for different evaluation purposes."""

from langchain_core.prompts import ChatPromptTemplate

def create_ptc_judge_conversation_prompt() -> ChatPromptTemplate:
    """Create prompt for PTC classification of entire conversation at once.
    
    Returns:
        ChatPromptTemplate for classifying all patient turns in a conversation
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant performing a linguistic analysis task for a research project on therapeutic conversations. 
This is for academic research, and the content may discuss mental health challenges.
Your task is to classify EACH patient turn from a therapy session into one of four categories:

**P (Problem)**: The patient is expressing:
- Confusion, distress, or emotional pain
- Feeling stuck or helpless
- Describing problems without insight
- Negative emotions without perspective
- Complaints or struggles

**T (Transition)**: The patient is showing:
- Beginning to reflect on their situation
- Gaining some perspective
- Starting to consider alternatives
- Expressing curiosity or questioning
- Moving from pure distress to thoughtful consideration

**C (Change)**: The patient is demonstrating:
- Emotional resolution or acceptance
- Reframing their situation positively
- New insights or understanding
- Active problem-solving or planning
- Hope, empowerment, or growth mindset
- Clear perspective shift from the problem

**F (Filler)**: The response is filler:
- Contains no meaningful therapeutic content and does not fit into P, T, or C categories
- Is small talk or neutral procedural social responses

Analyze each patient turn carefully, considering both the content and emotional tone.

Note: The conversation follows a strict turn-taking structure between the patient and the therapist — the patient never speaks twice in a row.
Each patient message always begins with "PATIENT:" and may contain newline characters (\n or \n\n) for formatting. \
    Do not treat newline characters as separate messages; they are part of the same patient message.

You must classify EVERY patient turn, including greetings, in the conversation.

You must respond with ONLY a JSON array. Each element should have:
- "content": the first ten words of the patient's message (truncated and add [...] if longer than ten words)
- "classification": one of P, T, C, or F

Output example:
```json
    [
    {{
        "content": "Hello.",
        "classification": "F"
    }},
    {{
        "content": "Hi \n\n I feel so lost.\n\n Everything seems overwhelming [...]",
        "classification": "P"
    }},
    {{
        "content": "But maybe if I try to think differently, I can [...]",
        "classification": "T"
    }}
    ]
```json
Do not include any explanation or additional text outside the JSON array."""),
        ("user", """Here is the complete conversation:

{conversation}

Classify each patient turn and return as JSON array:""")
    ])

def create_ptc_judge_single_turn_prompt() -> ChatPromptTemplate:
    """Create prompt for PTC classification of a single patient turn with history context.
    
    Returns:
        ChatPromptTemplate for classifying one patient message based on conversation history
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant performing a linguistic analysis task for a research project on therapeutic conversations. 
This is for academic research, and the content may discuss mental health challenges.
Your task is to classify a SINGLE patient message from a therapy session into one of four categories:

**P (Problem)**: The patient is expressing:
- Confusion, distress, or emotional pain
- Feeling stuck or helpless
- Describing problems without insight
- Negative emotions without perspective
- Complaints or struggles

**T (Transition)**: The patient is showing:
- Beginning to reflect on their situation
- Gaining some perspective
- Starting to consider alternatives
- Expressing curiosity or questioning
- Moving from pure distress to thoughtful consideration

**C (Change)**: The patient is demonstrating:
- Emotional resolution or acceptance
- Reframing their situation positively
- New insights or understanding
- Active problem-solving or planning
- Hope, empowerment, or growth mindset
- Clear perspective shift from the problem

**F (Filler)**: The response is filler:
- Contains no meaningful therapeutic content and does not fit into P, T, or C categories
- Is small talk or neutral procedural social responses

Analyze the current patient message carefully, considering both the content and emotional tone, as well as the conversation history provided.

You must respond with ONLY a single letter: P, T, C, or F.
Do not include any explanation or additional text."""),
        ("user", """Previous conversation history:
{history}

Current patient message to classify:
{current_message}

Classification:""")
    ])

def create_valence_judge_prompt() -> ChatPromptTemplate:
    """Create prompt for valence classification (positive / negative / neutral).

    Returns:
        ChatPromptTemplate for classifying a patient message into one of 3 valence categories
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant performing emotional valence analysis on therapeutic conversations.
    This is for academic research, and the content may discuss mental health challenges.

    Your task is to classify a SINGLE patient message according to its PRIMARY emotional valence:

    **Valence categories:**
    - positive
    - negative
    - neutral: the patient expresses no clear positive or negative affect

    Analyze the current patient message carefully, considering both the content and emotional tone, as well as the conversation history provided for context.
    The conversation history is used as context to help you understand the current message, but your classification should be based primarily on the current patient message.
    You must respond with ONLY one valence label from the list above (lowercase).
    Do not include any explanation or additional text."""),
        ("user", """Previous conversation history:
    {history}

    Current patient message to classify:
    {current_message}

Valence:""")
    ])


def create_emotion_judge_prompt() -> ChatPromptTemplate:
    """Create prompt for emotion classification based on Plutchik's 8 basic emotions.
    
    Returns:
        ChatPromptTemplate for classifying a patient message into one of 9 emotion categories
    """
    return ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant performing emotional analysis on therapeutic conversations. 
    This is for academic research, and the content may discuss mental health challenges.

    Your task is to classify a SINGLE patient message according to the PRIMARY emotion expressed, using Plutchik's 8 basic emotions plus a neutral category:

    **Emotions:**
    - anger
    - disgust
    - fear
    - joy
    - sadness
    - surprise
    - anticipation
    - trust
    - neutral (when no clear emotion is expressed)

    Analyze the current patient message carefully, considering both the content and emotional tone, as well as the conversation history provided for context.
    The conversation history is used as context to help you understand the current message, but your classification should be based primarily on the current patient message.
    You must respond with ONLY one emotion label from the list above (lowercase).
    Do not include any explanation or additional text."""),
            ("user", """Previous conversation history:
    {history}

    Current patient message to classify:
    {current_message}

Emotion:""")
    ])
    