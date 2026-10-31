"""
TODO: 
1. Extract cognitive model from real conversation. Adapt code from here:
https://github.com/ruiyiw/patient-psi/tree/main/python/generation 
2. Parse that cognitive model to Patient-Psi prompt.
"""

from langchain.output_parsers import PydanticOutputParser
from langchain_litellm import ChatLiteLLM

from psibench.agents.base import _SHARED_HTTP_HANDLER

from psibench.models.generation_template import GenerationModel
from dotenv import load_dotenv, find_dotenv
import os
import json
import argparse
import logging
from json_repair import repair_json
from litellm import batch_completion

load_dotenv(find_dotenv())

# # data_path = os.path.join(os.path.dirname(
# #     os.path.abspath('.env')), os.getenv('DATA_PATH'))
# out_path = os.path.join(os.path.dirname(
#     os.path.abspath('.env')), os.getenv('OUT_PATH'))
# --- Setup directories from .env ---
# --- Setup directories from .env ---
# Always anchor to the repo root (2 levels above this file)
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
data_path = os.path.join(repo_root, os.getenv("DATA_PATH", "data"))
out_path = os.path.join(repo_root, os.getenv("OUT_PATH", "output"))
os.makedirs(out_path, exist_ok=True)



# # --- Setup directories from .env ---
# data_path = os.getenv("DATA_PATH", "./data")
# out_path = os.getenv("OUT_PATH", "./output")
# os.makedirs(out_path, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List, Dict, Any
import pandas as pd

PATIENT_STYLE_DESCRIPTIONS = {
    "upset": (
        "An upset patient may 1) exhibit anger or resistance toward therapy, 2) challenge or dismiss the "
        "therapist's suggestions, 3) struggle to trust the therapist, and 4) argue, criticize, or express "
        "frustration during sessions."
    ),
    "verbose": (
        "A verbose patient may 1) provide highly detailed responses, 2) elaborate extensively on personal "
        "experiences, thoughts, and feelings, and 3) find it difficult to let the therapist guide the "
        "conversation."
    ),
    "reserved": (
        "A reserved patient may 1) answer briefly or evasively, 2) hesitate to share personal information "
        "or feelings, 3) need extra prompting to open up, and 4) express distrust or skepticism toward the "
        "therapist."
    ),
    "tangent": (
        "A tangent-prone patient may 1) begin answering but quickly drift into unrelated topics, 2) share "
        "anecdotes that are irrelevant to the question, 3) struggle to stay focused, and 4) require "
        "redirection to return to the main points."
    ),
    "pleasing": (
        "A pleasing patient may 1) downplay concerns to maintain a positive image, 2) act eager to please "
        "and avoid disagreement, 3) seek approval or validation from the therapist, and 4) readily agree "
        "with suggestions even if they are unsure."
    ),
    "plain": "",
}


def load_conversation(messages) -> str:
    """
    Format one full conversation (list of {role, content}) into Therapist/Client lines.
    Maps: assistant -> Therapist, user -> Client.
    """
    lines = []
    # print(f"Started - Type of messages: {type(messages)}")
    
    # Handle pandas Series
    if hasattr(messages, 'tolist'):
        # Convert Series to list
        messages_list = messages.tolist()
    elif hasattr(messages, 'values'):
        # Get values from Series
        messages_list = messages.values.tolist()
    elif isinstance(messages, list):
        messages_list = messages
    else:
        # Try to iterate directly
        messages_list = list(messages)
    
    # print(f"Converted to list with {len(messages_list)} items")
    
    for i, turn in enumerate(messages_list):
        # print(f"Processing turn {i}, type: {type(turn)}")
        
        # Handle different data structures for turn
        if isinstance(turn, dict):
            role = str(turn.get("role", "")).lower().strip()
            content = str(turn.get("content", "")).strip().replace("\n", " ")
        elif isinstance(turn, str):
            # If it's a string, try to parse it as JSON
            try:
                import json
                turn_dict = json.loads(turn)
                role = str(turn_dict.get("role", "")).lower().strip()
                content = str(turn_dict.get("content", "")).strip().replace("\n", " ")
            except:
                print(f"Could not parse string as JSON: {turn[:100]}")
                continue
        else:
            print(f"Unexpected turn format: {type(turn)}")
            print(f"Turn content: {turn}")
            continue
            
        if role == "user":
            lines.append(f"Therapist: {content}")
        elif role == "assistant":
            lines.append(f"Client: {content}")
        else:
            lines.append(f"{role.capitalize() or 'Unknown'}: {content}")
    
    # print("load_conversation completed")
    return "\n".join(lines)

# def load_conversation(data, conv_number):
#     # Inputs real conversational data - outputs the chosen conversation
#     # Remove conv_number logic and return conversation_texts if all conversations are desired
#     conversation_texts = []  # will store long strings

#     for convo in data:
#         lines = []
#         for turn in convo["dialog"]:
#             role = turn["speaker"]
#             content = turn["content"].strip().replace("\n", " ")

#             if role == "supporter":
#                 lines.append(f"Therapist: {content}")
#             elif role == "seeker":
#                 lines.append(f"Client: {content}")
#             else:
#                 lines.append(f"{role.capitalize()}: {content}")

#         # join all turns into one big string for the conversation
#         convo_text = "\n".join(lines)
#         conversation_texts.append(convo_text)
#     chosen_convo = conversation_texts[conv_number]
#     return chosen_convo


def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except (TypeError, OverflowError):
        return False

def format_patient_psi_prompt_from_ccd(
    ccd: dict,
    patient_type_content: str,
    name: str = "Patient",   # pass the real name if you have it
) -> str:
    """
    Build the official Patient-Ψ prompt from your CCD schema.
    Expects keys like:
      - life_history, core_beliefs, core_belief_description,
        intermediate_beliefs, intermediate_beliefs_during_depression,
        coping_strategies, cognitive_models: [ {situation, automatic_thoughts, emotion, behavior} ]
    """
    # map CCD → prompt fields (with safe fallbacks)
    history = ccd.get("life_history", "")
    core_belief = ccd.get("core_beliefs", "") or ccd.get("core_belief_description", "")
    interm = ccd.get("intermediate_beliefs", "")
    interm_dep = ccd.get("intermediate_beliefs_during_depression", "")
    coping = ccd.get("coping_strategies", "")

    # pull the first cognitive model item (if present)
    cm = (ccd.get("cognitive_models") or [{}])[0]
    situation = cm.get("situation", "")
    auto_thoughts = cm.get("automatic_thoughts", "")
    emotion = cm.get("emotion", "")
    behavior = cm.get("behavior", "")

    patient_style = (patient_type_content or "plain").strip()
    style_description = PATIENT_STYLE_DESCRIPTIONS.get(patient_style.lower())
    style_instruction = f"1. {patient_style}"
    if style_description:
        style_instruction += f"\n - {style_description}"

    prompt = (
        f"Imagine you are {name}, a patient who has been experiencing mental health challenges. "
        f"You have been attending therapy sessions for several weeks. Your task is to engage in a conversation with the therapist as {name} would during a cognitive behavioral therapy (CBT) session. "
        f"Align your responses with {name}'s background information provided in the 'Relevant history' section. Your thought process should be guided by the cognitive conceptualization diagram in the 'Cognitive Conceptualization Diagram' section, "
        f"but avoid directly referencing the diagram as a real patient would not explicitly think in those terms.\n\n"
        f"Patient History: {history}\n\n"
        f"Cognitive Conceptualization Diagram:\n"
        f"Core Beliefs: {core_belief}\n"
        f"Intermediate Beliefs: {interm}\n"
        f"Intermediate Beliefs during Depression: {interm_dep}\n"
        f"Coping Strategies: {coping}\n\n"
        f"You will be asked about your experiences over the past week. Engage in a conversation with the therapist regarding the following situation and behavior. "
        f"Use the provided emotions and automatic thoughts as a reference, but do not disclose the cognitive conceptualization diagram directly. "
        f"Instead, allow your responses to be informed by the diagram, enabling the therapist to infer your thought processes.\n\n"
        f"Situation: {situation}\n"
        f"Automatic Thoughts: {auto_thoughts}\n"
        f"Emotions: {emotion}\n"
        f"Behavior: {behavior}\n\n"
        f"In the upcoming conversation, you will simulate {name} during the therapy session, while the user will play the role of the therapist. Adhere to the following guidelines:\n"
        f"{style_instruction}\n"
        f"2. Emulate the demeanor and responses of a genuine patient to ensure authenticity in your interactions. "
        f"Use natural language, including hesitations, pauses, and emotional expressions, to enhance the realism of your responses.\n"
        f"3. Gradually reveal deeper concerns and core issues, as a real patient often requires extensive dialogue before delving into more sensitive topics. "
        f"This gradual revelation creates challenges for therapists in identifying the patient's true thoughts and emotions.\n"
        f"4. Maintain consistency with {name}'s profile throughout the conversation. Ensure that your responses align with the provided background information, cognitive conceptualization diagram, and the specific situation, thoughts, emotions, and behaviors described.\n"
        f"5. Engage in a dynamic and interactive conversation with the therapist. Respond to their questions and prompts in a way that feels authentic and true to {name}'s character. "
        f"Allow the conversation to flow naturally, and avoid providing abrupt or disconnected responses.\n\n"
        f"You are now {name}. Respond to the therapist's prompts as {name} would, regardless of the specific questions asked. "
        f"Limit each of your responses to a maximum of 5 sentences. If the therapist begins the conversation with a greeting like \"Hi,\" initiate the conversation as the patient."
    )
    return prompt

def generate_chain_batch(data_list, config):
    """Generate cognitive models for multiple conversations in batch.
    
    Args:
        data_list: List of conversation data (messages)
        config: Configuration dictionary
        
    Returns:
        List of tuples (ccd_dict, psi_prompt_string) or (None, None) on error
    """
    patient = config.get('patient')
    
    # Get API settings
    if patient.get("api_base"):
        api_base = patient.get("api_base")
        api_key = "sk-no-key-required"
    else:
        api_base = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Prepare all prompts
    print(f"Preparing {len(data_list)} prompts for cognitive model extraction...")
    pydantic_parser = PydanticOutputParser(
        pydantic_object=GenerationModel.CognitiveConceptualizationDiagram)
    
    all_messages = []
    for data in data_list:
        lines = load_conversation(data)
        query = "Based on the therapy session transcript, summarize the patient's personal history following the below instructions. Not that `Client` means the patient in the transcript.\n\n{lines}".format(
            lines=lines)
        
        # Format prompt messages synchronously
        prompt_messages = GenerationModel.prompt_template.format_messages(
            query=query,
            format_instructions=pydantic_parser.get_format_instructions()
        )
        
        # Convert to litellm format
        litellm_messages = []
        for i, msg in enumerate(prompt_messages):
            if hasattr(msg, 'type'):
                role = msg.type
                if role == 'human':
                    role = 'user'
                elif role not in ['system', 'assistant', 'user', 'function', 'tool', 'developer']:
                    role = 'system' if i == 0 else 'user'
            else:
                role = 'system' if i == 0 else 'user'
            
            litellm_messages.append({
                "role": role,
                "content": msg.content
            })
        all_messages.append(litellm_messages)
    
    print(f"All prompts prepared. Calling batch_completion with {len(all_messages)} requests...")
    # Batch completion call
    logger.info(f"Generating {len(all_messages)} cognitive models in batch...")
    try:
        responses = batch_completion(
            model=patient.get("model"),
            messages=all_messages,
            temperature=patient.get("temperature"),
            api_key=api_key,
            api_base=api_base,
        )
    except Exception as e:
        logger.error(f"Error in batch_completion: {e}")
        return [None] * len(data_list)
    
    print(f"Batch completion finished. Parsing {len(responses)} responses...")
    # Parse all responses
    results = []
    for i, response in enumerate(responses):
        try:
            # Handle different response formats
            if isinstance(response, str):
                response_text = response.strip()
            elif hasattr(response, 'choices') and len(getattr(response, 'choices', [])) > 0:
                content = response.choices[0].message.content
                if content is None:
                    logger.warning(f"Batch {i}: Response content is None")
                    results.append(None)
                    continue
                response_text = content.strip()
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                content = response.message.content
                if content is None:
                    logger.warning(f"Batch {i}: Response content is None")
                    results.append(None)
                    continue
                response_text = content.strip()
            elif hasattr(response, 'content'):
                content = response.content
                if content is None:
                    logger.warning(f"Batch {i}: Response content is None")
                    results.append(None)
                    continue
                response_text = content.strip()
            else:
                logger.warning(f"Batch {i}: Unexpected response format: {type(response)}")
                results.append(None)
                continue
            
            # Clean and parse JSON
            _output = repair_json(response_text, return_objects=True)
            
            if is_json_serializable(_output):
                patient_name = "Patient"
                psi_prompt = format_patient_psi_prompt_from_ccd(
                    ccd=_output,
                    patient_type_content=config.get('patient').get('conversation_type'),
                    name=patient_name
                )
                results.append((_output, psi_prompt))
            else:
                logger.warning(f"Batch {i}: Output not JSON serializable")
                results.append((None, None))
                
        except Exception as e:
            logger.error(f"Error parsing batch response {i}: {e}")
            results.append((None, None))
    
    successful = sum(1 for r in results if r is not None)
    print(f"Profile generation complete: {successful}/{len(results)} successful")
    return results

async def generate_chain(data, config):
    # with open(os.path.join(data_path, transcript_file), 'r') as f:
    #     lines = f.readlines()
    # --- Load JSON ---
    # with open("/u/adeora/PSI-bench/data/real/ESConv.json", "r") as f:
    #     data = json.load(f)
   
    # with open(os.path.join(data_path, "yoyoy"), "r") as f:
    #     data = json.load(f)
    # print("Started with open(os.path.join(data_pa")


    lines = load_conversation(data)
    # print(lines)

    query = "Based on the therapy session transcript, summarize the patient's personal history following the below instructions. Not that `Client` means the patient in the transcript.\n\n{lines}".format(
        lines=lines)

    pydantic_parser = PydanticOutputParser(
        pydantic_object=GenerationModel.CognitiveConceptualizationDiagram)

    _input = await GenerationModel.prompt_template.ainvoke({
        "query": query,
        "format_instructions": pydantic_parser.get_format_instructions()
    })
    patient = config.get('patient')
    max_attempts = patient.get('max_attempts')
    if patient.get("api_base"):
        api_base = patient.get("api_base")
        api_key = "sk-no-key-required"
    else:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        api_base = os.getenv("OPENAI_BASE_URL")
        api_key = os.getenv("OPENAI_API_KEY")
    
    # Initialize LLM
    llm = ChatLiteLLM(
        model=patient.get("model"),
        api_key=api_key,
        api_base=api_base,
        temperature=patient.get("temperature"),
        model_kwargs={"client": _SHARED_HTTP_HANDLER},
    )
    attempts = 0

    while attempts < max_attempts:
        try:
            response_content = llm.invoke(_input).content
            _output = pydantic_parser.parse(response_content).model_dump()
        except Exception as e:
            attempts += 1
            logger.warning(
                f"Failed to parse CCD output. Attempt {attempts}/{max_attempts}: {e}"
            )
            if attempts == max_attempts:
                raise ValueError(
                    "Could not parse CCD output after maximum attempts"
                ) from e
            continue
        # print(_output)

        if is_json_serializable(_output):
            # Default patient name
            patient_name = f"Patient"

            # # --- Determine output filenames ---
            # base_name = f"{patient_name}_CCD"
            # out_json_path = os.path.join(out_path, f"{base_name}.json")
            # out_prompt_path = os.path.join(out_path, f"{patient_name}_prompt.txt")

            # # --- Save CCD JSON (structured) ---
            # with open(out_json_path, "w") as f:
            #     json.dump(_output, f, indent=4)
            # logger.info(f"✅ CCD output saved to {out_json_path}")

            psi_prompt = format_patient_psi_prompt_from_ccd(
                ccd = _output,
                patient_type_content = config.get('patient').get('conversation_type'),
                name=patient_name
            )

            # # --- Save Patient-Psi prompt text ---
            # with open(out_prompt_path, "w") as f:
            #     f.write(psi_prompt)
            # logger.info(f"✅ Patient-Ψ prompt saved to {out_prompt_path}")

            return psi_prompt
            break
        else:
            attempts += 1
            logger.warning(
                f"Output is not JSON serializable. Attempting {attempts}/{max_attempts}")
            if attempts == max_attempts:
                logger.error(
                    "Max attempts reached. Could not generate a JSON serializable output.")
                raise ValueError(
                    "Could not generate a JSON serializable output after maximum attempts.")
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--transcript-file', type=str,
                        default="example_transcript.txt")
    parser.add_argument('--conv-number', type=int, default=0,
                    help="Conversation index in the dataset")
    args = parser.parse_args()
    generate_chain(args.transcript_file)


if __name__ == "__main__":
    main()
