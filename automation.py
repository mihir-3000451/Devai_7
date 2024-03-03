import os
import json
import stanza
import streamlit as st

# Initialize CoreNLP
try:
    nlp = stanza.Pipeline()
except Exception as e:
    st.error("Error initializing Stanza pipeline: {}".format(e))
    st.stop()

# Function to process text data for various CoreNLP features
def process_text_segments(text, actions):
    try:
        doc = nlp(text)
        segment_annotations = []

        for action in actions:
            action_annotations = []

            if action == 'Sentence Boundaries':
                for sent in doc.sentences:
                    action_annotations.append({
                        "id": f"{sent.start_char}_{sent.end_char}",
                        "from_name": "label",
                        "to_name": "text",
                        "type": "labels",
                        "value": {
                            "start": sent.start_char,
                            "end": sent.end_char,
                            "text": sent.text,
                            "labels": ["Sentence"]
                        }
                    })
            elif action == 'Parts of Speech':
                for sent in doc.sentences:
                    for word in sent.words:
                        action_annotations.append({
                            "id": f"{word.start_char}_{word.end_char}",
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                            "value": {
                                "start": word.start_char,
                                "end": word.end_char,
                                "text": word.text,
                                "labels": [word.upos]
                            }
                        })
            elif action == 'Named Entities':
                for sent in doc.sentences:
                    for ent in sent.ents:
                        action_annotations.append({
                            "id": f"{ent.start_char}_{ent.end_char}",
                            "from_name": "label",
                            "to_name": "text",
                            "type": "labels",
                            "value": {
                                "start": ent.start_char,
                                "end": ent.end_char,
                                "text": ent.text,
                                "labels": [ent.type]
                            }
                        })

            segment_annotations.append({
                "model_version": "one",
                "score": 0.5,  # Dummy score, you may adjust this as per your requirement
                "result": action_annotations
            })

        return {
            "data": {
                "text": text
            },
            "predictions": segment_annotations
        }
    except Exception as e:
        st.error("Error processing text segment: {}".format(e))
        return {
            "data": {
                "text": text
            },
            "predictions": []
        }

# Streamlit UI
st.title("CoreNLP Text Processing")
text_files = st.file_uploader("Upload text files", type=["txt"], accept_multiple_files=True)
output_folder_path = st.text_input("Enter the output folder path:", value="/home/gray/Downloads/Devai/Automation_json")

if text_files is not None:
    try:
        st.write(f"Output JSON files will be saved in folder: {output_folder_path}")

        for text_file in text_files:
            try:
                text = text_file.read().decode("utf-8")
                st.text("Text content of the uploaded file:")
                st.text(text)

                actions = st.multiselect("Select Actions", ["Sentence Boundaries", "Parts of Speech", "Named Entities"])

                if st.button("Process Text"):
                    segments = text.split('.')
                    annotations = [process_text_segments(seg.strip() + '.', actions) for seg in segments if seg.strip()]

                    # Save JSON file in provided folder path
                    if output_folder_path:
                        try:
                            output_file = os.path.join(output_folder_path, f'automation_{len(os.listdir(output_folder_path)) + 1}.json')
                            with open(output_file, 'w') as f:
                                json.dump(annotations, f, indent=4)
                            st.success(f"Annotations saved to {output_file}")
                        except Exception as e:
                            st.error("Error saving JSON file: {}".format(e))
                    else:
                        st.error("Please provide the output folder path.")
            except Exception as e:
                st.error("An error occurred while processing the text file: {}".format(e))
    except Exception as e:
        st.error("An error occurred while handling the uploaded text files: {}".format(e))
