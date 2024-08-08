# from flask import Flask, request, jsonify, render_template
# import speech_recognition as sr
# import os
# from pydub import AudioSegment
# import logging
# import spacy

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Load spaCy model
# nlp = spacy.load('en_core_web_sm')

# def extract_event_context(text):
#     event = None
#     context = None

#     # Define keywords for events
#     search_keywords = ["search", "looking for", "find", "show me"]
#     scroll_keywords = ["scroll"]
#     filter_keywords = ["filter", "show products in range"]

#     # Convert text to lowercase for easier matching
#     text = text.lower()

#     # Check for search event
#     for keyword in search_keywords:
#         if keyword in text:
#             event = "search"
#             context = text.replace(keyword, '').strip()
#             break

#     # Check for scroll event
#     if not event:
#         for keyword in scroll_keywords:
#             if keyword in text:
#                 event = "scroll"
#                 context = text.replace(keyword, '').strip()
#                 break

#     # Check for filter event
#     if not event:
#         for keyword in filter_keywords:
#             if keyword in text:
#                 event = "filter"
#                 context = text.replace(keyword, '').strip()
#                 break

#     return event, context

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/convert', methods=['POST'])
# def convert_speech_to_text():
#     if 'audio' not in request.files:
#         app.logger.error('No audio file provided')
#         return jsonify({'error': 'No audio file provided'}), 400
    
#     audio_file = request.files['audio']
#     audio_path = 'temp_audio.webm'
#     wav_path = 'temp_audio.wav'
#     audio_file.save(audio_path)
#     app.logger.debug(f'Audio file saved to {audio_path}')

#     # Convert audio to WAV format
#     try:
#         audio = AudioSegment.from_file(audio_path, format="webm")
#         audio.export(wav_path, format="wav")
#         app.logger.debug(f'Audio file converted to WAV format and saved to {wav_path}')
#     except Exception as e:
#         app.logger.error(f'Error converting audio: {str(e)}')
#         return jsonify({'error': f'Error converting audio: {str(e)}'}), 500

#     recognizer = sr.Recognizer()

#     try:
#         with sr.AudioFile(wav_path) as source:
#             audio_data = recognizer.record(source)
#             text = recognizer.recognize_google(audio_data)
#             app.logger.debug('Audio successfully transcribed')

#             # Extract event and context
#             event, context = extract_event_context(text)
#             app.logger.debug(f'Event: {event}, Context: {context}')

#             return jsonify({'text': text, 'event': event, 'context': context}), 200
#     except sr.UnknownValueError:
#         app.logger.error('Speech was unintelligible')
#         return jsonify({'error': 'Speech was unintelligible'}), 400
#     except sr.RequestError as e:
#         app.logger.error(f'Could not request results from Google Speech Recognition service; {e}')
#         return jsonify({'error': f'Could not request results from Google Speech Recognition service; {e}'}), 500
#     except Exception as e:
#         app.logger.error(f'Unexpected error: {str(e)}')
#         return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
#     finally:
#         os.remove(audio_path)
#         os.remove(wav_path)

# if __name__ == '__main__':
#     app.run(debug=True)



#hugging face
# from flask import Flask, request, jsonify, render_template
# import speech_recognition as sr
# import os
# from pydub import AudioSegment
# import logging
# from transformers import pipeline

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Load pre-trained pipelines
# intent_pipeline = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
# ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/convert', methods=['POST'])
# def convert_speech_to_text():
#     if 'audio' not in request.files:
#         app.logger.error('No audio file provided')
#         return jsonify({'error': 'No audio file provided'}), 400
    
#     audio_file = request.files['audio']
#     audio_path = 'temp_audio.webm'
#     wav_path = 'temp_audio.wav'
#     audio_file.save(audio_path)
#     app.logger.debug(f'Audio file saved to {audio_path}')

#     # Convert audio to WAV format
#     try:
#         audio = AudioSegment.from_file(audio_path, format="webm")
#         audio.export(wav_path, format="wav")
#         app.logger.debug(f'Audio file converted to WAV format and saved to {wav_path}')
#     except Exception as e:
#         app.logger.error(f'Error converting audio: {str(e)}')
#         return jsonify({'error': f'Error converting audio: {str(e)}'}), 500

#     recognizer = sr.Recognizer()

#     try:
#         with sr.AudioFile(wav_path) as source:
#             audio_data = recognizer.record(source)
#             text = recognizer.recognize_google(audio_data)
#             app.logger.debug('Audio successfully transcribed')

#             event, context = extract_event_context(text)
#             return jsonify({'text': text, 'event': event, 'context': context}), 200
#     except sr.UnknownValueError:
#         app.logger.error('Speech was unintelligible')
#         return jsonify({'error': 'Speech was unintelligible'}), 400
#     except sr.RequestError as e:
#         app.logger.error(f'Could not request results from Google Speech Recognition service; {e}')
#         return jsonify({'error': f'Could not request results from Google Speech Recognition service; {e}'}), 500
#     except Exception as e:
#         app.logger.error(f'Unexpected error: {str(e)}')
#         return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
#     finally:
#         os.remove(audio_path)
#         os.remove(wav_path)

# def extract_event_context(text):
#     # Use intent classification model to predict the event
#     intent_result = intent_pipeline(text)
#     event = intent_result[0]['label']

#     # Use NER model to extract context
#     ner_results = ner_pipeline(text)
#     context = ' '.join([result['word'] for result in ner_results if result['entity'] == 'MISC'])

#     return event, context

# if __name__ == '__main__':
#     app.run(debug=True)


# from flask import Flask, request, jsonify, render_template
# import speech_recognition as sr
# import os
# from pydub import AudioSegment
# import logging
# import spacy

# app = Flask(__name__)

# # Configure logging
# logging.basicConfig(level=logging.DEBUG)

# # Load spaCy model
# nlp = spacy.load('en_core_web_sm')

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/convert', methods=['POST'])
# def convert_speech_to_text():
#     if 'audio' not in request.files:
#         app.logger.error('No audio file provided')
#         return jsonify({'error': 'No audio file provided'}), 400
    
#     audio_file = request.files['audio']
#     audio_path = 'temp_audio.webm'
#     wav_path = 'temp_audio.wav'
#     audio_file.save(audio_path)
#     app.logger.debug(f'Audio file saved to {audio_path}')

#     # Convert audio to WAV format
#     try:
#         audio = AudioSegment.from_file(audio_path, format="webm")
#         audio.export(wav_path, format="wav")
#         app.logger.debug(f'Audio file converted to WAV format and saved to {wav_path}')
#     except Exception as e:
#         app.logger.error(f'Error converting audio: {str(e)}')
#         return jsonify({'error': f'Error converting audio: {str(e)}'}), 500

#     recognizer = sr.Recognizer()

#     try:
#         with sr.AudioFile(wav_path) as source:
#             audio_data = recognizer.record(source)
#             text = recognizer.recognize_google(audio_data)
#             app.logger.debug('Audio successfully transcribed')

#             event, context = extract_event_context(text)
#             return jsonify({'text': text, 'event': event, 'context': context}), 200
#     except sr.UnknownValueError:
#         app.logger.error('Speech was unintelligible')
#         return jsonify({'error': 'Speech was unintelligible'}), 400
#     except sr.RequestError as e:
#         app.logger.error(f'Could not request results from Google Speech Recognition service; {e}')
#         return jsonify({'error': f'Could not request results from Google Speech Recognition service; {e}'}), 500
#     except Exception as e:
#         app.logger.error(f'Unexpected error: {str(e)}')
#         return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
#     finally:
#         os.remove(audio_path)
#         os.remove(wav_path)

# def extract_event_context(text):
#     doc = nlp(text)
#     # Define keywords for events
#     events_keywords = {
#         "search": ["search", "find", "look for","looking","looking for","find", "show me"],
#         "scroll": ["scroll", "move down", "move up"],
#         "filter": ["filter", "show only", "limit to"]
#     }
    
#     event = None
#     context = []

#     # Identify event based on keywords
#     for token in doc:
#         for event_key, keywords in events_keywords.items():
#             if token.lemma_ in keywords:
#                 event = event_key
#                 break
#         if event:
#             break

#     # Extract context (everything except the event keyword)
#     if event:
#         context = [token.text for token in doc if token.lemma_ not in events_keywords[event]]
    
#     return event, ' '.join(context)

# # if __name__ == '__main__':
# #     app.run(debug=True)
# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)


# from transformers import pipeline

# # Load a pre-trained model for text classification
# model_name = "textattack/roberta-base-rotten-tomatoes"
# nlp = pipeline("text-classification", model=model_name)

# text = "i am looking new iPhone"
# result = nlp(text)
# print(result)
import os
from dotenv import load_dotenv
load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
from pydub import AudioSegment
import logging
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set up Hugging Face API token and model
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=hf_token)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
def convert_speech_to_text():
    if 'audio' not in request.files:
        app.logger.error('No audio file provided')
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_path = 'temp_audio.webm'
    wav_path = 'temp_audio.wav'
    audio_file.save(audio_path)
    app.logger.debug(f'Audio file saved to {audio_path}')

    # Convert audio to WAV format
    try:
        audio = AudioSegment.from_file(audio_path, format="webm")
        audio.export(wav_path, format="wav")
        app.logger.debug(f'Audio file converted to WAV format and saved to {wav_path}')
    except Exception as e:
        app.logger.error(f'Error converting audio: {str(e)}')
        return jsonify({'error': f'Error converting audio: {str(e)}'}), 500

    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(wav_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
            app.logger.debug('Audio successfully transcribed')

            # Extract event and context using Hugging Face model
            query = f"find event(like search, filter, scroll, back) and context(user want) of utterance '{text}'"
            response = llm.invoke(query)
            app.logger.debug(f'Hugging Face model response: {response}')

            # Parse the response (assuming the response format is as expected)
            event_context = response.strip().split("context:")
            event = event_context[0].replace("event:", "").strip()
            context = event_context[1].strip() if len(event_context) > 1 else ""

            return jsonify({'text': text, 'event': event, 'context': context}), 200
    except sr.UnknownValueError:
        app.logger.error('Speech was unintelligible')
        return jsonify({'error': 'Speech was unintelligible'}), 400
    except sr.RequestError as e:
        app.logger.error(f'Could not request results from Google Speech Recognition service; {e}')
        return jsonify({'error': f'Could not request results from Google Speech Recognition service; {e}'}), 500
    except Exception as e:
        app.logger.error(f'Unexpected error: {str(e)}')
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500
    finally:
        os.remove(audio_path)
        os.remove(wav_path)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
