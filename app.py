from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
import os
from pydub import AudioSegment
import logging
import spacy

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

def extract_event_context(text):
    event = None
    context = None

    # Define keywords for events
    search_keywords = ["search", "looking for", "find", "show me"]
    scroll_keywords = ["scroll"]
    filter_keywords = ["filter", "show products in range"]

    # Convert text to lowercase for easier matching
    text = text.lower()

    # Check for search event
    for keyword in search_keywords:
        if keyword in text:
            event = "search"
            context = text.replace(keyword, '').strip()
            break

    # Check for scroll event
    if not event:
        for keyword in scroll_keywords:
            if keyword in text:
                event = "scroll"
                context = text.replace(keyword, '').strip()
                break

    # Check for filter event
    if not event:
        for keyword in filter_keywords:
            if keyword in text:
                event = "filter"
                context = text.replace(keyword, '').strip()
                break

    return event, context

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

            # Extract event and context
            event, context = extract_event_context(text)
            app.logger.debug(f'Event: {event}, Context: {context}')

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
    app.run(debug=True)
