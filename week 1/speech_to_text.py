# -*- coding: utf-8 -*-
"""
Spyder Editor

V. Shoda
"""
filepath = "/home/verareyes/twitch_clips/owl/wav/"     #Input audio file path
output_filepath = "/home/verareyes/twitch_clips/" #Final transcript path
bucketname = "shodaclips" #Name of the bucket created in GCloud

# Import libraries
from pydub import AudioSegment
import io
import os
from google.cloud import speech_v1p1beta1 as speech #2 speakers
from google.cloud.speech_v1p1beta1 import enums #2 speakers
from google.cloud.speech_v1p1beta1 import types #2 speakers
import wave
from google.cloud import storage

def stereo_to_mono(audio_file_name):
    sound = AudioSegment.from_wav(audio_file_name)
    sound = sound.set_channels(1)
    sound.export(audio_file_name, format="wav")

def frame_rate_channel(audio_file_name):
    with wave.open(audio_file_name, "rb") as wave_file:
        frame_rate = wave_file.getframerate()
        channels = wave_file.getnchannels()
        return frame_rate,channels

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def delete_blob(bucket_name, blob_name):
    """Deletes a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.delete()

#using Google Speech API gCloud account to convert audio to text
def google_transcribe(audio_file_name):
    
    file_name = filepath + audio_file_name

    # The name of the audio file to transcribe
    
    frame_rate, channels = frame_rate_channel(file_name)
    
    if channels > 1:
        stereo_to_mono(file_name)
    
    bucket_name = bucketname
    source_file_name = filepath + audio_file_name
    destination_blob_name = audio_file_name
    
    upload_blob(bucket_name, source_file_name, destination_blob_name)
    
    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
    transcript = ''
    
    client = speech.SpeechClient()
    audio = types.RecognitionAudio(uri=gcs_uri)

    config = types.RecognitionConfig(
    encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=frame_rate,
    language_code='en-US',
    enable_speaker_diarization=True,
    diarization_speaker_count=2) #2 speakers

    # Detects speech in the audio file
    operation = client.long_running_recognize(config, audio)
    response = operation.result(timeout=10000)
    result = response.results[-1] #2 speakers
    words_info = result.alternatives[0].words #2 speakers
    
    tag=1 #2 speakers
    speaker="" #2 speakers

    for word_info in words_info: #2 speakers
        if word_info.speaker_tag==tag: #2 speakers
            speaker=speaker+" "+word_info.word #2 speakers
        else: #2 speakers
            transcript += "speaker {}: {}".format(tag,speaker) + '\n' #2 speakers
            tag=word_info.speaker_tag #2 speakers
            speaker=""+word_info.word #2 speakers
 
    transcript += "speaker {}: {}".format(tag,speaker) #2 speakers
    
    delete_blob(bucket_name, destination_blob_name)
    return transcript

#writing the transcripts and output as text file
def write_transcripts(transcript_filename,transcript):
    f= open(output_filepath + transcript_filename,"w+")
    f.write(transcript)
    f.close()
    
if __name__ == "__main__":
    for audio_file_name in os.listdir(filepath):
        transcript = google_transcribe(audio_file_name)
        transcript_filename = audio_file_name.split('.')[0] + '.txt'
        write_transcripts(transcript_filename,transcript)




