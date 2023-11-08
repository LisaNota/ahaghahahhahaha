from dask import delayed, compute
from transformers import pipeline
from pydub import AudioSegment
import math
import os


# Разделяем файл
def split_audio(input_file, output_folder, duration):
    audio = AudioSegment.from_mp3(input_file)
    total_length = len(audio)
    num_parts = math.ceil(total_length / (duration * 1000))
 
    for i in range(num_parts):
        start = i * duration * 1000
        end = (i + 1) * duration * 1000
        split_audio = audio[start:end]
        output_path = os.path.join(output_folder, f"part_{i+1}.mp3")
        split_audio.export(output_path, format="mp3")
        print(f"Exported {output_path}")
    
    return num_parts

SECONDS_PER_FILE = 60
count = split_audio('interview2.mp3', 'input/interview2/', SECONDS_PER_FILE)

# Функция для обработки одного файла
def process_audio_part(audio_path):
    asr = pipeline( 
        'automatic-speech-recognition', 
        model='jonatasgrosman/wav2vec2-large-xlsr-53-russian')
    
    spell_cor = pipeline(model='UrukHan/t5-russian-spell')

    res = asr(audio_path)
    text = res['text']
    
    res = spell_cor(text)[0]
    text = res['generated_text']
    
    return text

# создаем отложенные задачи вместо прямого выполнения
audio_paths = [f'input/interview2/part_{i}.mp3' for i in range(1, count + 1)]
processed_texts = [delayed(process_audio_part)(audio_path) for audio_path in audio_paths]

# распредленное вычисление (невероятно)
combined_text = delayed(lambda *texts: '\n'.join(texts))(*processed_texts)
computed_text = combined_text.compute()

# общий сбор буков
with open('output/combined_text.txt', 'w', encoding='utf-8') as f:
    f.write(computed_text)

model_name = "ruselkomp/sbert_large_nlu_ru-finetuned-squad"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

questions = [
    'не возражаешь, если я спрошу?',
    'не могли бы вы рассказать мне о вашей любимой книге и почему она вам так нравится?',
    'о каком месте путешествия вы мечтаете, и что вас в нем привлекает?',
    'если бы вы могли обладать какой-либо сверхспособностью, какой бы она была и как бы вы ее использовали?',
    'какое самое запоминающееся приключение или путешествие, в котором вы побывали?',
    'если бы вы могли поужинать с какой-либо исторической личностью, кто бы это был и о чем бы вы их спросили?']

for question in questions:
    QA_input = {
        'question': question,
        'context': computed_text
    }

    res = nlp(QA_input)
    print(f'{res["score"]:.4f}: {res["answer"]}')
