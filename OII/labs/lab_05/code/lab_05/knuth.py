from collections import defaultdict
import markovify

start_str = "Доказательство"
sentences_count = 1000

with open("texts/Кнут-том-1.txt") as f:
    text = f.read()
    for size in [1, 2, 4]:
        print(f'\nn = {size+1}:')
        sentences = defaultdict(int)
        text_model = markovify.Text(text, state_size=size)
        for i in range(sentences_count):
            sentence = text_model.make_sentence_with_start(start_str, strict=False, tries=100)
            sentences[sentence] += 1
        
        print(f'generated {len(sentences)} different sentences')
        
        for sentence in sorted(sentences, key=sentences.get, reverse=True)[:5]:
            print(f'{sentences[sentence]/sentences_count:.3f}:\t {sentence}')
