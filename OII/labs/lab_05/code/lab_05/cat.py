from collections import defaultdict
import markovify

start_str = "кошка"
sentences_count = 1000000
sentences = defaultdict(int)

model_1 = markovify.Text("мышку съела кошка", state_size=1)
model_2 = markovify.Text("кошка съела мышку", state_size=1)
text_model = markovify.combine([model_1, model_2], [ 1, 1 ])
for i in range(sentences_count):
    sentence = text_model.make_sentence_with_start(start_str, strict=False, tries=50)
    sentences[sentence] += 1

print(f'generated {len(sentences)} different sentences')

for sentence in sorted(sentences, key=sentences.get, reverse=True)[:10]:
    print(f'{sentences[sentence]/sentences_count:.3f}:\t {sentence}')
