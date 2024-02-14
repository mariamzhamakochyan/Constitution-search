import sys
import gensim
import gensim.downloader as api

class ConstitutionHelper:
    def __init__(self, filename):
        self.filename = filename

    def read_constitution_file(self):
        try:
            with open(self.filename, "r") as file:
                constitution_text = file.read()
            return constitution_text
        except FileNotFoundError:
            print(f"Error: The {self.filename} file does not exist.")
            return None

    @staticmethod
    def read_search_command():
        if len(sys.argv) < 3:
            print("Error: Please provide a search text.")
            sys.exit(1)
        return " ".join(sys.argv[2:])

    @staticmethod
    def split_into_sentences(text):
        sentences = []
        current_sentence = ""
        for char in text:
            current_sentence += char
            if char in ['.', '?']:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence:
            sentences.append(current_sentence.strip())
        return sentences


    @staticmethod
    def filter_matching_sentences(sentences, search_text):
        search_words = search_text.lower().split()
        matching_sentences = []
        for sentence in sentences:
            sentence_words = sentence.lower().split()
            if any(word in sentence_words for word in search_words):
                matching_sentences.append(sentence)
        return matching_sentences

    @staticmethod
    def enumerate_sentences(sentences):
        enumerated_sentences = []
        for i, sentence in enumerate(sentences, start=1):
            enumerated_sentences.append(f"{i}. {sentence}")
        return enumerated_sentences

    @staticmethod
    def find_closest_sentences(search_text, matching_sentences):
        word_vectors = api.load("glove-wiki-gigaword-100")
        search_text_tokens = gensim.utils.simple_preprocess(search_text)
        search_text_tokens = [token for token in search_text_tokens if token in word_vectors.key_to_index]
        if not search_text_tokens:
            print("No matches")
            return []
        search_vector = sum(word_vectors.get_vector(word) for word in search_text_tokens) / len(search_text_tokens)
        closest_sentence = None
        max_similarity = -1
        for sentence in matching_sentences:
            sentence_tokens = gensim.utils.simple_preprocess(sentence)
            sentence_tokens = [token for token in sentence_tokens if token in word_vectors.key_to_index]
            if sentence_tokens:
                sentence_vector = sum(word_vectors.get_vector(word) for word in sentence_tokens) / len(sentence_tokens)
                similarity = word_vectors.cosine_similarities(search_vector, [sentence_vector])[0]
                if similarity > max_similarity:
                    max_similarity = similarity
                    closest_sentence = sentence
            else:
                print(f"No matchesfor sentence: {sentence}")
        return closest_sentence

if __name__ == "__main__":
    helper = ConstitutionHelper("Constitution.txt") 
    constitution_text = helper.read_constitution_file()
    if constitution_text:
        search_text = ConstitutionHelper.read_search_command()
        constitution_sentences = ConstitutionHelper.split_into_sentences(constitution_text)
        matching_sentences = ConstitutionHelper.filter_matching_sentences(constitution_sentences, search_text)
        if matching_sentences:
            closest_sentence = ConstitutionHelper.find_closest_sentences(search_text, matching_sentences)
            if closest_sentence:
                print("Closest sentence:")
                print(closest_sentence)
                choice = input("Do you want to see all the closest sentences? (yes/no): ")
                if choice.lower() == "yes":
                    closest_sentences = [closest_sentence] + [sentence for sentence in matching_sentences if sentence != closest_sentence]
                    print("All closest sentences:")
                    for sentence in closest_sentences:
                        print(sentence)
            else:
                print("No matching sentences found.")
        else:
            print("No matching sentences found.")
