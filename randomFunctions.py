import os
import numpy as np
import openai
import time
import inspect
import requests
import datetime

from scipy import spatial
from sklearn.feature_extraction.text import TfidfVectorizer

# get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

# create a subdirectory path inside the script directory
corpus_dir = os.path.join(script_dir, "randomAI_memory_corpus")

# create a subdirectory path inside the script directory
memory_dir = os.path.join(corpus_dir, "memory")

# define running topic num
global num
num =1

# create the current_sentence.txt file
with open(os.path.join(script_dir, "current_sentence.txt"), "w") as sentence_file:
    sentence_file.write("")

# create the current_topic.txt file
with open(os.path.join(script_dir, "current_topic.txt"), "w") as topic_file:
    topic_file.write("")

# create the current_list.txt file
with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
    list_file.write("")

# create the topic_list.txt file
with open(os.path.join(script_dir, "topic_list.txt"), "w") as tlist_file:
    tlist_file.write("")

# create the memory_sentence.txt file
with open(os.path.join(script_dir, "memory_sentence.txt"), "w") as memory_file:
    memory_file.write("")

# create the interesting.txt file
with open(os.path.join(script_dir, "interesting.txt"), "w") as interesting_file:
    interesting_file.write("I am a super curious entity trying to understand the world from first principles.")

# set up the OpenAI API credentials
openai.api_key = "your-apikey"
api_key = "your-apikey"
url = "https://api.openai.com/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}",
}
# define the function to generate text and write it to the current_list.txt file
def generate_words50():
    print_function_name()
    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": "Generate 50 random words, numbered, separated by commas."}],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    text = response_json['choices'][0]['message']['content']

    print(text)

    # write the text to the current_list.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
        list_file.write(text)

# define the function to generate text and write it to the current_list.txt file
def generate_words25():
    print_function_name()
    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": "Generate 25 random words, numbered, separated by commas."}],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    text = response_json['choices'][0]['message']['content']

    print(text)

    # write the text to the current_list.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
        list_file.write(text)

# define the function to generate text based off of a topic and write it to the current_list.txt file
def generate_topic_words25():
    print_function_name()
    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # read the current_topic.txt file to get the current topic
    with open(os.path.join(script_dir, "current_topic.txt"), "r") as topic_file:
        current_topic = topic_file.read().strip()

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": f"Generate 10 random words about {current_topic}, numbered, separated by commas."}],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    text = response_json['choices'][0]['message']['content']

    print(text)

    # write the text to the current_topic.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
        list_file.write(text)

# define the function to generate text based off of a topic and write it to the current_list.txt file
def generate_topic_words50():
    print_function_name()
    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # read the current_topic.txt file to get the current topic
    with open(os.path.join(script_dir, "current_topic.txt"), "r") as topic_file:
        current_topic = topic_file.read().strip()

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": f"Generate 50 random words about {current_topic}, numbered, separated by commas."}],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    text = response_json['choices'][0]['message']['content']

    print(text)

    # write the text to the current_topic.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
        list_file.write(text)

# define the function to generate text based off of a prompt and write it to the topic_list.txt file
def generate_topic_list():
    print_function_name()
    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # set up the API parameters
    model_engine = "text-davinci-003"
    token_limit = 300

    # generate the text using the OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt="Generate a list of 5 topics(Tech focused but not exclusive).",
        max_tokens=token_limit
    )
    text = response.choices[0].text.strip()
    print(text)

    # write the text to the topic_list.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "topic_list.txt"), "w") as tlist_file:
        tlist_file.write(text)

# define the function to split the text and write it to the current_topic.txt file
def split_topic_text():
    print_function_name()
    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # read the topic_list.txt file
    with open(os.path.join(script_dir, "topic_list.txt"), "r") as tlist_file:
        topic_text = tlist_file.read()

    # split the text into lines
    lines = topic_text.split("\n")

    # get the desired text based on the current number and increment the number
    global num
    desired_text = ""
    for line in lines:
        if line.startswith(str(num) + ". "):
            desired_text = line[len(str(num)) + 2:].strip()
            num += 1
            break
    print(desired_text)

    # write the desired text to the current_topic.txt file
    with open(os.path.join(script_dir, 'current_topic.txt'), "w") as topic_file:
        topic_file.write(desired_text)

# generates a sentence based off of the current list
def generate_sentence():
    print_function_name()
    # Load the list of words from current_list.txt
    with open("current_list.txt", "r") as f:
        word_list = f.read()

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": f"Please create a sentence with words from this list:\n" + "\n".join(word_list)}],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    new_sentence = response_json['choices'][0]['message']['content']

    print(new_sentence)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    # checks and reduces length if needed
    string_length = len(new_sentence)
    print(string_length)
    time.sleep(2)
    check_string_length(new_sentence)

    return new_sentence

# Create a new sentence based off of the following sentence and list of words.
def generate_list_sentence():
    print_function_name()
    # Load the list of words from current_list.txt
    with open("current_list.txt", "r") as f:
        word_list = f.read()

    # Load the sentence from current_sentence.txt
    with open("current_sentence.txt", "r") as f:
        current_sentence = f.read()

    # Load the memory sentence
    with open("memory_sentence.txt", "w") as f:
        f.write(current_sentence)

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": f"create a new sentence with the following (please don't create vertical text, please dont just list words):\n" + "\n".join(
                          current_sentence) + "\n" + "\n".join(word_list)}],
        "temperature": 0.9
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    new_sentence = response_json['choices'][0]['message']['content']

    print(new_sentence)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    # checks and reduces length if needed
    string_length = len(new_sentence)
    print(string_length)
    time.sleep(2)
    check_string_length(new_sentence)

    return new_sentence

# makes the current sentence more concise
def more_consice():
    print_function_name()
    # Load the list of words from current_sentence.txt
    with open(os.path.join(script_dir, "current_sentence.txt"), "r") as list_file:
        current_sentence = list_file.read()

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": f"Please rewrite following sentence so that it is more concise.\n" + "\n".join(current_sentence)}],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    new_sentence = response_json['choices'][0]['message']['content']

    print(new_sentence)
    string_length = len(new_sentence)
    print(string_length)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    return new_sentence

# makes the current sentence much more concise
def much_more_consice():
    print_function_name()
    # Load the list of words from current_sentence.txt
    with open(os.path.join(script_dir, "current_sentence.txt"), "r") as list_file:
        current_sentence = list_file.read()

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": f"Please rewrite following sentence so that it is much more concise.\n" + "\n".join(
                          current_sentence)}],
        "temperature": 0.5
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    new_sentence = response_json['choices'][0]['message']['content']

    print(new_sentence)
    string_length = len(new_sentence)
    print(string_length)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    return new_sentence

# Create a list of words based on the following sentence,numbered, and separated by commas
def generate_sentence_list():
    print_function_name()
    # Load the sentence from current_sentence.txt
    with open("current_sentence.txt", "r") as f:
        current_sentence = f.read()

    # Call the OpenAI API to generate a sentence
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Create a list of words based on the following sentence,numbered, and separated by commas\n" + "\n".join(current_sentence),
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated sentence from the API response
    current_list = response.choices[0].text.strip()
    print(current_list)

    # Write the sentence to current_sentence.txt
    with open("current_list.txt", "w") as f:
        f.write(current_list)

    return current_list

# combine current sentence and memory sentence, offers nice summary
def combine_current_memory():
    print_function_name()
    time.sleep(1)

    # Load the sentence from current_sentence.txt
    with open("current_sentence.txt", "r") as f:
        current_sentence = f.read()

    # Load memory sentence
    with open("memory_sentence.txt", "r") as f:
        memory_sentence = f.read()

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": f"Summarize the following in a sentence or two:\n" + "\n".join(current_sentence) + "\n" + "\n".join(memory_sentence)}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    new_sentence = response_json['choices'][0]['message']['content']

    print(new_sentence)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    # check and reduce length if necessary
    string_length = len(new_sentence)
    print(string_length)
    time.sleep(2)
    check_string_length(new_sentence)

    return new_sentence

# combine current sentence and sentence from last outcome, offers nice summary
def combine_last_run():
    print_function_name()
    time.sleep(1)

    # Load the sentence from current_sentence.txt
    with open("current_sentence.txt", "r") as f:
        current_sentence = f.read()

    # Load last memory sentence
    with open("interesting.txt", "r") as f:
        memory_sentence = f.read()

    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user",
                      "content": f"Summarize the following in a sentence or two:\n" + "\n".join(current_sentence) + "\n" + "\n".join(memory_sentence)}],
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    new_sentence = response_json['choices'][0]['message']['content']

    print(new_sentence)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    # check and reduce length if necessary
    string_length = len(new_sentence)
    print(string_length)
    time.sleep(2)
    check_string_length(new_sentence)

    return new_sentence

# checks and reduces length based on two limits
def check_string_length(my_string):

    if len(my_string) > 900:
        much_more_consice()
        time.sleep(1)
    elif len(my_string) > 600:
        more_consice()
        time.sleep(1)

# Prints current function to console to help debug
def print_function_name():
    print("Running function:", inspect.stack()[1][3])

# attempt to get rid of columns
def column_to_row(input_str):
    delimiter = " "
    # split lines by delimiter
    lines = [line.strip().split(delimiter) for line in input_str.split('\n')]

    # transpose split lines from vertical to horizontal orientation
    transposed_lines = list(map(list, zip(*lines)))

    # join transposed lines with delimiter to form rows
    rows = [delimiter.join(line) + "\n" for line in transposed_lines]

    return "".join(rows)

# saves text to memory
def save_text_file(text):
    base_dir = os.path.join(corpus_dir, "memory")
    # Create a subdirectory with the current date and time as its name
    now = datetime.datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    filepath = os.path.join(base_dir, filename)

    # Write the text string to the new file
    with open(filepath, "w") as f:
        f.write(text)

# searches memory for things related to topic
def search_memory():
    with open("current_topic.txt", "r") as f:
        current_topic = f.read()

    # Define the search query
    query = f"{current_topic}"

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words=None, lowercase=False)

    # Load the corpus (memory) directory into memory
    corpus = []
    filenames = []
    for filename in os.listdir(memory_dir):
        with open(os.path.join(memory_dir, filename), "r") as f:
            corpus.append(f.read())
            filenames.append(filename)

    # Compute the TF-IDF matrix for the entire corpus
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Iterate over each document in the corpus
    for doc_text, filename in zip(corpus, filenames):
        # Compute the TF-IDF scores for each word in the document
        doc_tfidf = tfidf_matrix[filenames.index(filename)].toarray()[0]
        word_tfidf = {word: doc_tfidf[vectorizer.vocabulary_.get(word, -1)] for word in doc_text.split()}

        # Sort the words by their TF-IDF scores in descending order
        sorted_words = sorted(word_tfidf.items(), key=lambda x: x[1], reverse=True)

        # Print the top 5 words in the document by TF-IDF score
        print(f"\nTop 5 words in {filename} by TF-IDF score:")
        for word, score in sorted_words[:5]:
            print(word, score)

        # Compute the TF-IDF scores for the query
        query_tfidf = vectorizer.transform([query]).toarray()[0]
    # Find the document with the highest cosine similarity to the query
    similarity_scores = []
    for filename in os.listdir(memory_dir):
        with open(os.path.join(memory_dir, filename), "r") as f:
            doc_text = f.read()
        doc_tfidf = vectorizer.transform([doc_text]).toarray()[0]
        cosine_sim = 1 - spatial.distance.cosine(query_tfidf, doc_tfidf)
        similarity_scores.append(cosine_sim)
    doc_index = np.argmax(similarity_scores)

    # Print the most relevant document for the query
    print(f"\nMost relevant document for '{query}': {os.listdir(memory_dir)[doc_index]}")

# function that saves to memory at a time (instead of in another function)
def save_to_memory():

    with open("current_sentence.txt", "r") as f:
        current_sentence = f.read()
    time.sleep(1)
    save_text_file(current_sentence)
    time.sleep(1)

# for beginning 50 topic word sentence
def topic_50_sentence():
    time.sleep(1)
    split_topic_text()
    time.sleep(1)
    generate_topic_words50()
    time.sleep(1)
    generate_sentence()
    time.sleep(1)

# for beginning 25 topic word sentence
def topic_25_sentence():
    time.sleep(1)
    split_topic_text()
    time.sleep(1)
    generate_topic_words25()
    time.sleep(1)
    generate_sentence()
    time.sleep(1)

# for beginning 50 random word sentence
def random_50_sentence():
    time.sleep(1)
    generate_words50()
    time.sleep(1)
    generate_sentence()
    time.sleep(1)

# for beginning 25 random word sentence
def random_25_sentence():
    time.sleep(1)
    generate_words25()
    time.sleep(1)
    generate_sentence()
    time.sleep(1)

# assumes sentence exists, generates new sentence on new topic
def combine_topic():
    time.sleep(1)
    split_topic_text()
    time.sleep(1)
    generate_topic_words25()
    time.sleep(1)
    generate_list_sentence()
    time.sleep(1)


# assumes sentence exists, generates new sentence with random
def combine_random():
    time.sleep(1)
    generate_words25()
    time.sleep(1)
    generate_list_sentence()
    time.sleep(1)







# add secondary memory sentence
# taskbot3.0 for randomAI goal is make sentences make sense (objective function kinda)