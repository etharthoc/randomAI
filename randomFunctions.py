import os
import openai
import time

# get the current directory of the Python script
script_dir = os.path.dirname(os.path.abspath(__file__))

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
    interesting_file.write("Shreek is an exploration platform that enables users to create and share interactive lessons and activities about sustainability using smartphones, tablets, and other devices. It provides a secure and reliable platform for teachers and students to create, store, access, and share educational materials related to eco-friendliness, renewable energy, climate change, conservation, carbon, pollution, recycling, ecosystem, waste, natural, renewable, habitat, sustainability, nature, air, earth, water, resources, bio-friendliness, reuse, green, solar, organic, clean, and pollutants. It allows users to engage through a variety of methods such as viralizing, socializing, contenting, influencing, platforming, engaging, hashtaging, feeding, liking, profiling, sharing, online tweeting, commenting, connecting, following, posting, clicking, reaching, growing, communitying, interacting, visualizing, networking, and building relationships.")

# set up the OpenAI API credentials
openai.api_key = "insert api key"

# define the function to generate text and write it to the current_list.txt file
def generate_words50():

    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # set up the API parameters
    model_engine = "text-davinci-003"
    token_limit = 300

    # generate the text using the OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt="Generate 50 random words, numbered, separated by commas.",
        max_tokens=token_limit
    )
    text = response.choices[0].text.strip()
    print(text)

    # write the text to the current_list.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
        list_file.write(text)

# define the function to generate text and write it to the current_list.txt file
def generate_words25():

    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # set up the API parameters
    model_engine = "text-davinci-003"
    token_limit = 300

    # generate the text using the OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt="Generate 25 random words, numbered, separated by commas.",
        max_tokens=token_limit
    )
    text = response.choices[0].text.strip()
    print(text)

    # write the text to the current_list.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
        list_file.write(text)

# define the function to generate text based off of a topic and write it to the current_list.txt file
def generate_topic_words25():

    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # read the current_topic.txt file to get the current topic
    with open(os.path.join(script_dir, "current_topic.txt"), "r") as topic_file:
        current_topic = topic_file.read().strip()

    # set up the API parameters
    model_engine = "text-davinci-003"
    token_limit = 300

    # generate the text using the OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt=f"Generate 25 random words about {current_topic}, numbered, separated by commas.",
        max_tokens=token_limit
    )
    text = response.choices[0].text.strip()
    print(text)

    # write the text to the current_topic.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
        list_file.write(text)

# define the function to generate text based off of a topic and write it to the current_list.txt file
def generate_topic_words50():

    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # read the current_topic.txt file to get the current topic
    with open(os.path.join(script_dir, "current_topic.txt"), "r") as topic_file:
        current_topic = topic_file.read().strip()

    # set up the API parameters
    model_engine = "text-davinci-003"
    token_limit = 300

    # generate the text using the OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt=f"Generate 50 random words about {current_topic}, numbered, separated by commas.",
        max_tokens=token_limit
    )
    text = response.choices[0].text.strip()
    print(text)

    # write the text to the current_topic.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "current_list.txt"), "w") as list_file:
        list_file.write(text)

# define the function to generate text based off of a prompt and write it to the topic_list.txt file
def generate_topic_list():

    # get the current directory of the Python script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # set up the API parameters
    model_engine = "text-davinci-003"
    token_limit = 300

    # generate the text using the OpenAI API
    response = openai.Completion.create(
        engine=model_engine,
        prompt="Generate a list of 5 topics.",
        max_tokens=token_limit
    )
    text = response.choices[0].text.strip()
    print(text)

    # write the text to the topic_list.txt file, overwriting the contents if there are any
    with open(os.path.join(script_dir, "topic_list.txt"), "w") as tlist_file:
        tlist_file.write(text)

# define the function to split the text and write it to the current_topic.txt file
def split_topic_text():

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

    # Load the list of words from current_list.txt
    with open("current_list.txt", "r") as f:
        word_list = f.read().splitlines()

    # Call the OpenAI API to generate a sentence
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Please create a sentence with words from this list:\n" + "\n".join(word_list),
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated sentence from the API response
    sentence = response.choices[0].text.strip()
    check_string_length(sentence)
    print(sentence)
    string_length = len(sentence)
    print(string_length)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(sentence)

    return sentence

# Create a new sentence based off of the following sentence and list of words.
def generate_list_sentence():

    # Load the list of words from current_list.txt
    with open("current_list.txt", "r") as f:
        word_list = f.read()

    # Load the sentence from current_sentence.txt
    with open("current_sentence.txt", "r") as f:
        current_sentence = f.read()

    # Load the memory sentence
    with open("memory_sentence.txt", "w") as f:
        f.write(current_sentence)

    # Call the OpenAI API to generate a sentence
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="create something new with the following:\n" + "\n".join(current_sentence) + "\n".join(word_list),
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated sentence from the API response
    new_sentence = response.choices[0].text.strip()
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

    # Load the list of words from current_sentence.txt
    with open(os.path.join(script_dir, "current_sentence.txt"), "r") as list_file:
        current_sentence = list_file.read()

    # Call the OpenAI API to generate a sentence
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Please rewrite following sentence so that it is more concise.\n" + "\n".join(current_sentence),
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated sentence from the API response
    new_sentence = response.choices[0].text.strip()
    print(new_sentence)
    string_length = len(new_sentence)
    print(string_length)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    return new_sentence

# makes the current sentence much more concise
def much_more_consice():

    # Load the list of words from current_sentence.txt
    with open(os.path.join(script_dir, "current_sentence.txt"), "r") as list_file:
        current_sentence = list_file.read()

    # Call the OpenAI API to generate a sentence
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Please rewrite following sentence so that the sentence is much much more concise.\n" + "\n".join(current_sentence),
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated sentence from the API response
    new_sentence = response.choices[0].text.strip()
    print(new_sentence)
    string_length = len(new_sentence)
    print(string_length)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    return new_sentence

# Create a list of words based on the following sentence,numbered, and separated by commas
def generate_sentence_list():

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

    time.sleep(1)

    # Load the sentence from current_sentence.txt
    with open("current_sentence.txt", "r") as f:
        current_sentence = f.read()

    # Load memory sentence
    with open("memory_sentence.txt", "r") as f:
        memory_sentence = f.read()

    # Call the OpenAI API to generate a sentence
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Summarize the following:\n" + "\n".join(current_sentence) + "\n".join(memory_sentence),
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated sentence from the API response
    new_sentence = response.choices[0].text.strip()
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

    time.sleep(1)

    # Load the sentence from current_sentence.txt
    with open("current_sentence.txt", "r") as f:
        current_sentence = f.read()

    # Load last memory sentence
    with open("interesting.txt", "r") as f:
        memory_sentence = f.read()

    # Call the OpenAI API to generate a sentence
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Summarize the following:\n" + "\n".join(current_sentence) + "\n".join(memory_sentence),
        max_tokens=300,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the generated sentence from the API response
    new_sentence = response.choices[0].text.strip()
    print(new_sentence)

    # Write the sentence to current_sentence.txt
    with open("current_sentence.txt", "w") as f:
        f.write(new_sentence)

    # Check and reduce length if needed
    string_length = len(new_sentence)
    print(string_length)
    time.sleep(2)
    check_string_length(new_sentence)

    return new_sentence

# checks and reduces length based on two limits
def check_string_length(my_string):

    if len(my_string) > 900:
        much_more_consice()
    elif len(my_string) > 600:
        more_consice()

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