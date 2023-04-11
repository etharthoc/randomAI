from randomFunctions import search_memory
from randomFunctions import combine_current_memory
from randomFunctions import generate_topic_list
from randomFunctions import combine_random
from randomFunctions import combine_topic
from randomFunctions import topic_50_sentence
from randomFunctions import topic_25_sentence
from randomFunctions import random_50_sentence
from randomFunctions import random_25_sentence
from randomFunctions import combine_last_run
from randomFunctions import split_topic_text
import time

generate_topic_list()
time.sleep(1)
split_topic_text()
time.sleep(1)
search_memory()