from randomFunctions import combine_current_memory
from randomFunctions import generate_topic_list
from randomFunctions import combine_random
from randomFunctions import combine_topic
from randomFunctions import topic_50_sentence
from randomFunctions import topic_25_sentence
from randomFunctions import random_50_sentence
from randomFunctions import random_25_sentence
from randomFunctions import combine_last_run
from randomFunctions import save_to_memory
# proof of concept
if __name__ == "__main__":
   random_50_sentence()
   combine_random()
   combine_random()
   combine_last_run()
   combine_random()
   combine_current_memory()
   combine_random()
   save_to_memory()