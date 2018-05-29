# Use an import statement at the top
import random

"""
Quiz: Password Generator
Write a function called generate_password that selects three random words from the list of words word_list
and concatenates them into a single string. Your function should not accept any arguments and should
reference the global variable word_list to build the password.
"""

word_file = "words.txt"
word_list = []

#fill up the word_list
with open(word_file,'r') as words:
	for line in words:
		# remove white space and make everything lowercase
		word = line.strip().lower()
		# don't include words that are too long or too short
		if 3 < len(word) < 8:
			word_list.append(word)

# Add your function generate_password here
def generate_password():
    password = []
    for i in range(3):
        password.append(random.choice(word_list))
    password = ''.join(password)
    return(password)

# It should return a string consisting of three random words
# concatenated together without spaces



# test your function
print(generate_password())
