from json import load
from string import punctuation

global dictionary
global table
# Load the dictionary
with open("dictionary.json") as f:
    dictionary = load(f)

for ignore in ["'", "-"]:
    punctuation = punctuation.replace(ignore, "")

table = str.maketrans(dict.fromkeys(punctuation))


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    HIGHGREEN = "\x1b[6;30;42m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def check_dictionary(text):
    """Check if text contains a word that is in the dictionary and colourise output."""
    words = text.translate(table).split(" ")
    for each in words:
        if each.lower() in dictionary:
            target_index = words.index(each)
            words[target_index] = f"{bcolors.HIGHGREEN}{each}{bcolors.ENDC}"
    return " ".join(words)


def main():
    print("Starting test...")
    test_positive = "Oh no we accidentally caught a shark, not more by-catch!"
    text = check_dictionary(test_positive)
    print(text)

    test_positive = "Or wait it could also be a dolphin or a porpoise I'm not sure"
    text = check_dictionary(test_positive)
    print(text)

    test_negative = "There is a fishing line to port"
    text = check_dictionary(test_negative)
    print(text)


if __name__ == "__main__":
    main()
