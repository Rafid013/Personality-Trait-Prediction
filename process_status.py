import string


def clean_status(statuses):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    i = 0
    for status in statuses:
        # tokenize
        status = status.split()
        # convert to lower case
        status = [word.lower() for word in status]
        # remove punctuation from each token
        status = [w.translate(table) for w in status]
        # remove hanging 's' and 'a'
        status = [word for word in status if len(word) > 1]
        # remove tokens with numbers in them
        status = [word for word in status if word.isalpha()]
        # store as string
        statuses.iloc[i] = ' '.join(status)
        i += 1


def calculate_max_length(statuses):
    max_length = 0
    for status in statuses:
        length = len(status.split())
        if length > max_length:
            max_length = length
    return max_length
