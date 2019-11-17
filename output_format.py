

def wrap_text_length(text, length):
    wrapped = ''

    remain = length
    for word in text.split():
        wrapped += (word + ' ')
        remain -= len(word)

        if (remain <= 0):
            wrapped += '\n'
            remain = length

    return wrapped