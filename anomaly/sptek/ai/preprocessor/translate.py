import string

class Translate(object):
    
    def __init__(self, token):
        self._token_ = token

    def translate_split(self, text):
        exclist = ""
        if self._token_.punctuation:
            exclist = exclist + string.punctuation

        if self._token_.digits:
            exclist = exclist + string.digits

        table_ = str.maketrans(exclist, ' '*len(exclist))

        a = []
        for t in text:
              a.append(' '.join(t.translate(table_).split()))
        return a
