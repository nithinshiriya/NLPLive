import codecs

def get_file_content(filepath):
    with codecs.open(filepath) as f:
        return f.read()

def get_file_lines(filepath):
    with codecs.open(filepath) as f:
        return f.readlines()
    
def add_to_file(filepath, content):
    with codecs.open(filepath, 'a',  encoding=u'utf_8') as f:
        f.write(content)

def create_file(filepath, content):
    with codecs.open(filepath, 'w',  encoding=u'utf_8') as f:
        f.write(content)