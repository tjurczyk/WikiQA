import csv
from os import listdir
from os.path import isfile, join

"""Loads dependency structure from file, give object-oriented representations"""

# Usage: Pass a file name to initialize Dependency_Structure with sentences.
# Sentences (node-based structure) can be accessed through Dependency_Structure.
# Most functionality to examine specfic structure and relations will happen at this level.

class Dependency_Structure:
    def ds_from_file(self, file_name, limit=None):
        last_index = -1
        # structure to hold sentence objects after processing
        # chunk is a list of the words in the current sentence
        chunk = []
        with open(file_name, "r") as tsv_in:
            tsv_in = csv.reader(tsv_in, delimiter='\t', quotechar='\x07')
            for row in tsv_in:
                if not row:
                    if chunk:
                        self.sentences.append(Sentence(chunk))
                        chunk = []
                    last_index = -1
                    continue
                if limit and len(self.sentences) > limit:
                    break
                if row and int(row[0]) > last_index:
                    last_index = int(row[0])
                    chunk.append(row)
                else:
                    self.sentences.append(Sentence(chunk))
                    last_index = -1
                    chunk = [row]
            if chunk:
                self.sentences.append(Sentence(chunk))

        print("Token list in sen 1: %s" % self.sentences[1].get_token_list())

    def ds_from_dir(self, source, limit):
        files = [f for f in listdir(source) if isfile(join(source, f))]
        for f in files:
            self.ds_from_file(join(source + '/', f), limit)
        print 'file complete'


    def get_tokenized_sentences(self):
        return [a.get_token_list() for a in self.sentences]

    def get_sentences(self):
        return self.sentences

    def combine(self, ds):
        self.sentences = self.get_sentences() + ds.get_sentences()
        return self

    def combine_all(self, ds_list):
        """Extremely slow, need to optimize"""
        for ds in ds_list:
            self.combine(ds)
        return self

    def __init__(self, source, is_file=True, limit=None, stop_words=[]):
        # accepts name of tsv file containing dependency parsed corpus
        # track whether the current word (row) is the start of a new sentence

        self.sentences = []
        if is_file:
            self.ds_from_file(source, limit)
        else:
            self.ds_from_dir(source, limit)
        print 'Created ', len(self.sentences), ' sentences'


class Sentence:
    def __init__(self, sentence, stop_words=[]):
        self.token_list = None
        self.lemma_list = None
        self.nodes = None
        # for s in sentence:
        nodes = [Node()]
        nodes.extend([Node() for i in range(0, len(sentence))])
        for i, term in enumerate(sentence, 1):
            if term[1] in stop_words:
                nodes.remove(nodes[i])
                continue
            n = nodes[i]
            n.set_form(term[1])
            n.set_lemma(term[2])
            n.set_head(int(term[5]), term[6])
            n.set_pos(term[3])
            # Commented, b/c it's NE and we don't use it
            #n.set_ne_tag(term[9])

            if n.head >= len(nodes):
                print 'ERROR ', i, n.head, n.form
                print sentence
            else:
                h = nodes[n.head]
            h.add_dep(i, n.arc)
        self.nodes = nodes

    def get_node(self, index):
        return self.nodes[index]

    def get_word_index(self, word):
        """
        Return an index of a word in the nodes.
        """
        for idx, n in enumerate(self.nodes):
            if n.form and n.form.lower() == word.lower():
                return idx
        return -1
        #raise KeyError("No word: %s" % word)

    def get_head(self, index):
        if self.nodes[index].get_head_index() is None:
            return None
        else:
            return self.nodes[self.nodes[index].get_head_index()]

    def get_siblings(self, index):
        """
        Return a list of siblings of the index node.

        :rtype: list
        """
        if self.nodes[index].get_head_index() is None:
            # If no head there is no siblings
            return []
        else:
            head_index = self.nodes[index].get_head_index()
            # Return the list of children of the head, not including 'index' node
            return [x for x in self.get_children(head_index) if x != self.nodes[index]]

    def get_token_list(self):
        if not self.token_list:
            self.token_list = []
            for i, word in enumerate(self.nodes):
                if i != 0 and word.get_form() is not None:
                    self.token_list.append(word.get_form().lower())
        return self.token_list

    def get_lemma_list(self):
        if not self.lemma_list:
            self.lemma_list = []
            for i, word in enumerate(self.nodes):
                print ("Inside, get_lemma: %s" % word.get_lemma())
                if i != 0 and word.get_lemma() is not None:
                    self.lemma_list.append(word.get_lemma())
        return self.lemma_list

    def get_dependency_context(self, index):
        """
        Return a dictionary of head, siblings and children.

        :rtype: dict
        """
        context = {'head': self.get_head(index),
                   'siblings': self.get_siblings(index),
                   'children': self.get_children(index)}

        return context

    def get_children(self, index):
        """
        Return a list of children of an index node.

        :rtype list
        """
        children = []
        for i, node in enumerate(self.nodes):
            if node.form is not None and node != self.nodes[index] and node.get_head_index() == index:
                children.append(node)

        return children


class Node:
    def __init__(self):
        self.head = None
        self.arc = None
        self.lemma = None
        self.form = None
        self.pos = None
        self.ne_tag = None
        self.dep = []

    def set_ne_tag(self, tag):
        self.ne_tag = tag

    def set_form(self, form):
        self.form = form

    def set_lemma(self, lemma):
        self.lemma = lemma

    def set_pos(self, pos):
        self.pos = pos

    def set_head(self, head, arc):
        # sets head INDEX not pointer to head
        self.head = head
        self.arc = arc

    def get_form(self):
        return self.form

    def get_lemma(self):
        return self.lemma

    def add_dep(self, index, arc_type):
        self.dep.append((index, arc_type))

    def get_head_index(self):
        return self.head

    def get_head_arc(self):
        return self.arc

    def get_dep_list(self):
        return self.dep

    def is_verb(self):
        if self.pos and self.pos[0].lower() == 'v':
            return True
        else:
            return False