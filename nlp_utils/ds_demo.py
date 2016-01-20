from dependency_structure import *

# sentences becomes an object of type Dependency_Structure, which mainly manipulates a list of sentences.
# sentence objects hold a list of nodes that have relationship information available
sentences = Dependency_Structure(raw_input("Please enter file name:"))
sentences.get_tokenized_sentences()
