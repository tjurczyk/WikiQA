import logging
from os.path import exists
from os import makedirs
from generate_input_file import load_questions_from_file

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    questions = {}
    questions['train'], voc, idf = load_questions_from_file('train', -1)
    questions['validate'], voc, idf = load_questions_from_file('validate', -1)
    questions['test'], voc, idf = load_questions_from_file('test', -1)
    logging.info("Questions loaded.")

    data_path = "dependency_parse_data/"
    if not exists(data_path):
        makedirs(data_path)

    for question_set in ["train", "validate", "test"]:
        filename = question_set + ".txt"
        fw = open(data_path + filename, "w")

        for q in questions[question_set]:
            fw.write(q.question + "\n")
            for idx, a in enumerate(q.answers):
                fw.write(a + "\n")

            fw.flush()

        fw.close()

