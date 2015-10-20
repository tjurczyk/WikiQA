class Question:
    question_id = None
    document_title = None
    question = None
    answers = None
    correct_answer = None

    def __init__(self, question_id, document_title, question, answers=[], correct_answer=None):
        self.question_id = question_id
        self.document_title = document_title
        self.question = question
        self.answers = answers
        self.correct_answer = correct_answer

    def add_correct_answer(self, correct_answer):
        self.correct_answer = correct_answer

    def add_answer(self, answer):
        self.answers.append(answer)

    def __str__(self):
        s = \
        "Document title : '" + str(self.document_title) + "'" +\
        "\nQuestion ID    : " + str(self.question_id) +\
        "\nQuestion       : " + str(self.question) +\
        "\nAnswers        : " + "\n               : ".join(self.answers) +\
        "\nAnswers length : " + str(len(self.answers)) +\
        "\nCorrect (index): " + str(self.correct_answer)

        return s