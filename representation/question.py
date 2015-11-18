class Question:
    question_id = None
    document_title = None
    question = None
    answers = None
    correct_answer = None

    def __init__(self, question, answer, correct_answer=None):
        self.question = question
        self.answers = [answer,]
        if correct_answer is None:
            self.correct_answer = []
        else:
            self.correct_answer = [correct_answer, ]

    def add_correct_answer(self, correct_answer):
        self.correct_answer.append(correct_answer)

    def add_answer(self, answer):
        self.answers.append(answer)

    def __str__(self):
        s = "Question       : " + str(self.question) +\
            "\nAnswers        : " + "\n               : ".join(self.answers) +\
            "\nAnswers length : " + str(len(self.answers)) +\
            "\nCorrect (index): " + str(self.correct_answer)

        return s