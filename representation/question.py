class Question:
    question_id = None
    document_title = None
    question = None
    question_dep = None
    answers = None
    answer_dep = None
    correct_answer = None

    def __init__(self, question, question_dep, answer, answer_dep, correct_answer=None):
        self.question = question
        self.question_dep = question_dep
        self.answers = [answer,]
        self.answer_dep = [answer_dep,]
        if correct_answer is None:
            self.correct_answer = []
        else:
            self.correct_answer = [correct_answer, ]

    def add_correct_answer(self, correct_answer):
        self.correct_answer.append(correct_answer)

    def add_answer(self, answer, answer_dep):
        self.answers.append(answer)
        self.answer_dep.append(answer_dep)

    def __str__(self):
        s = "Question       : " + str(self.question) +\
            "\nAnswers        : " + "\n               : ".join(self.answers) +\
            "\nAnswers length : " + str(len(self.answers)) +\
            "\nCorrect (index): " + str(self.correct_answer)

        return s