"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None


def question2():
    """
    [Enter a description of what you did here.]
     Agent goes the direction it selects
    """

    answerDiscount = 0.9
    answerNoise = 0.0

    return answerDiscount, answerNoise


def question3a():
    """
    [Enter a description of what you did here.]
    Higher probabilty of going in an unintended direction and
    more punishes more for living
    """

    answerDiscount = 0.9
    answerNoise = 0.31
    answerLivingReward = -1.1

    return answerDiscount, answerNoise, answerLivingReward


def question3b():
    """
    [Enter a description of what you did here.]
    Smaller discount, punish more for living
    """

    answerDiscount = 0.5
    answerNoise = 0.2
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward


def question3c():
    """
    [Enter a description of what you did here.]
    punish more for living
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward


def question3d():
    """
    [Enter a description of what you did here.]
    Made living a little expensive
    """

    answerDiscount = 0.9
    answerNoise = 0.2
    answerLivingReward = 0.0

    return answerDiscount, answerNoise, answerLivingReward


def question3e():
    """
    [Enter a description of what you did here.]
    Expensive living reward and made action more uncertain
    """

    answerDiscount = 0.9
    answerNoise = 1
    answerLivingReward = -2

    return answerDiscount, answerNoise, answerLivingReward


def question6():
    """
    [Enter a description of what you did here.]
    Optimal policy requires a lot of exploration, not enough time to calculate
    """
    return NOT_POSSIBLE


if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
