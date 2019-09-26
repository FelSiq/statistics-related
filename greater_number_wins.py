"""In this game, there is two random hidden numbers
inbetween 0 and 100 (inclusive).

The player choose one hidden number at random. He or she look
at the number and can decide to swap or not his or her number.

The player wins if and only if he or she gets the highest number.

This program tests two startegies, and calculate the probability
of the player winning with each startegy.
"""
import numpy as np

repeat_num = int(1.0e4)

# Random strategy
"""
Random startegy:
    1. Player choose one random hidden number.
    2. Player choose to swap at random.
"""
wins = 0
for _ in range(repeat_num):
    numbers = np.random.choice(range(101), 2, replace=False)
    player_choice = np.random.choice((0, 1))
    swap = np.random.choice((0, 1))

    player_choice = abs(player_choice - swap)

    wins += numbers[player_choice] > numbers[1 - player_choice]

print("Win probability with random strategy:", wins / repeat_num)

# Weighted Random strategy
"""
Weighted Random startegy:
    1. Player choose one random hidden number N.
    2. Player choose to swap with probability p = (|50 - N| / 100).
"""
wins = 0
for _ in range(repeat_num):
    numbers = np.random.choice(range(101), 2, replace=False)
    player_choice = np.random.choice((0, 1))

    swap_prob = abs(50 - numbers[player_choice]) / 100
    swap = np.random.choice((0, 1), p=(1 - swap_prob, swap_prob))

    player_choice = abs(player_choice - swap)

    wins += numbers[player_choice] > numbers[1 - player_choice]

print("Win probability with weighted random strategy:", wins / repeat_num)

# Winner strategy
"""
Winning strategy:
    1. Player choose one random hidden number N.
    2. Player don't look at the chosen random number yet.
    3. Player choose one random threshold R in the set
        {0 + 1/2, 1 + 1/2, 2 + 1/2, ..., 99 + 1/2}
        at random (uniformly distributed).
    4. Now player look at the chosen number N and
        swaps if and only in N < R.
"""
wins = 0
for _ in range(repeat_num):
    numbers = np.random.choice(range(101), 2, replace=False)

    player_choice = np.random.choice((0, 1))

    swap_threshold = np.random.randint(0, 100 - 1) + 0.5

    swap = numbers[player_choice] < swap_threshold

    player_choice = abs(player_choice - swap)

    wins += numbers[player_choice] > numbers[1 - player_choice]

print("Win probability with winner strategy:", wins / repeat_num)
