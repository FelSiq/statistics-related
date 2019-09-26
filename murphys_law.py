import matplotlib.pyplot as plt
import numpy as np


def build_deck():
    """Deck is composed by 54 cards, which values of all cards
    below 10 (excluding) is it's own value, and the values of
    Aces, 10's, J's (jacks), Q's (queens), K (kings) and Jokers
    are all worth 1.
    """
    special_card_values = {
        "Ace": 1,
        "10": 1,
        "Jack": 1,
        "Queen": 1,
        "King": 1,
        "Jokers": 1,
    }

    card_suits = ("Spades", "Clubs", "Hearts", "Diamonds")

    usual_values = np.array(
        [np.repeat(i, len(card_suits)) for i in range(2, 10)]).flatten()

    special_values = np.concatenate(
        np.array([
            np.repeat(
                special_card_values.get(card),
                len(card_suits) if card != "Jokers" else 2)
            for card in special_card_values.keys()
        ]))

    deck = np.concatenate((usual_values, special_values))

    print("deck length =", len(deck))

    return deck


experiment_repeats = int(1e+6)

deck = build_deck()
np.random.shuffle(deck)

freq = np.array([0 for i in range(len(deck))])
foot_prints = np.array([0 for i in range(len(deck))])

for _ in range(experiment_repeats):
    start_number = np.random.randint(0, 10)

    result = start_number
    cum_sum = start_number

    while cum_sum < len(deck):
        foot_prints[cum_sum] += 1
        result = deck[cum_sum]
        cum_sum += result

    freq[cum_sum - result] += 1

print("Result probabilities:")
print({
    index: freq[index] / experiment_repeats
    for index in range(len(deck)) if freq[index] > 0
})

plt.bar(range(len(deck)), foot_prints / sum(foot_prints))
plt.title("Density of footprints")
plt.show()
