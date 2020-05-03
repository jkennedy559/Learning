from itertools import product, combinations
suits = 'SHDC'
numbers = 'AKQJT98765432'


def generate_five(cards):
    """Return sets of 5 cards faces for a hand"""
    while len(cards) > 4:
        yield cards[0:5]
        cards = cards[1:]


def generate_straight_flush(numbers, suits):
    """Return tuple of straight flush hand and it's superiority ranking."""
    superiority = 1
    for five_numbers in generate_five(numbers):
        for suit in suits:
            hand = []
            for number in five_numbers:
                card = number + suit
                hand.append(card)
            yield (' '.join(hand), superiority)
        superiority += 1


def generate_quads(numbers, suits):
    """Return four of kind hands, no superiority ranking required."""
    for number in numbers:
        kickers = numbers.replace(number, '')
        quads = [quad + suit for quad in number for suit in suits]
        for kicker in kickers:
            for suit in suits:
                kicker_card = kicker + suit
                yield ' '.join(quads + [kicker_card])


def generate_full_houses(numbers, suits):
    """Return full houses with superiority."""
    superiority = 1
    for number in numbers:
        triples = [[num + c1, num + c2, num + c3] for num in number for c1, c2, c3 in combinations(suits, 3)]
        offset_numbers = numbers.replace(number, '')
        for offset_number in offset_numbers:
            offset_pair = [[offset + c1, offset + c2] for offset in offset_number for c1, c2 in combinations(suits, 2)]
            full_houses = [full_house[0] + full_house[1] for full_house in product(triples, offset_pair)]
            for full_house in full_houses:
                yield(' '.join(full_house), superiority)
            superiority += 1


class PokerHand(object):
    RESULT = ["Loss", "Tie", "Win"]

    def __init__(self):
        pass

    def compare_with(self, other):
        pass
