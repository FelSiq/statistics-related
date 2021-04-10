"""This code experimentaly evaluates the win rate of Monty Hall problem."""
import numpy as np

total_games = int(1.0e+6)
print("Total games played (for each situation):", total_games)

# Switch the chosen door
prize_num = 0

for _ in range(total_games):
    doors = [True, False, False]
    np.random.shuffle(doors)

    # Choose a door
    door_id = np.random.randint(0, 3)

    # Now Monty Hall opens a "False" door
    # and player switches the door
    remaining_doors = tuple({0, 1, 2} - {door_id})

    if not doors[remaining_doors[0]]:
        open_door, door_id = remaining_doors

    else:
        door_id, open_door = remaining_doors

    prize_num += doors[door_id]

print("Win rate switching doors:", prize_num / total_games)

# Don't switch the chosen door
prize_num = 0

for _ in range(total_games):
    doors = [True, False, False]
    np.random.shuffle(doors)

    # Choose a door
    door_id = np.random.randint(0, 3)

    # Now Monty Hall opens a "False" door,
    # but player don't switches the chosen door
    prize_num += doors[door_id]

print("Win rate not switching doors:", prize_num / total_games)
