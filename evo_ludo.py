import ludopy
import numpy as np

home = 0
stars = np.array([5, 12, 18, 25, 31, 38, 44, 51])
goal_entrance = 53
goal = 59
globes = np.array([9, 22, 35, 48])
enemy_globes = np.array([14, 27, 40])

class Simple:
    def __init__(self, weights=None):
        if weights == None:
            self.weights = {
                "out": np.random.uniform(low=0.0, high=100),
                "safe": np.random.uniform(low=0, high=100),
                "done": np.random.uniform(low=0, high=100),
                "kill": np.random.uniform(low=0, high=100),
                "globe": np.random.uniform(low=0, high=100)
            #, "stacked": np.random.uniform(low=0, high=100) #or just same as globe
            #, "danger": np.random.uniform(low=-100, high=0)
            #, "progress": np.random.uniform(low=0, high=5)
            }
        else:
            self.weights = weights

    def play(self, dice, move_pieces, player_pieces, enemy_pieces):
        current_hvalues = np.array([self.heuristic(piece, enemy_pieces) for piece in player_pieces])
        future_hvalues = np.array([self.heuristic(self.new_position(dice, piece, enemy_pieces), enemy_pieces)
                                   for piece in player_pieces])
        hvalue_gains = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        hvalue_gains[move_pieces] = future_hvalues[move_pieces] - current_hvalues[move_pieces]
        piece_to_move = np.argmax(hvalue_gains)         #we need to still have an array of same size for argmax to work
        return piece_to_move

    def heuristic(self, position, enemy_pieces):
        if position == 53:
            return (self.safe(position)*self.weights["safe"] + self.out(position)*self.weights["out"]
                + self.done(position)*self.weights["done"] + self.kill(1, enemy_pieces)*self.weights["kill"]
                + self.globe(position)*self.weights["globe"])
        else:
            return (self.safe(position)*self.weights["safe"] + self.out(position)*self.weights["out"]
                + self.done(position)*self.weights["done"] + self.kill(position, enemy_pieces)*self.weights["kill"]
                + self.globe(position)*self.weights["globe"])


    #get the enemy pieces which are not at home or in the goal area
    #and with indices from your PoV
    def enemy_pieces_my_PoV(self, enemy_pieces):
        #for each piece and each enemy player
        adjusted_enemy_pieces = []
        for enemy, enemy_globe in zip(enemy_pieces, enemy_globes):
            enemy_positions = enemy[np.where(home < enemy)]
            enemy_positions = enemy_positions[np.where(enemy_positions < goal_entrance)]
            adjusted_enemy_positions = enemy_positions + enemy_globe - 1

            inx = np.where(adjusted_enemy_positions > goal_entrance -1)
            mod = adjusted_enemy_positions % (goal_entrance - 1)
            adjusted_enemy_positions[inx] = mod[inx]

            adjusted_enemy_pieces.append(adjusted_enemy_positions)
        return adjusted_enemy_pieces

    #check if you can kill
    def kill(self, position, enemy_pieces):
            adjusted_enemy_pieces = self.enemy_pieces_my_PoV(enemy_pieces)
            for enemies in adjusted_enemy_pieces:
                if any(enemies == position):
                    return True
            return False

    def safe(self, player_piece):
        return player_piece >= goal_entrance

    def done(self, player_piece):
        return player_piece == goal

    def out(self, player_piece):
        return player_piece > home

    def globe(self, player_piece):
        return any(globes == player_piece) or player_piece == 1

    def dies(self, position, enemy_pieces):
        a, b = position, enemy_pieces
        enemies = self.enemy_pieces_my_PoV(enemy_pieces)
        enemies_at_new_position = 0
        for enemy in enemies:
            for pieces in enemy:
                if pieces == position:
                    enemies_at_new_position += 1

        if enemies_at_new_position == 2:
            if position == home or position == goal_entrance:
                return False
            else:
                return True

        if enemies_at_new_position == 1:
            if self.globe(position):
                return True
            for enemy, enemy_globe in zip(enemy_pieces, enemy_globes):
                if position == enemy_globe and any(enemy == enemy_globe):
                    return True
            else:
                return False

    def new_position(self, dice, player_piece, enemy_pieces):
        #if dies
        if self.dies(player_piece + dice, enemy_pieces):
            return 0
        #if at start
        if player_piece == 0 and dice == 6:
            return 1
        #if at end
        if player_piece + dice > goal:
            return goal - (player_piece + dice - goal)
        #if on star
        landed_on = np.where(stars == player_piece + dice)
        if landed_on[0] == None: #needs to find the star you land on, so it knows if 6 or 7 free moves
            if landed_on + 1 == len(stars):
                return goal
            else:
                return stars[landed_on + 1]
        #if nothing special
        else:
            return player_piece + dice

"""
class evolutionary:
    def __init__(self):
        #TODO: a dict with different weights
        pass

    #population struct

    def train(self, game, population_size, iterations):
        population = start_population(population_size)
        while iterations:
            evaluation = evaluate(population)
            population = breed(population, evaluation)
            population = mutate(population)

    def evaluate(self, population):
        pass

    def breed(self, population, evaluation):
        pass

    def mutate(self, population):
        pass
"""
#set up some weights (value of progress (multiple terms x, x^2, x^3, x^4),
# value of standing on globe
# value of standing on star
# value of standing on enemy home
# value of standing on own home
# value of being out
# value of being done
# value of enemy mean progess
# value of enemy greatest progress
# value of being in front of enemy
# value of being in front of enemy home
# value of being behind enemy/enemies

#set up a heuristic (distance from goal or distance from goal - enemies distance from goal)