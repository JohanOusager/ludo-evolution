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
            , "progress": np.random.uniform(low=0, high=25)
            }
        else:
            self.weights = weights

    def play(self, dice, move_pieces, player_pieces, enemy_pieces):
        adjust = self.enemy_pieces_my_PoV(enemy_pieces)
        current_hvalues = np.array([self.heuristic(piece, enemy_pieces) for piece in player_pieces])
        future_hvalues = np.squeeze(np.array([self.heuristic(self.new_position(dice, piece, enemy_pieces), enemy_pieces)
                                   for piece in player_pieces]))
        hvalue_gains = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
        hvalue_gains[move_pieces] = future_hvalues[move_pieces] - current_hvalues[move_pieces]
        piece_to_move = np.argmax(hvalue_gains)         #we need to still have an array of same size for argmax to work
        return piece_to_move

    def heuristic(self, position, enemy_pieces):
        if position == 53:
            return (self.safe(position)*self.weights["safe"] + self.out(position)*self.weights["out"]
                + self.done(position)*self.weights["done"] + self.kill(1, enemy_pieces)*self.weights["kill"]
                + self.globe(position)*self.weights["globe"]+position*self.weights["progress"])
        else:
            return (self.safe(position)*self.weights["safe"] + self.out(position)*self.weights["out"]
                + self.done(position)*self.weights["done"] + self.kill(position, enemy_pieces)*self.weights["kill"]
                + self.globe(position)*self.weights["globe"]+position*self.weights["progress"])

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
                if position == enemy_globe and any(np.array(enemy) == enemy_globe):
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
        if np.any(landed_on): #needs to find the star you land on, so it knows if 6 or 7 free moves
            if landed_on[0] + 1 == len(stars):
                return goal
            else:
                if self.dies(stars[landed_on[0] + 1], enemy_pieces):
                    return 0
                else:
                    return stars[landed_on[0] + 1]
        #if nothing special
        else:
            return player_piece + dice


#TODO: ADD NORMALIZATION

class Evolutionary:
    def __init__(self, population_size=None, population=None):
        if population == None and population_size != None:
            self.population = np.array([Simple() for i in range(population_size)])
        elif population != None:
            self.population = population
        else:
            raise ValueError("Evolutionary class must be initialized with a population size or an existing population")
        self.evaluation=np.zeros(self.population.shape[0])

    def train(self, train_iterations):
        for i in range(train_iterations):
            print("Gen ", i)
            self.evaluation = self.evaluate(10)
            self.population = self.breed()
            self.population = self.mutate()
            print("Best: ", np.max(self.evaluation))

    def evaluate(self, iterations=100):

        def fitness(individual, matches):
            def match():
                indices = np.random.choice(range(self.population.shape[0]), 3, replace=False)
                opponents = self.population[indices]
                i = play_match(individual, opponents[0], opponents[1], opponents[2])
                return i == 0
            return np.average(np.array([match() for i in range(matches)]))

        evaluation = np.array([fitness(individual, iterations) for individual in self.population])
        return evaluation

    #roulette wheel select
    def roulette(self, nr_of_winners):
        winners = []

        sum = np.sum(self.evaluation)
        wheel = np.array([np.sum(self.evaluation[:i]) for i in range(len(self.evaluation))])

        for w in range(nr_of_winners):
            result = np.random.randint(0, sum)
            for i in range(len(wheel)):
                if result <= wheel[i]:
                    winners.append(self.population[i])
                    break

        return winners

    def breed(self, nr_of_parents=None):
        if nr_of_parents==None:
            nr_of_parents = np.max([2, int(len(self.evaluation)/10)])
            if nr_of_parents % 2 != 0:
                nr_of_parents -= 1

        parents = self.roulette(nr_of_parents)

        #pair the parents
        parents = np.reshape(parents, [int(len(parents)/2), 2])
        nr_of_kids = int(self.population.shape[0]/parents.shape[0])
        if nr_of_kids * parents.shape[0] < 100:
            nr_of_kids += 1

        def make_kids(parents):

            def make_kid(parents):
                weights = parents[0].weights.copy()
                for key in weights.keys():
                    if np.random.uniform(0, 1) >= 0.5:
                        weights[key] = parents[1].weights[key]
                return Simple(weights)

            kids = np.array([make_kid(parents) for i in range(nr_of_kids)])
            return kids

        new_generation = []
        for pair in parents:
            new_generation.append(make_kids(pair))

        return new_generation[:self.population.shape[0]]

    def mutate(self, rate=0.05):
        for individual in self.population:
            a = self.population[0]
            for key in individual.weights.keys():
                if np.random.uniform(0, 1) <= rate:
                    individual.weights[key] *= np.random.uniform(0.5, 1.5)



def play_match(player_0, player_1, player_2, player_3):
    g = ludopy.Game()
    there_is_a_winner = False

    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()
        if len(move_pieces):
            if player_i == 0:
                piece_to_move = player_0.play(dice, move_pieces, player_pieces, enemy_pieces)
            elif player_i == 1:
                piece_to_move = player_1.play(dice, move_pieces, player_pieces, enemy_pieces)
            elif player_i == 2:
                piece_to_move = player_2.play(dice, move_pieces, player_pieces, enemy_pieces)
            elif player_i == 3:
                piece_to_move = player_3.play(dice, move_pieces, player_pieces, enemy_pieces)
            else:
                raise ValueError("No players turn")
        else:
            piece_to_move = -1

        _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        if there_is_a_winner:
            return player_i

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
evo = Evolutionary(population_size=10)
evo.train(10)