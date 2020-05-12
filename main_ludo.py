import ludopy
import numpy as np
import evo_ludo
import time

weighties = {
                "out": 0,
                "safe": 0,
                "done": 0,
                "kill": 0,
                "globe": 0
            #, "stacked": np.random.uniform(low=0, high=100) #or just same as globe
            #, "danger": np.random.uniform(low=-100, high=0)
            , "progress": 0
            }

player_0 = evo_ludo.Simple()
player_1 = evo_ludo.Simple()
player_2 = evo_ludo.Simple()
player_3 = evo_ludo.Simple(weighties)

for i in range(10):
    start = time.time()

    wins = evo_ludo.play_matches(100, player_0, player_1, player_2, player_3)

    end = time.time()
    #print("Full game took ", end - start, " seconds")
    print("Win distributiom of ", wins)
#print("Saving history to numpy file")
#g.save_hist(f"game_history.npy")
#print("Saving game video")
#g.save_hist_video(f"game_video.mp4")

