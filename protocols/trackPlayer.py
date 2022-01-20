
import pydirectinput
import time

def trackPlayer(playerCenter, playerBox, playerLock, left, top):

    # print(playerCenter)
    if playerCenter != None: # player found!

        mouse = pydirectinput.position()
        mouseRel = (mouse[0] - left, mouse[1] - top)

        if playerCenter[1] - top > 150:  # Check if a jump maneuver required

            if playerLock == 0: # if this is the first time seeing the player
                print("[SUCCESS] Tracking player...")
                # print("{}. {}".format(player[0], player[1]))

                # print("mouse - screen: {}".format(mouseRel))
                # print(player[0]-mouseRel[0])
                # print(player[1]-mouseRel[1])

                pydirectinput.move(playerCenter[0]-mouseRel[0], 0)

                if playerBox[1][1] - playerBox[0][1] > 200: # if the player is significantly far away
                    pydirectinput.keyDown('w')
                    time.sleep(0.01) # nudge forwards
                    pydirectinput.keyUp('w')
                    if playerCenter[0]-mouseRel[0] > 0: # if player is to the left
                        playerLock = 2 # lock player with right side noted (will check right side first when lcok is lost)
                    else:
                        playerLock = 1
                else:
                    pydirectinput.keyDown('w')
                    time.sleep(0.5) # walk forwards for longer
                    pydirectinput.keyUp('w')
                    # don't lock player

            else: # if player is locked
                if playerCenter[0]-mouseRel[0] > 0: # if player is to the left
                    playerLock = 2 # lock player with right side noted
                else:
                    playerLock = 1
                if playerBox[1][1] - playerBox[0][1] < 200: # if player is too far away
                    playerLock = 0 # try tracking the player again

        else:
            # print("Player is higher up.")
            pydirectinput.move(playerCenter[0]-mouseRel[0], 0)
            pydirectinput.keyDown('w')
            pydirectinput.keyDown(' ')
            time.sleep(0.01)
            pydirectinput.keyUp(' ')
            pydirectinput.keyUp('w')
            playerLock = 0

    else:
        if playerLock == 1: # check left first
            playerLock = 0
            pydirectinput.move(-400, 0)
        else: # check right first
            playerLock = 0
            pydirectinput.move(400, 0)


    return playerLock