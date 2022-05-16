from pynput.keyboard import Key, Controller
# import human_pose_estimation.inference

class KeyboardSim:
    def __init__(self, id):
        self.id = id
        self.keyboard = Controller()

    def startProcess(self):
        key = "a"

        try:
            while True:
                self.keyboard.press(key)
                self.keyboard.release(key)
                pass #Do something

        except KeyboardInterrupt:
            pass

if __name__ == "__main__":
    kbs = KeyboardSim(1)
    kbs.startProcess()

    print("keyboard interrupted")