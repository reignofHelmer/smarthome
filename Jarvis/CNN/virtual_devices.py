class VirtualLight:
    def __init__(self):
        self.state = "OFF"

    def turn_on(self):
        self.state = "ON"
        print("💡 Light is ON")

    def turn_off(self):
        self.state = "OFF"
        print("💡 Light is OFF")

class VirtualDoorLock:
    def __init__(self):
        self.locked = True

    def unlock(self):
        self.locked = False
        print("🔓 Door UNLOCKED")

    def lock(self):
        self.locked = True
        print("🔒 Door LOCKED")