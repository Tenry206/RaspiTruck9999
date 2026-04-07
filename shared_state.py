import threading

class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.frame = None
        self.steering = (0,0)
        self.override = None #for action like STOP or SPIN safely
        self.current_shape = ""

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame
    
    def get_frame(self):
        with self.lock:
            return self.frame
        
    def set_steering(self, left, right):
        with self.lock:
            self.steering = (left, right)

    def get_steering(self):
        with self.lock:
            return self.steering
    
    def set_override(self, action):
        with self.lock:
            self.override = action
    
    def get_override(self):
        with self.lock:
            return self.override
        
   
        