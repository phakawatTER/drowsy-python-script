import time
import Jetson.GPIO as GPIO
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)


class beep:
    def __init__(self,interval=3,beep_duration=0.5):
        self.buzzer_pin = 31 # PIN31 of Jetson Nano Board
        self.led_pin = 11 # PIN32 of Jetson Nano Board
#        self.output_pins = [self.buzzer_pin]
        self.output_pins = [self.buzzer_pin,self.led_pin]
        self.force_stop = False # a flag use to stop beeping
        self.interval = interval
        self.beep_duration = beep_duration
        GPIO.setup(self.buzzer_pin,GPIO.IN)
        GPIO.setup(self.output_pins,GPIO.OUT,initial=GPIO.HIGH)
    def beep(self):
        GPIO.output(self.output_pins,GPIO.LOW)
        time.sleep(self.beep_duration)
        GPIO.output(self.output_pins,GPIO.HIGH)
        time.sleep(self.beep_duration/2)
    def play_beep_loop(self):
        start_beep_time = time.time()
        while True:
            try:
                if int(time.time() - start_beep_time) >= self.interval or self.force_stop:
                    break
                self.beep()
            except KeyboardInterrupt:
#                print("Keyboard Interrupt Raised!")
                break

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-d","--duration",required=False,default=0.40,type=float,help="beep duration")
    args = vars(ap.parse_args())
    b = beep(interval=200000,beep_duration=args["duration"])
    b.play_beep_loop()
    GPIO.cleanup()

