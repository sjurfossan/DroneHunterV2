
import RPi.GPIO as GPIO
import time
import threading
from gpiozero import MCP3008

# -----------------------------
# Button Setup (using RPi.GPIO)
# -----------------------------
BUTTON1_PIN = 17
BUTTON2_PIN = 27
BUTTON3_PIN = 22

def button_callback(channel):
    if channel == BUTTON1_PIN:
        print("Button 1 pressed")
    elif channel == BUTTON2_PIN:
        print("Button 2 pressed")
    elif channel == BUTTON3_PIN:
        print("Button 3 pressed")

# Use BCM numbering
GPIO.setmode(GPIO.BCM)

# Set up each button pin as an input with an internal pull-up resistor
GPIO.setup(BUTTON1_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON2_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(BUTTON3_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Add event detection on falling edge (button press) with debounce
GPIO.add_event_detect(BUTTON1_PIN, GPIO.FALLING, callback=button_callback, bouncetime=200)
GPIO.add_event_detect(BUTTON2_PIN, GPIO.FALLING, callback=button_callback, bouncetime=200)
GPIO.add_event_detect(BUTTON3_PIN, GPIO.FALLING, callback=button_callback, bouncetime=200)

# ---------------------------------
# Potentiometer Setup (using MCP3008)
# ---------------------------------
# Create two MCP3008 objects: one for each potentiometer channel
pot1 = MCP3008(channel=0)  # Potentiometer 1 on CH0
pot2 = MCP3008(channel=1)  # Potentiometer 2 on CH1

def read_pots():
    """
    Continuously reads both potentiometers (0.0 to 1.0),
    maps the values to integers between 0 and 100,
    and prints them when a significant change is detected.
    """
    last_value1 = None
    last_value2 = None
    threshold = 2  # Only print if the change is at least 2 units
    while True:
        # Read and map the analog values (0-1 scaled to 0-100)
        current_value1 = int(pot1.value * 100)
        current_value2 = int(pot2.value * 100)
        
        # Check for significant change on potentiometer 1
        if last_value1 is None or abs(current_value1 - last_value1) >= threshold:
            print("Potentiometer 1 value:", current_value1)
            last_value1 = current_value1
        
        # Check for significant change on potentiometer 2
        if last_value2 is None or abs(current_value2 - last_value2) >= threshold:
            print("Potentiometer 2 value:", current_value2)
            last_value2 = current_value2
        
        time.sleep(0.1)  # Adjust sleep time as needed

# Start the potentiometer reading in a separate daemon thread
pot_thread = threading.Thread(target=read_pots, daemon=True)
pot_thread.start()

# -----------------------------
# Main Program Loop
# -----------------------------
try:
    while True:
        time.sleep(1)  # Main loop does nothing; work is done in callbacks and the pot thread
except KeyboardInterrupt:
    print("Exiting program")
finally:
    GPIO.cleanup()
