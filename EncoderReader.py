import RPi.GPIO as GPIO
import time

# Setup GPIO mode
GPIO.setmode(GPIO.BCM)

# Mapping parameters
MIN_VALUE = 0
MAX_VALUE = 100
STEP = 1  # One encoder "click" = one unit (this is easily changeable)

# Encoder 1 GPIO pins
ENC1_CLK = 17
ENC1_DT  = 18
ENC1_SW  = 27

# Encoder 2 GPIO pins
ENC2_CLK = 22
ENC2_DT  = 23
ENC2_SW  = 24

# Global encoder values (starting mid-range for example)
encoder1_value = 50
encoder2_value = 50

# Global state variables for quadrature decoding (2-bit state: bit1=CLK, bit0=DT)
encoder1_prev_state = None
encoder2_prev_state = None

# Transition lookup table:
# Key: (previous_state, current_state)
# Value: +1 for a clockwise step, -1 for counter-clockwise step.
transition_table = {
    (0b00, 0b01): +1,
    (0b01, 0b11): +1,
    (0b11, 0b10): +1,
    (0b10, 0b00): +1,
    (0b00, 0b10): -1,
    (0b10, 0b11): -1,
    (0b11, 0b01): -1,
    (0b01, 0b00): -1,
}

# Setup GPIO pins for both encoders with internal pull-ups
encoder_pins = [ENC1_CLK, ENC1_DT, ENC1_SW, ENC2_CLK, ENC2_DT, ENC2_SW]
for pin in encoder_pins:
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

def update_encoder(encoder, clk_pin, dt_pin, prev_state_var):
    """
    General function to update encoder value using a state-machine approach.
    """
    # Read the current state from the two channels (forming a 2-bit number)
    current_state = (GPIO.input(clk_pin) << 1) | GPIO.input(dt_pin)
    
    # If no previous state exists, initialize it and return (first call)
    if prev_state_var[0] is None:
        prev_state_var[0] = current_state
        return 0  # no change

    transition = (prev_state_var[0], current_state)
    movement = transition_table.get(transition, 0)  # Default to 0 if transition is invalid

    # Update previous state
    prev_state_var[0] = current_state
    return movement

def encoder1_callback(channel):
    global encoder1_value
    # Using a mutable container (list) to hold the previous state for encoder 1
    movement = update_encoder(1, ENC1_CLK, ENC1_DT, encoder1_prev_state_container)
    if movement != 0:
        encoder1_value += movement * STEP
        encoder1_value = max(MIN_VALUE, min(MAX_VALUE, encoder1_value))
        print(f"Encoder 1 value: {encoder1_value}")

def encoder2_callback(channel):
    global encoder2_value
    movement = update_encoder(2, ENC2_CLK, ENC2_DT, encoder2_prev_state_container)
    if movement != 0:
        encoder2_value += movement * STEP
        encoder2_value = max(MIN_VALUE, min(MAX_VALUE, encoder2_value))
        print(f"Encoder 2 value: {encoder2_value}")

def button_callback(channel):
    if channel == ENC1_SW:
        print("Encoder 1 button pressed")
    elif channel == ENC2_SW:
        print("Encoder 2 button pressed")

# Using a mutable container (a one-element list) to hold the previous state
encoder1_prev_state_container = [None]
encoder2_prev_state_container = [None]

# Instead of just monitoring one channel, add event detection on both CLK and DT for each encoder
GPIO.add_event_detect(ENC1_CLK, GPIO.BOTH, callback=encoder1_callback, bouncetime=2)
GPIO.add_event_detect(ENC1_DT, GPIO.BOTH, callback=encoder1_callback, bouncetime=2)

GPIO.add_event_detect(ENC2_CLK, GPIO.BOTH, callback=encoder2_callback, bouncetime=2)
GPIO.add_event_detect(ENC2_DT, GPIO.BOTH, callback=encoder2_callback, bouncetime=2)

# Setup button events (FALLING edge triggers when the button is pressed)
GPIO.add_event_detect(ENC1_SW, GPIO.FALLING, callback=button_callback, bouncetime=200)
GPIO.add_event_detect(ENC2_SW, GPIO.FALLING, callback=button_callback, bouncetime=200)

print("Rotary encoder demo running. Press CTRL+C to exit.")

try:
    while True:
        time.sleep(0.001)  # Short sleep to keep the main loop responsive

except KeyboardInterrupt:
    print("Exiting program.")

finally:
    GPIO.cleanup()
