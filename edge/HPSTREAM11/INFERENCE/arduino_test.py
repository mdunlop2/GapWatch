import pyfirmata
import time

board = pyfirmata.Arduino('/dev/ttyACM0')

while True:
    try:
        board.digital[13].write(1)
        time.sleep(1)
        board.digital[13].write(0)
        time.sleep(1)

    except KeyboardInterrupt:
        print("Ending")
        break