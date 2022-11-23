from boardcomms.coral_coms.coms_py.coral_pb_out import send_dx_dy
from periphery import I2C

i2c = I2C("/dev/i2c-3")

while True:
    dx = int(input('dx: '))
    dy = int(input('dy: '))
    print(f'sending dx = {dx}, dy = {dy} to motors..\n')

    send_dx_dy(dx, dy, i2c)