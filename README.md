# TrasherCasher
This is an AI program that cashes the one who trashes. Basically giving them a fine. To run this program on a normal computer, run the following command in the terminal:
```bash
python main.py --no-gpio
```
This disables the distance sensor which is resposible for detecting when the trash goes into the trash bin. To run this on a Raspberry Pi along with HC-SR04 Distance Sensor, remove the `--no-gpio` part from the command.

If you do not want to see GUI elements, use the following command:
```bash
python main.py --no-gui
```
The `--no-gpio` option can be used here as well.
