#!/bin/bash
set -e
cd "$(dirname "$0")"
sudo install -m 644 99-lerobot-serial.rules /etc/udev/rules.d/99-lerobot-serial.rules
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=tty --action=change
sleep 1
ls -l /dev/arm_* /dev/base_neck
