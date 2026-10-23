#!/bin/sh

set -eu

BITRATE="${CAN_BITRATE:-1000000}"

# Authenticate once before changing any network interface. The sysfs type check
# ensures that an interface whose name merely starts with "can" is ignored.
if [ "$(id -u)" -ne 0 ]; then
    exec sudo -- /bin/sh "$0" "$@"
fi

can_interfaces=""
for interface_path in /sys/class/net/*; do
    [ -f "$interface_path/type" ] || continue
    [ "$(cat "$interface_path/type")" = "280" ] || continue

    interface_name=${interface_path##*/}
    can_interfaces="$can_interfaces $interface_name"
done

if [ -z "$can_interfaces" ]; then
    echo "No kernel CAN interfaces found." >&2
    exit 1
fi

echo "Detected CAN interfaces:$can_interfaces"

for interface_name in $can_interfaces; do
    echo "Resetting CAN interface: $interface_name"
    ip link set dev "$interface_name" down
    ip link set dev "$interface_name" type can bitrate "$BITRATE"
    ip link set dev "$interface_name" up
done

echo "All CAN interfaces have been reset at ${BITRATE} bit/s."
