#!/bin/bash

# Configuration
SSID="CSI_NET"
PASSWORD="1234567890"
IFACE="wlan0" # Change this if your interface is different (wlpXsY)

# Check if nmcli is installed
if ! command -v nmcli &> /dev/null; then
    echo "Error: nmcli is not installed. Please install NetworkManager."
    exit 1
fi

echo "Scanning for interface..."
# Auto-detect interface if default wlan0 not found
if ! ip link show "$IFACE" &> /dev/null; then
    DETECTED=$(nmcli device | grep wifi | awk '{print $1}' | head -n 1)
    if [ -n "$DETECTED" ]; then
        IFACE=$DETECTED
        echo "Auto-detected Wi-Fi interface: $IFACE"
    else
        echo "Error: No Wi-Fi interface found."
        exit 1
    fi
fi

echo "Creating Hotspot '$SSID' on interface $IFACE..."

# Delete existing connection if it exists to avoid duplicates
nmcli connection delete "$SSID" &> /dev/null

# Create new connection
# 802-11-wireless.mode ap = Access Point
# ipv4.method shared = Share internet (NAT/DHCP)
nmcli con add type wifi ifname "$IFACE" con-name "$SSID" \
    ssid "$SSID" \
    802-11-wireless.mode ap \
    802-11-wireless.band bg \
    ipv4.method shared \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk "$PASSWORD"

if [ $? -eq 0 ]; then
    echo "Hotspot '$SSID' created successfully."
    echo "Bringing up connection..."
    nmcli con up "$SSID"
    
    echo "========================================"
    echo "Access Point Status:"
    nmcli con show "$SSID" | grep -E "ipv4.addresses|GENERAL.STATE"
    echo "========================================"
    echo "You can now connect your ESP32 to SSID: $SSID with Pass: $PASSWORD"
else
    echo "Error: Failed to create hotspot."
    exit 1
fi
