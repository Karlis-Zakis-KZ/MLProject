from scapy.all import *
import netaddr
import random

# Suppress Scapy warnings
conf.verb = 0

# Define IP range
network = "192.168.1.0/24"

try:
    # Define a list of source IP addresses
    src_ips = ["192.168.1.102"]

    # Define a list of payload lengths
    payload_lengths = [64, 128, 256, 512, 1024]

    for src_ip in src_ips:
        for payload_length in payload_lengths:
            # Create a payload of the specified length
            payload = "X" * payload_length

            # Send an ICMP request to each host in the network
            ans, unans = sr(IP(dst=network, src=src_ip)/ICMP()/payload, timeout=0.5)

            # Print out the IP addresses of the hosts that responded
            for sent, received in ans:
                print(f"Received response from {received.src} to ICMP request from {src_ip} with payload length {payload_length}")

except KeyboardInterrupt:
    print("[*] Exiting....")
    exit(0)