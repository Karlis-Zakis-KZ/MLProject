import pyshark
import csv

# Create a live capture object
capture = pyshark.LiveCapture(interface='Ethernet')

# Open a CSV file for writing
with open('ddos_data.csv', 'w', newline='') as csvfile:
    # Create a CSV writer
    writer = csv.writer(csvfile)

    # Define a header for the CSV file
    headers_written = False

    # Start the capture
    for packet in capture.sniff_continuously():
        # Create a dictionary to hold the relevant packet data
        packet_dict = {
            'timestamp': packet.sniff_time,
            'packet_length': packet.length
        }

        # Check for Ethernet layer and extract relevant fields
        if 'eth' in packet:
            packet_dict['src_mac'] = packet.eth.src
            packet_dict['dst_mac'] = packet.eth.dst

        # Check for IP layer and extract relevant fields
        if 'ip' in packet:
            packet_dict['src_ip'] = packet.ip.src
            packet_dict['dst_ip'] = packet.ip.dst
            packet_dict['ip_ttl'] = packet.ip.ttl

        # Check for TCP layer and extract relevant fields
        if 'tcp' in packet:
            packet_dict['src_port'] = packet.tcp.srcport
            packet_dict['dst_port'] = packet.tcp.dstport
            packet_dict['tcp_flags'] = packet.tcp.flags

        # Check for UDP layer and extract relevant fields
        if 'udp' in packet:
            packet_dict['src_port'] = packet.udp.srcport
            packet_dict['dst_port'] = packet.udp.dstport

        # Check for ICMP layer and extract relevant fields
        if 'icmp' in packet:
            packet_dict['icmp_type'] = packet.icmp.type
            packet_dict['icmp_code'] = packet.icmp.code

        # Write headers if not already written
        if not headers_written:
            writer.writerow(packet_dict.keys())
            headers_written = True

        # Write the packet data to the CSV file
        writer.writerow(packet_dict.values())
