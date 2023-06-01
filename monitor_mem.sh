#!/bin/bash

# Set the command to run
command="deepspeed --num_gpus 1 T0_3B_12GB.py"

# Start the process in the background
$command &

# Retrieve the process ID (PID) of the running process
pid=$!

# Continuously monitor the memory usage and process status concurrently
while true; do
    # Check if the process is still running
    if ! ps -p $pid > /dev/null; then
        echo "Process has terminated."
        break
    fi

    # Get the memory usage of the process
    mem=$(cat "/proc/$pid/status" | grep -e VmSwap -e VmRSS | awk '{print $2}')

    # Format the memory usage in human-readable format
    mem_human=$(numfmt --to=iec --suffix=B --format='%.1f' <<< "$mem")

    # Print the memory usage and current timestamp to aa.txt
    echo "$(date +'%Y-%m-%d %H:%M:%S') $mem_human" >> mem_log.txt

    # Sleep for 5 seconds
    sleep 5
done
