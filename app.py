import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from collections import deque

# --- Page Config ---
st.set_page_config(
    page_title="OS Algorithm Simulator",
    layout='wide',
    page_icon="🧠",
    initial_sidebar_state='collapsed'
)

# --- Custom CSS ---
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&family=Source+Code+Pro&display=swap');
        
        :root {
            --primary-color: #9b87f5;
            --secondary-color: #6c5ce7;
            --accent-color: #a29bfe;
            --text-color: #2c3e50;
            --bg-color: black;
            --card-bg: #ffffff;
        }
        
        .stApp {
            background-color: var(--bg-color);
        }
        
        h1, h2, h3, h4 {
            font-family: 'Roboto', sans-serif;
            font-weight: 700;
            color: var(--text-color);
        }
        
        .metric-card {
            border: 1px solid #e9ecef;
            border-radius: 10px;
            padding: 1.5rem;
            background-color: white;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.03);
            transition: transform 0.3s ease;
        }
        
        .memory-block {
            margin-bottom: 12px;
            padding: 10px;
            border-radius: 8px;
            background-color: #f8f9fa;
            transition: transform 0.2s ease;
        }
        
        .disk-visualization {
            margin: 20px 0;
            padding: 15px;
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# --- Utility Functions ---
def parse_cpu_textbox(data, include_priority=False):
    processes = []
    try:
        for line in data.strip().split('\n'):
            parts = line.split()
            if len(parts) >= 3:
                pid, at, bt = parts[0], parts[1], parts[2]
                process = {'pid': pid, 'at': int(at), 'bt': int(bt)}
                if include_priority and len(parts) >= 4:
                    process['priority'] = int(parts[3])
                processes.append(process)
    except Exception as e:
        st.error(f"Error parsing input: {e}")
    return processes

def parse_memory_textbox(block_text, proc_text):
    blocks, processes = [], []
    
    # Parse memory blocks
    try:
        for line in block_text.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 2:
                bid, size = parts[0], parts[1]
                blocks.append({'id': bid, 'size': int(size), 'remaining': int(size), 'allocated_to': None})
    except Exception as e:
        st.error(f"Error parsing memory blocks: {e}")
        
    # Parse processes
    try:
        for line in proc_text.strip().split('\n'):
            parts = line.strip().split()
            if len(parts) >= 2:
                pid, size = parts[0], parts[1]
                processes.append({'id': pid, 'size': int(size)})
    except Exception as e:
        st.error(f"Error parsing processes: {e}")
        
    return blocks, processes

def parse_disk_textbox(data):
    try:
        parts = [int(x.strip()) for x in data.split(',')]
        return parts
    except Exception as e:
        st.error(f"Error parsing disk requests: {e}")
        return []

# --- CPU Scheduling Algorithms ---
def fcfs(processes):
    if not processes:
        st.error("No valid processes provided")
        return [], []
        
    processes_copy = processes.copy()
    processes_copy.sort(key=lambda p: p['at'])
    time_ = 0
    gantt_chart, results = [], []
    
    for p in processes_copy:
        start = max(time_, p['at'])
        end = start + p['bt']
        
        # Calculate metrics
        waiting_time = start - p['at']
        turnaround_time = end - p['at']
        
        results.append({
            'pid': p['pid'], 
            'at': p['at'], 
            'bt': p['bt'], 
            'st': start, 
            'ct': end,
            'wt': waiting_time, 
            'tat': turnaround_time, 
            'rt': waiting_time
        })
        
        gantt_chart.append({'pid': p['pid'], 'start': start, 'end': end})
        time_ = end
        
    return gantt_chart, results

def sjf(processes, preemptive=False):
    if not processes:
        st.error("No valid processes provided")
        return [], []
        
    processes_copy = sorted([p.copy() for p in processes], key=lambda x: (x['at'], x['bt']))
    completed, time_ = 0, 0
    n = len(processes_copy)
    gantt_chart, ready_queue, results = [], [], []
    
    # Initialize process attributes
    for p in processes_copy:
        p.update({'remaining': p['bt'], 'st': None})
        
    # Main scheduling loop
    while completed < n:
        # Add newly arrived processes to ready queue
        for p in processes_copy:
            if p['at'] <= time_ and p not in ready_queue and p['remaining'] > 0:
                ready_queue.append(p)
                
        # Handle idle time when no process is ready
        if not ready_queue:
            time_ += 1
            continue
            
        # Sort ready queue by remaining burst time
        ready_queue.sort(key=lambda x: x['remaining'])
        current = ready_queue[0]
        
        # Set start time if not already set
        if current['st'] is None:
            current['st'] = time_
            
        # Preemptive SJF (SRTF)
        if preemptive:
            current['remaining'] -= 1
            
            # Update Gantt chart
            if not gantt_chart or gantt_chart[-1]['pid'] != current['pid']:
                gantt_chart.append({'pid': current['pid'], 'start': time_, 'end': time_ + 1})
            else:
                gantt_chart[-1]['end'] += 1
                
            time_ += 1
            
            # Process completion
            if current['remaining'] == 0:
                current.update({
                    'ct': time_,
                    'tat': time_ - current['at'],
                    'wt': time_ - current['at'] - current['bt'],
                    'rt': current['st'] - current['at']
                })
                results.append(current.copy())
                ready_queue.remove(current)
                completed += 1
        else:
            # Non-preemptive SJF
            time_ += current['remaining']
            current.update({
                'ct': time_,
                'tat': time_ - current['at'],
                'wt': time_ - current['at'] - current['bt'],
                'rt': current['st'] - current['at'],
                'remaining': 0
            })
            gantt_chart.append({'pid': current['pid'], 'start': current['st'], 'end': time_})
            results.append(current.copy())
            ready_queue.remove(current)
            completed += 1
            
    return gantt_chart, results

def priority_scheduling(processes, preemptive=False):
    if not processes:
        st.error("No valid processes provided")
        return [], []
        
    processes_copy = sorted([p.copy() for p in processes], key=lambda x: (x['at'], x.get('priority', 0)))
    completed, time_ = 0, 0
    n = len(processes_copy)
    gantt_chart, ready_queue, results = [], [], []
    
    # Initialize process attributes
    for p in processes_copy:
        p.update({'remaining': p['bt'], 'st': None})
        
    # Main scheduling loop
    while completed < n:
        # Add newly arrived processes to ready queue
        for p in processes_copy:
            if p['at'] <= time_ and p not in ready_queue and p['remaining'] > 0:
                ready_queue.append(p)
                
        # Handle idle time when no process is ready
        if not ready_queue:
            time_ += 1
            continue
            
        # Sort ready queue by priority (lower number = higher priority)
        ready_queue.sort(key=lambda x: x.get('priority', 0))
        current = ready_queue[0]
        
        # Set start time if not already set
        if current['st'] is None:
            current['st'] = time_
            
        # Preemptive Priority
        if preemptive:
            current['remaining'] -= 1
            
            # Update Gantt chart
            if not gantt_chart or gantt_chart[-1]['pid'] != current['pid']:
                gantt_chart.append({'pid': current['pid'], 'start': time_, 'end': time_ + 1})
            else:
                gantt_chart[-1]['end'] += 1
                
            time_ += 1
            
            # Process completion
            if current['remaining'] == 0:
                current.update({
                    'ct': time_,
                    'tat': time_ - current['at'],
                    'wt': time_ - current['at'] - current['bt'],
                    'rt': current['st'] - current['at']
                })
                results.append(current.copy())
                ready_queue.remove(current)
                completed += 1
        else:
            # Non-preemptive Priority
            time_ += current['remaining']
            current.update({
                'ct': time_,
                'tat': time_ - current['at'],
                'wt': time_ - current['at'] - current['bt'],
                'rt': current['st'] - current['at'],
                'remaining': 0
            })
            gantt_chart.append({'pid': current['pid'], 'start': current['st'], 'end': time_})
            results.append(current.copy())
            ready_queue.remove(current)
            completed += 1
            
    return gantt_chart, results

def round_robin(processes, time_quantum):
    if not processes:
        st.error("No valid processes provided")
        return [], []
        
    processes_copy = sorted([p.copy() for p in processes], key=lambda x: x['at'])
    n = len(processes_copy)
    completed = 0
    time_ = 0
    ready_queue = []
    gantt_chart = []
    results = []
    
    # Initialize process attributes
    for p in processes_copy:
        p.update({'remaining': p['bt'], 'st': None, 'last_executed': None})
    
    while completed < n:
        # Add arriving processes to ready queue
        for p in processes_copy:
            if p['at'] <= time_ and p['remaining'] > 0 and p not in ready_queue and p['last_executed'] is None:
                ready_queue.append(p)
        
        if not ready_queue:
            time_ += 1
            continue
        
        current = ready_queue.pop(0)
        
        # Set start time if not already set
        if current['st'] is None:
            current['st'] = time_
        
        # Execute for time quantum or remaining time, whichever is smaller
        exec_time = min(time_quantum, current['remaining'])
        
        # Update Gantt chart
        gantt_chart.append({'pid': current['pid'], 'start': time_, 'end': time_ + exec_time})
        
        time_ += exec_time
        current['remaining'] -= exec_time
        current['last_executed'] = time_
        
        # Check if process completed
        if current['remaining'] == 0:
            current.update({
                'ct': time_,
                'tat': time_ - current['at'],
                'wt': time_ - current['at'] - current['bt'],
                'rt': current['st'] - current['at']
            })
            results.append(current.copy())
            completed += 1
        else:
            # Re-add arriving processes to ready queue before re-adding current process
            for p in processes_copy:
                if p['at'] <= time_ and p['remaining'] > 0 and p not in ready_queue and p != current and p['last_executed'] is None:
                    ready_queue.append(p)
            ready_queue.append(current)
    
    return gantt_chart, results

# --- Memory Allocation Algorithms ---
def memory_allocation(blocks, processes, strategy):
    """
    Allocate memory blocks to processes using the specified strategy
    
    Args:
        blocks: List of memory blocks (each with id, size, remaining, allocated_to)
        processes: List of processes (each with id, size)
        strategy: 'First_Fit', 'Best_Fit', or 'Worst_Fit'
    
    Returns:
        tuple: (updated blocks, allocation results)
    """
    # Make copies to avoid modifying the original data
    blocks_copy = [b.copy() for b in blocks]
    allocations = []
    
    for process in processes:
        # Find suitable blocks based on strategy
        if strategy == 'First_Fit':
            # First Fit: Allocate to first block that's large enough
            for block in blocks_copy:
                if block['remaining'] >= process['size'] and block['allocated_to'] is None:
                    block['remaining'] -= process['size']
                    block['allocated_to'] = process['id']
                    allocations.append({
                        'process_id': process['id'],
                        'size': process['size'],
                        'allocated_block': block['id'],
                        'internal_fragmentation': block['remaining']
                    })
                    break
            else:
                allocations.append({
                    'process_id': process['id'],
                    'size': process['size'],
                    'allocated_block': 'Not Allocated',
                    'internal_fragmentation': None
                })
                
        elif strategy == 'Best_Fit':
            # Best Fit: Allocate to smallest block that's large enough
            suitable_blocks = [
                b for b in blocks_copy 
                if b['remaining'] >= process['size'] and b['allocated_to'] is None
            ]
            
            if suitable_blocks:
                best_block = min(suitable_blocks, key=lambda x: x['remaining'])
                best_block['remaining'] -= process['size']
                best_block['allocated_to'] = process['id']
                allocations.append({
                    'process_id': process['id'],
                    'size': process['size'],
                    'allocated_block': best_block['id'],
                    'internal_fragmentation': best_block['remaining']
                })
            else:
                allocations.append({
                    'process_id': process['id'],
                    'size': process['size'],
                    'allocated_block': 'Not Allocated',
                    'internal_fragmentation': None
                })
                
        elif strategy == 'Worst_Fit':
            # Worst Fit: Allocate to largest available block
            suitable_blocks = [
                b for b in blocks_copy 
                if b['remaining'] >= process['size'] and b['allocated_to'] is None
            ]
            
            if suitable_blocks:
                worst_block = max(suitable_blocks, key=lambda x: x['remaining'])
                worst_block['remaining'] -= process['size']
                worst_block['allocated_to'] = process['id']
                allocations.append({
                    'process_id': process['id'],
                    'size': process['size'],
                    'allocated_block': worst_block['id'],
                    'internal_fragmentation': worst_block['remaining']
                })
            else:
                allocations.append({
                    'process_id': process['id'],
                    'size': process['size'],
                    'allocated_block': 'Not Allocated',
                    'internal_fragmentation': None
                })
    
    return blocks_copy, allocations

# --- Disk Scheduling Algorithms ---
def fcfs_disk_scheduling(requests, initial_position):
    sequence = [initial_position] + requests
    total_movement = sum(abs(sequence[i] - sequence[i-1]) for i in range(1, len(sequence)))
    return sequence, total_movement

def sstf_disk_scheduling(requests, initial_position):
    sequence = [initial_position]
    remaining_requests = requests.copy()
    current_position = initial_position
    total_movement = 0
    
    while remaining_requests:
        # Find the request with the shortest seek time
        closest_request = min(remaining_requests, key=lambda x: abs(x - current_position))
        distance = abs(closest_request - current_position)
        total_movement += distance
        current_position = closest_request
        sequence.append(current_position)
        remaining_requests.remove(current_position)
    
    return sequence, total_movement

def scan_disk_scheduling(requests, initial_position, disk_size=200, direction='up'):
    sequence = [initial_position]
    total_movement = 0
    current_position = initial_position
    remaining_requests = requests.copy()
    
    if direction == 'up':
        # Move to the end of disk
        while current_position < disk_size:
            current_position += 1
            if current_position in remaining_requests:
                sequence.append(current_position)
                remaining_requests.remove(current_position)
                if not remaining_requests:
                    break
        total_movement += disk_size - initial_position
    else:
        # Move to the start of disk
        while current_position > 0:
            current_position -= 1
            if current_position in remaining_requests:
                sequence.append(current_position)
                remaining_requests.remove(current_position)
                if not remaining_requests:
                    break
        total_movement += initial_position
    
    return sequence, total_movement

def c_scan_disk_scheduling(requests, initial_position, disk_size=200):
    sequence = [initial_position]
    total_movement = 0
    current_position = initial_position
    remaining_requests = requests.copy()
    
    # Move to the end of disk
    while current_position < disk_size:
        current_position += 1
        if current_position in remaining_requests:
            sequence.append(current_position)
            remaining_requests.remove(current_position)
            if not remaining_requests:
                break
    
    # Jump to start if there are remaining requests
    if remaining_requests:
        total_movement += disk_size - initial_position
        current_position = 0
        sequence.append(current_position)
        
        # Move up again
        while current_position < disk_size and remaining_requests:
            current_position += 1
            if current_position in remaining_requests:
                sequence.append(current_position)
                remaining_requests.remove(current_position)
    else:
        total_movement += disk_size - initial_position
    
    return sequence, total_movement

def look_disk_scheduling(requests, initial_position, disk_size=200, direction='up'):
    sequence = [initial_position]
    total_movement = 0
    current_position = initial_position
    remaining_requests = requests.copy()
    
    if direction == 'up':
        # Find the highest request
        max_request = max(remaining_requests) if remaining_requests else initial_position
        
        # Move up to max request
        while current_position < max_request:
            current_position += 1
            if current_position in remaining_requests:
                sequence.append(current_position)
                remaining_requests.remove(current_position)
                if not remaining_requests:
                    break
        total_movement += max_request - initial_position
    else:
        # Find the lowest request
        min_request = min(remaining_requests) if remaining_requests else initial_position
        
        # Move down to min request
        while current_position > min_request:
            current_position -= 1
            if current_position in remaining_requests:
                sequence.append(current_position)
                remaining_requests.remove(current_position)
                if not remaining_requests:
                    break
        total_movement += initial_position - min_request
    
    return sequence, total_movement

def c_look_disk_scheduling(requests, initial_position, disk_size=200):
    sequence = [initial_position]
    total_movement = 0
    current_position = initial_position
    remaining_requests = requests.copy()
    
    if not remaining_requests:
        return sequence, 0
    
    # Find the highest and lowest requests
    max_request = max(remaining_requests)
    min_request = min(remaining_requests)
    
    # Move up to max request
    while current_position < max_request:
        current_position += 1
        if current_position in remaining_requests:
            sequence.append(current_position)
            remaining_requests.remove(current_position)
            if not remaining_requests:
                break
    
    # If there are remaining requests, jump to min request
    if remaining_requests:
        total_movement += max_request - initial_position + (max_request - min_request)
        current_position = min_request
        sequence.append(current_position)
        remaining_requests.remove(current_position)
        
        # Move up again
        while current_position < max_request and remaining_requests:
            current_position += 1
            if current_position in remaining_requests:
                sequence.append(current_position)
                remaining_requests.remove(current_position)
    else:
        total_movement += max_request - initial_position
    
    return sequence, total_movement

# --- Display CPU Scheduling Results ---
def display_cpu_results(gantt_data, results_data):
    # Calculate metrics
    avg_wt = sum(p['wt'] for p in results_data) / len(results_data) if results_data else 0
    avg_tat = sum(p['tat'] for p in results_data) / len(results_data) if results_data else 0
    avg_rt = sum(p['rt'] for p in results_data) / len(results_data) if results_data else 0
    
    # Display metrics in attractive cards
    metrics_cols = st.columns(3)
    with metrics_cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Avg. Waiting Time</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{int(avg_wt) if avg_wt.is_integer() else avg_wt:.2f}</h2>
            <p style="color:#666;margin:0">time units</p>
        </div>
        """, unsafe_allow_html=True)
    
    with metrics_cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Avg. Turnaround Time</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{int(avg_tat) if avg_tat.is_integer() else avg_tat:.2f}</h2>
            <p style="color:#666;margin:0">time units</p>
        </div>
        """, unsafe_allow_html=True)
        
    with metrics_cols[2]:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Avg. Response Time</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{int(avg_rt) if avg_rt.is_integer() else avg_rt:.2f}</h2>
            <p style="color:#666;margin:0">time units</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gantt Chart with matplotlib
    st.subheader("⏱️ Gantt Chart")
    
    if gantt_data:
        # Create a color mapping for processes
        colors = {p['pid']: get_random_color() for p in gantt_data}
        
        fig, ax = plt.subplots(figsize=(10, 2))
        
        # Draw the Gantt bars
        for p in gantt_data:
            ax.barh(0, p['end'] - p['start'], left=p['start'], 
                   color=colors[p['pid']], edgecolor='black')
            
            # Add process ID labels
            bar_center = (p['start'] + p['end']) / 2
            ax.text(bar_center, 0, p['pid'], 
                   ha='center', va='center', color='white', fontweight='bold')
        
        # Customize the chart appearance
        ax.set_yticks([])
        ax.set_xlabel("Time Units")
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Display the chart
        st.pyplot(fig)
        
        # Results Table
        st.subheader("📊 Process Table")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(results_data)
        df = df.sort_values('pid')
        
        # Convert all numeric columns to integers
        numeric_cols = ['at', 'bt', 'st', 'ct', 'wt', 'tat', 'rt']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Style the dataframe
        styled_df = df.style.format({
            'wt': '{:d}',
            'tat': '{:d}',
            'rt': '{:d}'
        })
        
        st.dataframe(styled_df, use_container_width=True)

# --- Display Memory Allocation Results ---
def display_memory_results(blocks, allocations):
    # Get total memory size and allocated memory
    total_memory = sum(b['size'] for b in blocks)
    allocated_memory = sum(b['size'] - b['remaining'] for b in blocks)
    allocation_percent = (allocated_memory / total_memory * 100) if total_memory > 0 else 0
    
    # Calculate fragmentation
    internal_fragmentation = sum(b['remaining'] for b in blocks if b['allocated_to'] is not None)
    
    # Memory utilization metrics
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Memory Utilization</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{allocation_percent:.1f}%</h2>
            <p style="color:#666;margin:0">{allocated_memory} / {total_memory} KB</p>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Internal Fragmentation</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{internal_fragmentation} KB</h2>
            <p style="color:#666;margin:0">wasted memory</p>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[2]:
        allocated_processes = sum(1 for a in allocations if a['allocated_block'] != 'Not Allocated')
        total_processes = len(allocations)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Allocation Rate</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{allocated_processes}/{total_processes}</h2>
            <p style="color:#666;margin:0">processes allocated</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Block Allocation Visualization
    st.subheader("💾 Memory Blocks")
    
    for b in blocks:
        used = b['size'] - b['remaining']
        percent = (used / b['size']) * 100 if b['size'] > 0 else 0
        
        if b['allocated_to']:
            bar_color = "#6c5ce7"  # Purple for allocated
            status = f"Allocated to: {b['allocated_to']}"
        else:
            bar_color = "#74b9ff"  # Blue for free
            status = "Free"
            
        st.markdown(f"""
        <div class="memory-block">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                    <strong>Block {b['id']}</strong> ({b['size']} KB)
                </div>
                <div style="color:#666">
                    {status}
                </div>
            </div>
            <div style="background:#e9ecef;height:30px;border-radius:6px;margin-top:5px;overflow:hidden">
                <div style="width:{percent}%;height:100%;background:{bar_color};display:flex;align-items:center;justify-content:flex-end;padding-right:10px;color:white">
                    {used} KB
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Process Allocation Table
    st.subheader("📋 Process Allocation")
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(allocations)
    
    # Style the dataframe
    def highlight_allocation(val):
        if val == 'Not Allocated':
            return 'background-color: #ffcccb'
        return 'background-color: #d1f0c2'
    
    styled_df = df.style.applymap(highlight_allocation, subset=['allocated_block'])
    
    st.dataframe(styled_df, use_container_width=True)

# --- Display Disk Scheduling Results ---
def display_disk_results(requests, initial_position, sequence, total_movement, algorithm_name):
    # Display metrics
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Total Head Movement</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{total_movement}</h2>
            <p style="color:#666;margin:0">cylinders</p>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[1]:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Requests Processed</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{len(requests)}</h2>
            <p style="color:#666;margin:0">disk requests</p>
        </div>
        """, unsafe_allow_html=True)
        
    with cols[2]:
        avg_seek = total_movement / len(requests) if requests else 0
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color:#6c5ce7;margin:0">Avg. Seek Time</h3>
            <h2 style="margin:10px 0;font-size:2.2rem">{avg_seek:.1f}</h2>
            <p style="color:#666;margin:0">cylinders per request</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Disk Visualization
    st.subheader("💽 Disk Head Movement")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot the sequence
    x = range(len(sequence))
    ax.plot(x, sequence, marker='o', color='#6c5ce7', linewidth=2, markersize=8)
    
    # Highlight initial position
    ax.plot(0, sequence[0], marker='o', color='#e17055', markersize=10, label='Initial Position')
    
    # Highlight requests
    request_indices = [sequence.index(r) for r in requests if r in sequence]
    ax.plot(request_indices, requests, 'o', color='#00b894', markersize=8, label='Requests')
    
    # Customize the plot
    ax.set_xlabel('Step')
    ax.set_ylabel('Cylinder Number')
    ax.set_title(f'{algorithm_name} Disk Scheduling')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Display the plot
    st.pyplot(fig)
    
    # Sequence Table
    st.subheader("📋 Movement Sequence")
    
    # Create a DataFrame for the sequence
    df = pd.DataFrame({
        'Step': range(len(sequence)),
        'Cylinder': sequence,
        'Movement': [abs(sequence[i] - sequence[i-1]) if i > 0 else 0 for i in range(len(sequence))]
    })
    
    # Calculate cumulative movement
    df['Cumulative Movement'] = df['Movement'].cumsum()
    
    # Style the DataFrame
    def highlight_row(row):
        if row['Step'] == 0:
            return ['background-color: #ffeaa7'] * len(row)
        return [''] * len(row)
    
    styled_df = df.style.apply(highlight_row, axis=1)
    st.dataframe(styled_df, use_container_width=True)

def get_random_color():
    # Generate vibrant colors for better visualization
    h = random.random()
    s = 0.7 + random.random() * 0.3  # Higher saturation
    v = 0.8 + random.random() * 0.2  # Higher value
    
    # Convert HSV to RGB
    i = int(h * 6)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if i % 6 == 0:
        r, g, b = v, t, p
    elif i % 6 == 1:
        r, g, b = q, v, p
    elif i % 6 == 2:
        r, g, b = p, v, t
    elif i % 6 == 3:
        r, g, b = p, q, v
    elif i % 6 == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    
    # Convert to hex
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

# --- Initialize session state variables ---
if 'category' not in st.session_state:
    st.session_state.category = None
    
if 'num_proc' not in st.session_state:
    st.session_state.num_proc = 4
    
if 'num_mem_proc' not in st.session_state:
    st.session_state.num_mem_proc = 5
    
if 'num_blocks' not in st.session_state:
    st.session_state.num_blocks = 4
    
if 'num_disk_req' not in st.session_state:
    st.session_state.num_disk_req = 8
    
# Initialize default data
if 'cpu_data' not in st.session_state:
    st.session_state.cpu_data = '\n'.join([f"P{i+1} {i} {random.randint(1, 5)}" for i in range(st.session_state.num_proc)])
    
if 'memory_blocks' not in st.session_state:
    memory_sizes = [100, 500, 200, 300, 600]
    st.session_state.memory_blocks = '\n'.join([f"B{i+1} {memory_sizes[i % len(memory_sizes)]}" for i in range(st.session_state.num_blocks)])
    
if 'memory_procs' not in st.session_state:
    proc_sizes = [80, 200, 150, 300, 120]
    st.session_state.memory_procs = '\n'.join([f"P{i+1} {proc_sizes[i % len(proc_sizes)]}" for i in range(st.session_state.num_mem_proc)])

if 'disk_requests' not in st.session_state:
    st.session_state.disk_requests = ', '.join(str(random.randint(0, 199)) for _ in range(st.session_state.num_disk_req))

if 'initial_position' not in st.session_state:
    st.session_state.initial_position = 50

# --- Main App Logic ---
st.markdown("""
<div style="background:linear-gradient(to right,#9b87f5,#6c5ce7);padding:1.5rem;border-radius:12px;margin-bottom:2rem;box-shadow:0 4px 12px rgba(0,0,0,0.05);color:white;text-align:center;">
    <h1>🧠 OS Algorithm Simulator</h1>
    <p>Visualize and understand operating system algorithms through interactive simulations</p>
</div>
""", unsafe_allow_html=True)

# Main category selector
category_options = [None, "CPU Scheduling", "Memory Management", "Disk Scheduling"]
display_options = ["Select a category", "🖥️ CPU Scheduling", "💾 Memory Management", "💽 Disk Scheduling"]
category_mapping = {display: value for display, value in zip(display_options, category_options)}
selected_display = st.selectbox("Select Category", display_options, index=0)
category = category_mapping[selected_display]
st.session_state.category = category

# Sidebar content
with st.sidebar:
    st.markdown("""
    ### 🧠 OS Simulator
    
    This simulator helps you visualize operating system algorithms:
    
    - **CPU Scheduling**
      - First Come First Serve (FCFS)
      - Shortest Job First (SJF)
      - Shortest Remaining Time First (SRTF)
      - Priority Scheduling
      - Round Robin
    
    - **Memory Management**
      - First Fit
      - Best Fit
      - Worst Fit
      
    - **Disk Scheduling**
      - FCFS
      - SSTF
      - SCAN
      - C-SCAN
      - LOOK
      - C-LOOK
    """)

# CPU Scheduling Section
if st.session_state.category == "CPU Scheduling":
    st.markdown("## CPU Scheduling Algorithms")
    
    algo = st.selectbox(
        "Select Algorithm",
        [
            "First Come First Serve (FCFS)",
            "Shortest Job First (Non-preemptive)",
            "Shortest Remaining Time First (Preemptive)",
            "Priority Scheduling (Non-preemptive)",
            "Priority Scheduling (Preemptive)",
            "Round Robin"
        ],
        index=0
    )
    
    with st.expander("Configure Processes", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col2:
            num_proc = st.number_input(
                "Number of Processes", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.num_proc, 
                step=1,
                key="process_count_input"
            )
            
            if num_proc != st.session_state.num_proc:
                st.session_state.num_proc = num_proc
                # Generate data without priority for Round Robin
                if algo == "Round Robin":
                    st.session_state.cpu_data = '\n'.join([f"P{i+1} {i} {random.randint(1, 5)}" for i in range(st.session_state.num_proc)])
                else:
                    st.session_state.cpu_data = '\n'.join([f"P{i+1} {i} {random.randint(1, 5)} {random.randint(1, 5)}" for i in range(st.session_state.num_proc)])
                st.rerun()
            
            if algo == "Round Robin":
                time_quantum = st.number_input(
                    "Time Quantum",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1
                )
        
        with col1:
            if "Priority" in algo:
                st.markdown("""
                **Enter processes in format: `PID AT BT Priority`**
                - `PID`: Process ID
                - `AT`: Arrival Time
                - `BT`: Burst Time
                - `Priority`: Priority (lower number = higher priority)
                - `Note`:Enter each process on a new line
                """)
            else:
                st.markdown("""
                **Enter processes in format: `PID AT BT`**
                - `PID`: Process ID
                - `AT`: Arrival Time
                - `BT`: Burst Time
                - `Note`:Enter each process on a new line
                """)
                
            data = st.text_area(
                "Process Data",
                value=st.session_state.cpu_data,
                height=150,
                help="Enter each process on a new line",
                key="cpu_data_input"
            )
            
            st.session_state.cpu_data = data
    
    col1, col2 = st.columns([1, 4])
    with col1:
        simulate = st.button("▶️ Simulate", use_container_width=True, type='secondary')
    
    if simulate:
        with st.spinner("Running simulation..."):
            include_priority = "Priority" in algo
            processes = parse_cpu_textbox(data, include_priority)
            
            if not processes:
                st.error("Please enter valid process data")
            else:
                if "FCFS" in algo:
                    gantt, table = fcfs(processes)
                elif "Non-preemptive" in algo and "Priority" not in algo:
                    gantt, table = sjf(processes, preemptive=False)
                elif "Preemptive" in algo and "Priority" not in algo:
                    gantt, table = sjf(processes, preemptive=True)
                elif "Non-preemptive" in algo and "Priority" in algo:
                    gantt, table = priority_scheduling(processes, preemptive=False)
                elif "Preemptive" in algo and "Priority" in algo:
                    gantt, table = priority_scheduling(processes, preemptive=True)
                elif algo == "Round Robin":
                    gantt, table = round_robin(processes, time_quantum)
                
                if gantt and table:
                    display_cpu_results(gantt, table)
                    st.success(f"Simulation completed successfully!")
                else:
                    st.error("Simulation failed. Please check your input data.")

# Memory Management Section
elif st.session_state.category == "Memory Management":
    st.markdown("## Memory Allocation Algorithms")
    
    algo = st.selectbox("Select Algorithm", ["First Fit", "Best Fit", "Worst Fit"], index=0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Memory Blocks")
        
        with st.expander("Configure Memory Blocks", expanded=True):
            nb = st.number_input(
                "Number of Blocks", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.num_blocks, 
                step=1,
                key="blocks_count_input"
            )
            
            if nb != st.session_state.num_blocks:
                st.session_state.num_blocks = nb
                memory_sizes = [100, 500, 200, 300, 600]
                st.session_state.memory_blocks = '\n'.join([
                    f"B{i+1} {memory_sizes[i % len(memory_sizes)]}" 
                    for i in range(st.session_state.num_blocks)
                ])
                st.rerun()
                
            block_text = st.text_area(
                "Memory Blocks (ID SIZE)",
                value=st.session_state.memory_blocks,
                height=150,
                help="Enter each memory block on a new line: ID SIZE",
                key="memory_blocks_input"
            )
            
            st.session_state.memory_blocks = block_text
    
    with col2:
        st.markdown("#### Processes")
        
        with st.expander("Configure Processes", expanded=True):
            np_mem = st.number_input(
                "Number of Processes", 
                min_value=1, 
                max_value=10, 
                value=st.session_state.num_mem_proc, 
                step=1,
                key="mem_procs_count_input"
            )
            
            if np_mem != st.session_state.num_mem_proc:
                st.session_state.num_mem_proc = np_mem
                proc_sizes = [80, 200, 150, 300, 120]
                st.session_state.memory_procs = '\n'.join([
                    f"P{i+1} {proc_sizes[i % len(proc_sizes)]}" 
                    for i in range(st.session_state.num_mem_proc)
                ])
                st.rerun()
                
            proc_text = st.text_area(
                "Processes (ID SIZE)",
                value=st.session_state.memory_procs,
                height=150,
                help="Enter each process on a new line: ID SIZE",
                key="memory_procs_input"
            )
            
            st.session_state.memory_procs = proc_text
    
    col1, col2 = st.columns([1, 4])
    with col1:
        mem_simulate = st.button("▶️ Simulate", use_container_width=True, type='secondary')
    
    if mem_simulate:
        with st.spinner("Running allocation simulation..."):
            blocks, procs = parse_memory_textbox(block_text, proc_text)
            
            if not blocks or not procs:
                st.error("Please enter valid block and process data")
            else:
                strat = algo.replace(" ", "_")
                updated_blocks, allocations = memory_allocation(blocks, procs, strat)
                display_memory_results(updated_blocks, allocations)
                st.success(f"Memory allocation completed successfully!")

# Disk Scheduling Section
elif st.session_state.category == "Disk Scheduling":
    st.markdown("## Disk Scheduling Algorithms")
    
    algo = st.selectbox(
        "Select Algorithm",
        [
            "FCFS (First Come First Serve)",
            "SSTF (Shortest Seek Time First)",
            "SCAN (Elevator)",
            "C-SCAN (Circular SCAN)",
            "LOOK",
            "C-LOOK"
        ],
        index=0
    )
    
    with st.expander("Configure Disk Requests", expanded=True):
        col1, col2 = st.columns([3, 1])
        
        with col2:
            num_req = st.number_input(
                "Number of Requests",
                min_value=1,
                max_value=20,
                value=st.session_state.num_disk_req,
                step=1,
                key="disk_req_count_input"
            )
            
            if num_req != st.session_state.num_disk_req:
                st.session_state.num_disk_req = num_req
                st.session_state.disk_requests = ', '.join(str(random.randint(0, 199)) for _ in range(st.session_state.num_disk_req))
                st.rerun()
            
            initial_pos = st.number_input(
                "Initial Head Position",
                min_value=0,
                max_value=199,
                value=st.session_state.initial_position,
                step=1,
                key="initial_pos_input"
            )
            
            if initial_pos != st.session_state.initial_position:
                st.session_state.initial_position = initial_pos
            
            disk_size = st.number_input(
                "Disk Size (Cylinders)",
                min_value=100,
                max_value=500,
                value=200,
                step=10,
                key="disk_size_input"
            )
            
            if algo in ["SCAN (Elevator)", "LOOK"]:
                direction = st.selectbox(
                    "Initial Direction",
                    ["Up", "Down"],
                    index=0,
                    key="direction_input"
                )
        
        with col1:
            st.markdown("""
            **Enter disk requests as comma-separated cylinder numbers**
            - Example: `98, 183, 37, 122, 14, 124, 65, 67`
            - Each number represents a cylinder to access
            - Numbers should be between 0 and disk size
            """)
            
            requests = st.text_area(
                "Disk Requests",
                value=st.session_state.disk_requests,
                height=150,
                help="Enter comma-separated cylinder numbers",
                key="disk_requests_input"
            )
            
            st.session_state.disk_requests = requests
    
    col1, col2 = st.columns([1, 4])
    with col1:
        disk_simulate = st.button("▶️ Simulate", use_container_width=True, type='secondary')
    
    if disk_simulate:
        with st.spinner("Running disk scheduling simulation..."):
            request_list = parse_disk_textbox(requests)
            
            if not request_list:
                st.error("Please enter valid disk requests")
            else:
                if algo == "FCFS (First Come First Serve)":
                    sequence, total_movement = fcfs_disk_scheduling(request_list, initial_pos)
                elif algo == "SSTF (Shortest Seek Time First)":
                    sequence, total_movement = sstf_disk_scheduling(request_list, initial_pos)
                elif algo == "SCAN (Elevator)":
                    sequence, total_movement = scan_disk_scheduling(
                        request_list, 
                        initial_pos, 
                        disk_size, 
                        direction.lower()
                    )
                elif algo == "C-SCAN (Circular SCAN)":
                    sequence, total_movement = c_scan_disk_scheduling(
                        request_list, 
                        initial_pos, 
                        disk_size
                    )
                elif algo == "LOOK":
                    sequence, total_movement = look_disk_scheduling(
                        request_list, 
                        initial_pos, 
                        disk_size, 
                        direction.lower()
                    )
                elif algo == "C-LOOK":
                    sequence, total_movement = c_look_disk_scheduling(
                        request_list, 
                        initial_pos, 
                        disk_size
                    )
                
                display_disk_results(
                    request_list, 
                    initial_pos, 
                    sequence, 
                    total_movement, 
                    algo.split()[0]
                )
                st.success(f"Disk scheduling completed successfully!")

# Welcome Screen
else:
    st.markdown("""
    <div style="text-align:center;padding:4rem 0">
        <h2 style="font-size:1.8rem;margin-bottom:1rem">Welcome to OS Algorithm Simulator</h2>
        <p style="font-size:1.2rem;color:#666;margin-bottom:2rem">
            Please select a category from the dropdown above to get started.
        </p>
        <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap">
            <div style="text-align:center;padding:1.5rem;border:1px solid #ddd;border-radius:12px;width:220px">
                <div style="font-size:2.5rem;margin-bottom:1rem">🖥️</div>
                <h3 style="margin-bottom:0.5rem">CPU Scheduling</h3>
                <p style="color:#666;font-size:0.9rem">FCFS, SJF, SRTF, Priority, Round Robin</p>
            </div>
            <div style="text-align:center;padding:1.5rem;border:1px solid #ddd;border-radius:12px;width:220px">
                <div style="font-size:2.5rem;margin-bottom:1rem">💾</div>
                <h3 style="margin-bottom:0.5rem">Memory Management</h3>
                <p style="color:#666;font-size:0.9rem">First, Best, Worst Fit</p>
            </div>
            <div style="text-align:center;padding:1.5rem;border:1px solid #ddd;border-radius:12px;width:220px">
                <div style="font-size:2.5rem;margin-bottom:1rem">💽</div>
                <h3 style="margin-bottom:0.5rem">Disk Scheduling</h3>
                <p style="color:#666;font-size:0.9rem">FCFS, SSTF, SCAN, C-SCAN, LOOK</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown('---')
st.markdown("""
    <div style="text-align:center;color:#666;font-size:0.8rem">
        <p>OS Algorithm Simulator 💻 - CPU Scheduling | Memory Management | Disk Scheduling</p>
    </div>
""", unsafe_allow_html=True)