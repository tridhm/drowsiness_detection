with open('advanced_drowsiness_detection.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find all FSM block starts
fsm_starts = []
for i, line in enumerate(lines):
    if '# ====================== FSM & SIGNAL PROCESSING ======================' in line:
        fsm_starts.append(i)

print('FSM blocks at lines:', [s+1 for s in fsm_starts])

# Find STATE VARIABLES
state_vars_line = None
for i, line in enumerate(lines):
    if '# ====================== STATE VARIABLES ======================' in line:
        state_vars_line = i
        break

print('STATE VARIABLES at line:', state_vars_line+1 if state_vars_line else 'not found')

if len(fsm_starts) > 1 and state_vars_line:
    first_fsm = fsm_starts[0]
    second_fsm = fsm_starts[1]
    
    # Take: before first FSM + first FSM block only + STATE VARIABLES onwards
    new_content = lines[:first_fsm] + lines[first_fsm:second_fsm] + lines[state_vars_line:]
    
    with open('advanced_drowsiness_detection.py', 'w', encoding='utf-8') as f:
        f.writelines(new_content)
    print('Reduced from', len(lines), 'to', len(new_content), 'lines')
else:
    print('No cleanup needed')
