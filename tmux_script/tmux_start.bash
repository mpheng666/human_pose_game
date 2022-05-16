#!/bin/bash
session="movelei_tmux"

tmux has-session -t $session
if [ $? != 0 ]; then
    # create new tmux session 
    tmux new-session -d -s $session

    tmux set -g mouse on

    tmux set -g status-bg black
    tmux set -g status-fg white

    # roscore
    window=0
    tmux rename-window -t $session:$window 'workspace'
    tmux send-keys -t $session:$window 'cd ~/movelei' C-m
    tmux send-keys -t $session:$window 'roscore' C-m

    tmux split-window -v
    tmux send-keys -t $session:$window 'cd ~/movelei/demo/; python3 movelei_demo.py ' C-m
    tmux select-layout tiled

    tmux split-window -h
    tmux send-keys -t $session:$window 'sleep 5; cd ~/movelei/pygame/examples/; python3 aliens.py ' C-m
    tmux select-layout tiled

    tmux split-window -h
    tmux send-keys -t $session:$window 'sleep 5 ;' C-m
    tmux select-layout tiled

    tmux split-window -h
    tmux send-keys -t $session:$window 'sleep 6 ;' C-m
    tmux select-layout tiled

fi

tmux attach-session -t $session