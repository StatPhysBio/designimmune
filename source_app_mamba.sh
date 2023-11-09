__conda_setup="$('/opt/minimamba/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/minimamba/etc/profile.d/conda.sh" ]; then
        . "/opt/minimamba/etc/profile.d/conda.sh"
    else
        export PATH="/opt/minimamba/bin:$PATH"
    fi
fi
unset __conda_setup

if [ -f "/opt/minimamba/etc/profile.d/mamba.sh" ]; then
    . "/opt/minimamba/etc/profile.d/mamba.sh"
fi