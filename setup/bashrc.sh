
function act() {
	if [[ -f .venv/bin/activate ]];
	then source .venv/bin/activate
	fi
}

cd() {
	builtin cd $1
	act
}

