# CD must be the docs directory
sphinx-apidoc --module-first --remove-old -f --templatedir=source/_templates -f  --remove-old -o source/generated ../ramannoodle
