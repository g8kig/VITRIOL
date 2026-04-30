savedcmd_vitriol.mod := printf '%s\n'   vitriol.o | awk '!x[$$0]++ { print("./"$$0) }' > vitriol.mod
