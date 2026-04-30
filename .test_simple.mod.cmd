savedcmd_test_simple.mod := printf '%s\n'   test_simple.o | awk '!x[$$0]++ { print("./"$$0) }' > test_simple.mod
