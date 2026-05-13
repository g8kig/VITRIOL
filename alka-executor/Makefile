# Alka Executor Makefile
# Builds the userspace stream executor and GGUF offset resolver

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c11
LDFLAGS = -lrt

TARGETS = alka-executor gguf-offset-resolver

.PHONY: all clean test

all: $(TARGETS)

alka-executor: executor.c vitriol_alka_user.h
	$(CC) $(CFLAGS) -o $@ executor.c $(LDFLAGS)

gguf-offset-resolver: gguf-offset-resolver.c
	$(CC) $(CFLAGS) -o $@ gguf-offset-resolver.c

clean:
	rm -f $(TARGETS)

test: alka-executor
	@echo "=== Dry-run: stream_960 ==="
	./alka-executor ../alka-handoff/stream_960.alka.alkas ../alka-handoff/gtx960_2gb.alkavl --dry-run
	@echo ""
	@echo "=== Dry-run: purify_1070ti ==="
	./alka-executor ../alka-handoff/purify_1070ti.alka.alkas ../alka-handoff/ivyb_pascal.alkavl --dry-run

install: $(TARGETS)
	install -m 755 $(TARGETS) /usr/local/bin/
