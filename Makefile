.PHONY: all build-kabukinai test clean

all: build-kabukinai

build-kabukinai: build/ build/Makefile
	make -C $<

build/:
	mkdir -p $@

build/Makefile: build/ src/ test/
	(cd $< ; cmake ..)

test: build/ build/Makefile src/ test/
	make -C $<
	make -C $< test

clean:
	rm -r build/
