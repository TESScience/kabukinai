.PHONY: all build-kabukinai test_kabukinai test clean

all: build-kabukinai

test_kabukinai: build/ build/Makefile
	make -C build test_kabukinai

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
