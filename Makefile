.PHONY: all build-kabukinai test_kabukinai test clean

CMAKE3_PRESENT := $(shell command -v cmake3 2> /dev/null)

all: build-kabukinai

test_kabukinai: build/ build/Makefile
	make -C build test_kabukinai

build-kabukinai: build/ build/Makefile
	make -C $<

build/:
	mkdir -p $@

build/Makefile: build/ src/ test/
ifdef CMAKE3_PRESENT
	(cd $< ; cmake3 ..)
else
	(cd $< ; cmake ..)
endif

test: build/ build/Makefile src/ test/
	make -C $<
	make -C $< test

clean:
	rm -r build/
